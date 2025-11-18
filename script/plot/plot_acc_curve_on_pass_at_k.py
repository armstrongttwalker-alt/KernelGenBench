#!/usr/bin/env python3
"""
Script to calculate pass@k statistics and plot accuracy curves.

This script analyzes evaluation results from multiple sample runs,
computes the pass@k success rate for different k values (1 to 10),
and generates visualization plots.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import re


def load_test_reports(eval_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all test reports from log directories.
    
    Args:
        eval_dir: Path to the evaluation directory containing log_0 to log_9
        
    Returns:
        Dictionary mapping operator names to list of test results across samples
    """
    op_results = {}
    
    # Iterate through log_0 to log_9 directories
    for i in range(10):
        log_dir = eval_dir / f"log_{i}"
        if not log_dir.exists():
            print(f"Warning: {log_dir} does not exist, skipping...")
            continue
        
        # Load result.json which contains the overall success status
        result_file = log_dir / "result.json"
        if not result_file.exists():
            print(f"Warning: {result_file} does not exist, skipping...")
            continue
        
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # result.json is a list of operator results
            if not isinstance(result_data, list):
                print(f"Warning: {result_file} is not a list, skipping...")
                continue
            
            for op_result in result_data:
                if not isinstance(op_result, dict):
                    continue
                
                op_name = op_result.get('op_name')
                if not op_name:
                    continue
                
                # Initialize list for this operator if not exists
                if op_name not in op_results:
                    op_results[op_name] = []
                
                # Use the pre-calculated success value from result.json
                success = op_result.get('success', False)
                info = op_result.get('info', {})
                traceback = op_result.get('traceback')
                
                # Append the result for this sample
                op_results[op_name].append({
                    'sample_id': i,
                    'success': success,
                    'info': info,
                    'traceback': traceback
                })
                
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    return op_results


def classify_error(traceback: str) -> str:
    """
    Classify error based on traceback message.
    
    Args:
        traceback: Error traceback string
        
    Returns:
        Error type classification
    """
    if not traceback:
        return 'other'
    
    traceback_lower = traceback.lower()
    
    # Check error patterns in order of specificity
    if 'no func' in traceback_lower and 'valueerror' in traceback_lower:
        return 'no_func'
    
    if 'has no attribute' in traceback_lower or 'attributeerror' in traceback_lower:
        return 'no_attribute'
    
    if 'triton.compiler.errors.compilationerror' in traceback_lower or \
       'compilationerror' in traceback_lower:
        return 'triton_compilation_error'
    
    if 'assertionerror' in traceback_lower and \
       ('are not close' in traceback_lower or 'are not equal' in traceback_lower or \
        'mismatch' in traceback_lower):
        return 'output_mismatch'
    
    if 'not supported' in traceback_lower or 'unsupported' in traceback_lower:
        return 'unsupported_language_construct'
    
    if 'assertionerror' in traceback_lower:
        return 'assertion_error'
    
    if 'is not defined' in traceback_lower or 'nameerror' in traceback_lower:
        return 'not_defined'
    
    if 'unexpected keyword argument' in traceback_lower or \
       'got an unexpected keyword argument' in traceback_lower:
        return 'unexpected_keyword_argument'
    
    if 'syntaxerror' in traceback_lower or 'indentationerror' in traceback_lower:
        return 'compilation_error'
    
    if 'not enough values to unpack' in traceback_lower:
        return 'not_enough_values_to_unpack'
    
    if 'only.*fp16.*supported' in traceback_lower or \
       'fp16.*only' in traceback_lower:
        return 'only_fp16_supported'
    
    return 'other'


def calculate_pass_at_k(op_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[int, float]]:
    """
    Calculate pass@k for each operator.
    log_0 evaluates all operators, log_i re-evaluates previous failures.
    Once an operator succeeds, it's not tested in subsequent logs.
    Pass@k means: with k attempts (log_0 to log_(k-1)), did it succeed at least once?
    
    Args:
        op_results: Dictionary mapping operator names to list of test results
        
    Returns:
        Dictionary mapping operator names to dict of {k: success_rate}
    """
    pass_at_k_stats = {}
    
    for op_name, results in op_results.items():
        # Sort results by sample_id to ensure consistent ordering
        results = sorted(results, key=lambda x: x['sample_id'])
        
        # Build a complete picture: which iteration did it first succeed?
        first_success_at = None
        for i, result in enumerate(results):
            if result['success']:
                first_success_at = result['sample_id']
                break
        
        k_stats = {}
        # For k=1 to 10, check if succeeded within first k attempts
        for k in range(1, 11):
            if first_success_at is not None and first_success_at < k:
                # Succeeded at iteration first_success_at, which is < k
                k_stats[k] = 1.0
            else:
                # Either never succeeded, or succeeded at iteration >= k
                k_stats[k] = 0.0
        
        pass_at_k_stats[op_name] = k_stats
    
    return pass_at_k_stats


def calculate_error_type_stats(op_results: Dict[str, List[Dict[str, Any]]]) -> Dict[int, Dict[str, int]]:
    """
    Calculate error type statistics for each pass@k.
    For log_x (x>0), we only count errors that were fixed in that iteration.
    
    Args:
        op_results: Dictionary mapping operator names to list of test results
        
    Returns:
        Dictionary mapping k to error type counts at that k
    """
    error_types = [
        'no_func',
        'no_attribute',
        'triton_compilation_error', 
        'output_mismatch',
        'unsupported_language_construct',
        'assertion_error',
        'not_defined',
        'unexpected_keyword_argument',
        'compilation_error',
        'not_enough_values_to_unpack',
        'only_fp16_supported',
        'other'
    ]
    
    # Initialize error counts for k=1 to 10
    error_stats = {k: {error_type: 0 for error_type in error_types} for k in range(1, 11)}
    
    for op_name, results in op_results.items():
        results = sorted(results, key=lambda x: x['sample_id'])
        
        # Track the initial error (from log_0) for this operator
        initial_error_type = None
        if results and not results[0]['success']:
            initial_error_type = classify_error(results[0].get('traceback', ''))
        
        # Find when it first succeeded
        first_success_at = None
        for result in results:
            if result['success']:
                first_success_at = result['sample_id']
                break
        
        # For each k, count this operator's error if it hasn't succeeded yet
        for k in range(1, 11):
            # Check if operator failed at all attempts < k
            failed_at_k = True
            for result in results:
                if result['sample_id'] < k and result['success']:
                    failed_at_k = False
                    break
            
            if failed_at_k and initial_error_type:
                # If it succeeded at iteration k-1, don't count it for k
                if first_success_at is not None and first_success_at < k:
                    pass  # Error was fixed before k
                else:
                    error_stats[k][initial_error_type] += 1
    
    return error_stats


def calculate_overall_pass_at_k(pass_at_k_stats: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    """
    Calculate overall pass@k across all operators.
    
    Args:
        pass_at_k_stats: Per-operator pass@k statistics
        
    Returns:
        Dictionary mapping k to overall success rate
    """
    overall_stats = {}
    
    if not pass_at_k_stats:
        return overall_stats
    
    # Get all k values (should be 1 to 10)
    all_k_values = set()
    for op_stats in pass_at_k_stats.values():
        all_k_values.update(op_stats.keys())
    
    for k in sorted(all_k_values):
        # Calculate success rate across all operators for this k
        success_count = sum(
            1 for op_stats in pass_at_k_stats.values() 
            if op_stats.get(k, 0.0) > 0
        )
        total_count = len(pass_at_k_stats)
        overall_stats[k] = success_count / total_count if total_count > 0 else 0.0
    
    return overall_stats


def plot_pass_at_k_curve(overall_stats: Dict[int, float], 
                         error_stats: Dict[int, Dict[str, int]],
                         output_path: Path):
    """
    Plot pass@k curve with error type analysis and save to file.
    
    Args:
        overall_stats: Dictionary mapping k to success rate
        error_stats: Dictionary mapping k to error type counts
        output_path: Path to save the plot
    """
    if not overall_stats:
        print("No data to plot")
        return
    
    # Error type descriptions
    error_descriptions = {
        'no_func': 'No Function in Code',
        'no_attribute': 'Missing Attribute',
        'triton_compilation_error': 'Triton Compilation Error',
        'output_mismatch': 'Output Mismatch',
        'unsupported_language_construct': 'Unsupported Language Construct',
        'assertion_error': 'Assertion Error',
        'not_defined': 'Not Defined Error',
        'unexpected_keyword_argument': 'Unexpected Keyword Argument',
        'compilation_error': 'Compilation Error',
        'not_enough_values_to_unpack': 'Not Enough Values to Unpack',
        'only_fp16_supported': 'Only FP16 Supported',
        'other': 'Other Errors'
    }
    
    # Prepare data
    k_values = sorted(overall_stats.keys())
    success_rates = [overall_stats[k] * 100 for k in k_values]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Pass@k curve
    ax1.plot(k_values, success_rates, marker='o', linewidth=2.5, markersize=8, 
             label='Overall Pass@k', color='#2E86AB', zorder=3)
    
    # Add value labels on points
    for k, rate in zip(k_values, success_rates):
        ax1.text(k, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('k (Number of Attempts)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Pass@k Accuracy Curve', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_xlim(0.5, max(k_values) + 0.5)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(k_values)
    
    # Plot 2: Error type distribution
    # Select top error types to plot (exclude 'other' or include it based on count)
    error_type_order = ['triton_compilation_error', 'output_mismatch', 'no_func', 'no_attribute',
                       'unsupported_language_construct', 'assertion_error', 'other']
    
    colors = ['#E63946', '#F77F00', '#FF6B9D', '#FCBF49', '#06A77D', '#457B9D', '#999999']
    
    for idx, error_type in enumerate(error_type_order):
        error_counts = [error_stats[k].get(error_type, 0) for k in k_values]
        if sum(error_counts) > 0:  # Only plot if there are errors of this type
            ax2.plot(k_values, error_counts, marker='s', linewidth=2, markersize=6,
                    label=error_descriptions.get(error_type, error_type),
                    color=colors[idx % len(colors)], alpha=0.8)
    
    ax2.set_xlabel('k (Number of Attempts)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error Count', fontsize=12, fontweight='bold')
    ax2.set_title('Error Type Distribution across Pass@k', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    ax2.set_xlim(0.5, max(k_values) + 0.5)
    ax2.set_xticks(k_values)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Close figure to free memory
    plt.close()


def save_statistics(pass_at_k_stats: Dict[str, Dict[int, float]], 
                   overall_stats: Dict[int, float],
                   error_stats: Dict[int, Dict[str, int]],
                   output_path: Path):
    """
    Save pass@k statistics to JSON file.
    
    Args:
        pass_at_k_stats: Per-operator statistics
        overall_stats: Overall statistics
        error_stats: Error type statistics
        output_path: Path to save the JSON file
    """
    output_data = {
        'overall': overall_stats,
        'error_statistics': error_stats,
        'per_operator': pass_at_k_stats,
        'summary': {
            'total_operators': len(pass_at_k_stats),
            'k_values': sorted(overall_stats.keys()) if overall_stats else []
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Statistics saved to {output_path}")


def print_summary(overall_stats: Dict[int, float]):
    """
    Print summary statistics to console.
    
    Args:
        overall_stats: Overall pass@k statistics
    """
    if not overall_stats:
        print("No statistics to display")
        return
    
    print("\n" + "="*60)
    print("Pass@k Summary Statistics")
    print("="*60)
    
    for k in sorted(overall_stats.keys()):
        rate = overall_stats[k] * 100
        print(f"Pass@{k:2d}: {rate:6.2f}%")
    
    print("="*60 + "\n")


def main():
    """Main function to run the pass@k analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate pass@k statistics and plot accuracy curves from evaluation results."
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Path to the evaluation directory containing log_0 to log_9 subdirectories"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as eval_dir)"
    )
    
    args = parser.parse_args()
    
    # Set the evaluation directory
    eval_dir = Path(args.eval_dir)
    
    if not eval_dir.exists():
        print(f"Error: Evaluation directory {eval_dir} does not exist")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test reports from {eval_dir}...")
    
    # Load all test reports
    op_results = load_test_reports(eval_dir)
    print(f"Loaded results for {len(op_results)} operators")
    
    if not op_results:
        print("No test results found!")
        return 1
    
    # Calculate pass@k for each operator
    print("Calculating pass@k statistics...")
    pass_at_k_stats = calculate_pass_at_k(op_results)
    
    # Calculate overall pass@k
    overall_stats = calculate_overall_pass_at_k(pass_at_k_stats)
    
    # Calculate error type statistics
    print("Calculating error type statistics...")
    error_stats = calculate_error_type_stats(op_results)
    
    # Print summary
    print_summary(overall_stats)
    
    # Save statistics
    stats_output = output_dir / "pass_at_k_summary.json"
    save_statistics(pass_at_k_stats, overall_stats, error_stats, stats_output)
    
    # Plot and save curve
    plot_output = output_dir / "pass_at_k_curve.png"
    plot_pass_at_k_curve(overall_stats, error_stats, plot_output)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
