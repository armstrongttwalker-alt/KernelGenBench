#!/usr/bin/env python3
"""
Script to visualize speedup test results.

This script analyzes speedup test results from performance evaluation runs,
extracts average speedup for each operator, and creates bar chart visualizations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_speedup_results(eval_dir: Path) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Load speedup results from all log directories.
    
    Args:
        eval_dir: Path to the evaluation directory containing log_0 to log_9
        
    Returns:
        Tuple of (successful_ops_dict, failed_ops_list)
        - successful_ops_dict: Mapping from operator name to average speedup
        - failed_ops_list: List of operators that passed acc test but failed perf test
    """
    successful_ops = {}
    failed_ops = []
    paths = [p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("log_")]
    # Iterate through log_0 to log_9 directories
    for p in paths:
        log_dir = p
        if not log_dir.exists():
            continue
        
        result_file = log_dir / "result.json"
        if not result_file.exists():
            continue
        
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            if not isinstance(result_data, list):
                continue
            
            for op_result in result_data:
                if not isinstance(op_result, dict):
                    continue
                
                op_name = op_result.get('op_name')
                if not op_name:
                    continue
                
                success = op_result.get('success')
                speedup_data = op_result.get('speedup')
                
                # Check if this operator was tested in this log
                # An operator is considered "tested" if it has either:
                # 1. speedup data (success case)
                # 2. success=False with traceback (failure case)
                has_speedup = speedup_data is not None
                has_failure = success is not False
                
                if not has_speedup and not has_failure:
                    # Operator not tested in this log (success=None, no speedup, no traceback)
                    continue
                
                if success:
                    # Extract average speedup (last item in speedup list)
                    if isinstance(speedup_data, list) and len(speedup_data) > 0:
                        last_item = speedup_data[-1]
                        if isinstance(last_item, dict) and 'speedup' in last_item:
                            avg_speedup = last_item['speedup']
                            successful_ops[op_name] = avg_speedup
                elif success is False:
                    # Operator passed acc test but failed perf test
                    traceback = op_result.get('traceback', 'No traceback available')
                    failed_ops.append({
                        'op_name': op_name,
                        'log_id': log_dir.name,
                        'traceback': traceback
                    })
        
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    print(f"Total successful operators loaded: {len(successful_ops)}")
    print(f"Total failed operators loaded: {len(failed_ops)}")
    return successful_ops, failed_ops


def plot_speedup_chart(speedup_dict: Dict[str, float], output_path: Path):
    """
    Create a bar chart of speedup results.
    
    Args:
        speedup_dict: Dictionary mapping operator names to average speedup
        output_path: Path to save the plot
    """
    if not speedup_dict:
        print("No speedup data to plot")
        return
    
    # Sort operators by speedup (descending)
    sorted_ops = sorted(speedup_dict.items(), key=lambda x: x[1], reverse=True)
    op_names = [op[0] for op in sorted_ops]
    speedups = [op[1] for op in sorted_ops]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(op_names) * 0.3)))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(op_names))
    colors = ['#2ECC71' if s >= 1.0 else '#E74C3C' for s in speedups]
    
    bars = ax.barh(y_pos, speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        label_x = width + 0.05 if width >= 0 else width - 0.05
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{speedup:.3f}x',
                ha=ha, va='center', fontsize=8, fontweight='bold')
    
    # Add vertical line at x=1.0 (baseline)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Baseline (1.0x)')
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(op_names, fontsize=9)
    ax.set_xlabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Operator Speedup Comparison', fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')
    
    # Set x-axis limits with some padding
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    x_margin = (max_speedup - min_speedup) * 0.1
    ax.set_xlim(min(0, min_speedup - x_margin), max_speedup + x_margin)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Speedup chart saved to {output_path}")
    
    # Close figure to free memory
    plt.close()


def save_summary(speedup_dict: Dict[str, float], failed_ops: List[Dict], output_path: Path):
    """
    Save summary statistics to JSON file.
    
    Args:
        speedup_dict: Dictionary mapping operator names to average speedup
        failed_ops: List of operators that failed perf test
        output_path: Path to save the JSON file
    """
    speedups = list(speedup_dict.values())
    
    summary = {
        'statistics': {
            'total_operators': len(speedup_dict),
            'avg_speedup': float(np.mean(speedups)) if speedups else 0.0,
            'median_speedup': float(np.median(speedups)) if speedups else 0.0,
            'max_speedup': float(max(speedups)) if speedups else 0.0,
            'min_speedup': float(min(speedups)) if speedups else 0.0,
            'speedup_gt_1': sum(1 for s in speedups if s >= 1.0),
            'speedup_lt_1': sum(1 for s in speedups if s < 1.0),
        },
        'successful_operators': speedup_dict,
        'failed_operators': failed_ops,
        'failed_count': len(failed_ops)
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary statistics saved to {output_path}")


def print_summary(speedup_dict: Dict[str, float], failed_ops: List[Dict]):
    """
    Print summary statistics to console.
    
    Args:
        speedup_dict: Dictionary mapping operator names to average speedup
        failed_ops: List of operators that failed perf test
    """
    speedups = list(speedup_dict.values())
    
    print("\n" + "="*80)
    print("Speedup Test Summary")
    print("="*80)
    
    if speedups:
        print(f"Total successful operators: {len(speedup_dict)}")
        print(f"Average speedup: {np.mean(speedups):.4f}x")
        print(f"Median speedup: {np.median(speedups):.4f}x")
        print(f"Max speedup: {max(speedups):.4f}x")
        print(f"Min speedup: {min(speedups):.4f}x")
        print(f"Operators with speedup >= 1.0x: {sum(1 for s in speedups if s >= 1.0)}")
        print(f"Operators with speedup < 1.0x: {sum(1 for s in speedups if s < 1.0)}")
    else:
        print("No successful operators found")
    
    print(f"\nOperators that passed acc test but failed perf test: {len(failed_ops)}")
    if failed_ops:
        print("\nFailed operators:")
        for op in failed_ops:
            print(f"  - {op['op_name']} (log_{op['log_id']})")
    
    print("="*80 + "\n")


def main():
    """Main function to run the speedup visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize speedup test results from performance evaluation."
    )
    parser.add_argument(
        "--eval_dir",
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
    
    print(f"Loading speedup results from {eval_dir}...")
    
    # Load speedup results
    speedup_dict, failed_ops = load_speedup_results(eval_dir)
    print(f"Loaded results for {len(speedup_dict)} successful operators")
    print(f"Found {len(failed_ops)} operators that failed perf test")
    
    if not speedup_dict and not failed_ops:
        print("No speedup test results found!")
        return 1
    
    # Print summary
    print_summary(speedup_dict, failed_ops)
    
    # Save summary
    summary_output = output_dir / "speedup_summary.json"
    save_summary(speedup_dict, failed_ops, summary_output)
    
    # Plot speedup chart
    if speedup_dict:
        plot_output = output_dir / "speedup_chart.png"
        plot_speedup_chart(speedup_dict, plot_output)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
