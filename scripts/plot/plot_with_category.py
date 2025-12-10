#!/usr/bin/env python3
"""
Script to analyze evaluation results by API category and plot accuracy curves.

This script:
1. Reads result.json files from each evaluation round (log_X directories)
2. Categorizes operators based on api_categories.json
3. Computes per-category and overall accuracy statistics
4. Generates visualization plots showing accuracy by category across rounds
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import matplotlib.pyplot as plt
import numpy as np


def load_api_categories(categories_path: Path) -> Dict[str, Set[str]]:
    """
    Load API categories from JSON file.
    
    Args:
        categories_path: Path to api_categories.json
        
    Returns:
        Dictionary mapping category name to set of operator names
    """
    with open(categories_path, 'r') as f:
        data = json.load(f)
    
    category_to_ops = {}
    for category_name, category_info in data.get('categories', {}).items():
        ops = category_info.get('ops', [])
        # Use set for faster lookup, handle duplicates in the list
        category_to_ops[category_name] = set(ops)
    
    return category_to_ops


def get_category_for_op(op_name: str, category_to_ops: Dict[str, Set[str]]) -> str:
    """
    Get the category for a given operator.
    
    Args:
        op_name: Operator name
        category_to_ops: Category to operators mapping
        
    Returns:
        Category name or 'uncategorized' if not found
    """
    for category, ops in category_to_ops.items():
        if op_name in ops:
            return category
    return 'uncategorized'


def load_results_from_log(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load results from a single log directory's result.json.
    
    Args:
        log_dir: Path to log_X directory
        
    Returns:
        Dictionary mapping operator name to result info
    """
    result_file = log_dir / "result.json"
    if not result_file.exists():
        return {}
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        if not isinstance(result_data, list):
            return {}
        
        op_results = {}
        for op_result in result_data:
            if not isinstance(op_result, dict):
                continue
            
            op_name = op_result.get('op_name')
            if not op_name:
                continue
            
            op_results[op_name] = {
                'success': op_result.get('success', False),
                'info': op_result.get('info', {}),
                'traceback': op_result.get('traceback')
            }
        
        return op_results
        
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return {}


def load_all_round_results(eval_dir: Path) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], List[int]]:
    """
    Load results from all rounds (log_0 to log_N).
    
    Args:
        eval_dir: Path to the verification directory
        
    Returns:
        Tuple of (round_results, round_indices)
        - round_results: Dict mapping round index to op results
        - round_indices: Sorted list of available round indices
    """
    round_results = {}
    
    # Find all log_X directories
    log_dirs = list(eval_dir.glob("log_*"))
    round_indices = []
    
    for log_dir in log_dirs:
        if log_dir.is_dir():
            try:
                round_idx = int(log_dir.name.split('_')[1])
                results = load_results_from_log(log_dir)
                if results:
                    round_results[round_idx] = results
                    round_indices.append(round_idx)
            except (ValueError, IndexError):
                continue
    
    round_indices = sorted(round_indices)
    return round_results, round_indices


def compute_category_stats(
    round_results: Dict[int, Dict[str, Dict[str, Any]]],
    round_indices: List[int],
    category_to_ops: Dict[str, Set[str]]
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, float], Set[str]]:
    """
    Compute accuracy statistics by category for each round.
    
    For pass@k semantics:
    - log_0 evaluates all operators
    - log_i (i>0) re-evaluates previous failures
    - Once an operator succeeds, it's considered successful for all subsequent rounds
    
    Args:
        round_results: Results from each round
        round_indices: List of round indices
        category_to_ops: Category to operators mapping
        
    Returns:
        Tuple of:
        - category_stats: Dict[category][round] -> accuracy (0-1)
        - overall_stats: Dict[round] -> accuracy (0-1)
        - all_ops: Set of all operator names evaluated
    """
    # First, collect all operators from round 0 (which has all operators)
    if 0 not in round_results:
        print("Warning: log_0 not found, using first available round as baseline")
        first_round = round_indices[0] if round_indices else 0
    else:
        first_round = 0
    
    all_ops = set(round_results.get(first_round, {}).keys())
    
    # Track cumulative success for each operator (pass@k semantics)
    # An operator is successful at round k if it succeeded at any round <= k
    op_success_by_round = {op: {} for op in all_ops}
    
    for round_idx in round_indices:
        round_data = round_results.get(round_idx, {})
        
        for op_name in all_ops:
            # Check if succeeded in this round
            if op_name in round_data:
                current_success = round_data[op_name]['success']
            else:
                # Not tested in this round (already succeeded or skipped)
                current_success = False
            
            # Check if already succeeded in previous rounds
            prev_success = False
            for prev_round in round_indices:
                if prev_round >= round_idx:
                    break
                if op_success_by_round[op_name].get(prev_round, False):
                    prev_success = True
                    break
            
            # Cumulative success: succeeded now OR previously
            op_success_by_round[op_name][round_idx] = current_success or prev_success
    
    # Now compute per-category and overall statistics
    category_stats = {}
    overall_stats = {}
    
    # Group operators by category
    ops_by_category = {}
    for op_name in all_ops:
        category = get_category_for_op(op_name, category_to_ops)
        if category not in ops_by_category:
            ops_by_category[category] = []
        ops_by_category[category].append(op_name)
    
    # Compute accuracy for each category and round
    for category, ops in ops_by_category.items():
        if category not in category_stats:
            category_stats[category] = {}
        
        for round_idx in round_indices:
            successes = sum(1 for op in ops if op_success_by_round[op].get(round_idx, False))
            accuracy = successes / len(ops) if ops else 0.0
            category_stats[category][round_idx] = accuracy
    
    # Compute overall accuracy
    for round_idx in round_indices:
        successes = sum(1 for op in all_ops if op_success_by_round[op].get(round_idx, False))
        accuracy = successes / len(all_ops) if all_ops else 0.0
        overall_stats[round_idx] = accuracy
    
    return category_stats, overall_stats, all_ops


def plot_category_accuracy(
    category_stats: Dict[str, Dict[int, float]],
    overall_stats: Dict[int, float],
    round_indices: List[int],
    output_path: Path,
    title_suffix: str = "",
    top_n: int = 12
):
    """
    Plot accuracy curves by category.
    
    Args:
        category_stats: Per-category accuracy statistics
        overall_stats: Overall accuracy statistics
        round_indices: List of round indices
        output_path: Path to save the plot
        title_suffix: Optional suffix for the title
        top_n: Number of top categories to show in the curve plot
    """
    if not round_indices:
        print("No data to plot")
        return
    
    # Color palette for categories
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Sort categories by final accuracy (descending) for better visualization
    sorted_categories = sorted(
        category_stats.keys(),
        key=lambda c: category_stats[c].get(round_indices[-1], 0),
        reverse=True
    )
    
    # Filter out categories with very few operators or no data
    # Keep categories with at least 1 operator and meaningful accuracy changes
    significant_categories = []
    for cat in sorted_categories:
        if cat in ['default', 'uncategorized', 'inplace', 'high_order', 'non_tensor', 'sparse']:
            continue  # Skip less interesting categories
        accuracies = [category_stats[cat].get(r, 0) for r in round_indices]
        if max(accuracies) > 0:  # Has at least some success
            significant_categories.append(cat)
    
    # Take top N categories for main plot
    top_categories = significant_categories[:top_n]
    
    # Plot 1: Top category accuracy curves
    x_values = [r + 1 for r in round_indices]  # Convert to 1-indexed for display
    
    for idx, category in enumerate(top_categories):
        accuracies = [category_stats[category].get(r, 0) * 100 for r in round_indices]
        ax1.plot(x_values, accuracies, marker='o', linewidth=2, markersize=6,
                 label=category.replace('_', ' ').title(),
                 color=colors[idx % len(colors)], alpha=0.8)
    
    # Plot overall accuracy
    overall_accuracies = [overall_stats.get(r, 0) * 100 for r in round_indices]
    ax1.plot(x_values, overall_accuracies, marker='s', linewidth=3, markersize=8,
             label='Overall', color='black', linestyle='--', zorder=10)
    
    ax1.set_xlabel('Round (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Accuracy by Category (Top {top_n}){title_suffix}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9, loc='lower right', ncol=3)
    ax1.set_xlim(0.5, max(x_values) + 0.5)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x_values)
    
    # Plot 2: Bar chart showing final accuracy by category
    all_categories = sorted_categories
    final_accuracies = [category_stats[cat].get(round_indices[-1], 0) * 100 
                        for cat in all_categories]
    
    # Color bars by accuracy level
    bar_colors = []
    for acc in final_accuracies:
        if acc >= 80:
            bar_colors.append('#2E86AB')
        elif acc >= 60:
            bar_colors.append('#06A77D')
        elif acc >= 40:
            bar_colors.append('#F77F00')
        elif acc >= 20:
            bar_colors.append('#FCBF49')
        else:
            bar_colors.append('#E63946')
    
    y_pos = np.arange(len(all_categories))
    bars = ax2.barh(y_pos, final_accuracies, color=bar_colors, alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([cat.replace('_', ' ').title() for cat in all_categories], fontsize=9)
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Final Accuracy by Category (Round {round_indices[-1] + 1}){title_suffix}',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        if acc > 0:
            ax2.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                     f'{acc:.1f}%', va='center', fontsize=8)
    
    # Invert y-axis for better readability (highest at top)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_category_heatmap(
    category_stats: Dict[str, Dict[int, float]],
    round_indices: List[int],
    output_path: Path,
    title_suffix: str = ""
):
    """
    Plot a heatmap showing accuracy by category and round.
    
    Args:
        category_stats: Per-category accuracy statistics
        round_indices: List of round indices
        output_path: Path to save the plot
        title_suffix: Optional suffix for the title
    """
    if not round_indices:
        print("No data to plot heatmap")
        return
    
    # Sort categories by average accuracy
    sorted_categories = sorted(
        category_stats.keys(),
        key=lambda c: np.mean([category_stats[c].get(r, 0) for r in round_indices]),
        reverse=True
    )
    
    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(sorted_categories), len(round_indices)))
    for i, cat in enumerate(sorted_categories):
        for j, round_idx in enumerate(round_indices):
            accuracy_matrix[i, j] = category_stats[cat].get(round_idx, 0) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(round_indices) * 0.8), 
                                     max(8, len(sorted_categories) * 0.4)))
    
    # Create heatmap
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(round_indices)))
    ax.set_yticks(np.arange(len(sorted_categories)))
    ax.set_xticklabels([f'Round {r + 1}' for r in round_indices], fontsize=10)
    ax.set_yticklabels([cat.replace('_', ' ').title() for cat in sorted_categories], fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(sorted_categories)):
        for j in range(len(round_indices)):
            value = accuracy_matrix[i, j]
            text_color = 'white' if value < 50 else 'black'
            ax.text(j, i, f'{value:.0f}', ha="center", va="center", 
                    color=text_color, fontsize=8)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom", fontsize=11)
    
    ax.set_title(f'Accuracy Heatmap by Category and Round{title_suffix}',
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    plt.close()


def save_statistics(
    category_stats: Dict[str, Dict[int, float]],
    overall_stats: Dict[int, float],
    round_indices: List[int],
    all_ops: Set[str],
    category_to_ops: Dict[str, Set[str]],
    output_path: Path
):
    """
    Save statistics to JSON file.
    
    Args:
        category_stats: Per-category statistics
        overall_stats: Overall statistics
        round_indices: List of round indices
        all_ops: Set of all operators
        category_to_ops: Category mapping
        output_path: Path to save the JSON file
    """
    # Count operators per category
    ops_by_category_count = {}
    for op_name in all_ops:
        category = get_category_for_op(op_name, category_to_ops)
        ops_by_category_count[category] = ops_by_category_count.get(category, 0) + 1
    
    # Convert sets to counts for JSON serialization
    output_data = {
        'summary': {
            'total_operators': len(all_ops),
            'total_categories': len(category_stats),
            'rounds': round_indices,
        },
        'operators_per_category': ops_by_category_count,
        'overall_accuracy': {str(k): v for k, v in overall_stats.items()},
        'category_accuracy': {
            cat: {str(r): acc for r, acc in rounds.items()}
            for cat, rounds in category_stats.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Statistics saved to {output_path}")


def print_summary(
    category_stats: Dict[str, Dict[int, float]],
    overall_stats: Dict[int, float],
    round_indices: List[int],
    all_ops: Set[str],
    category_to_ops: Dict[str, Set[str]]
):
    """
    Print summary statistics to console.
    """
    print("\n" + "=" * 60)
    print("ACCURACY BY CATEGORY SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal operators evaluated: {len(all_ops)}")
    print(f"Total rounds: {len(round_indices)}")
    print(f"Categories: {len(category_stats)}")
    
    # Overall accuracy progression
    print("\n" + "-" * 40)
    print("Overall Accuracy Progression:")
    print("-" * 40)
    for round_idx in round_indices:
        acc = overall_stats.get(round_idx, 0) * 100
        print(f"  Round {round_idx + 1}: {acc:.1f}%")
    
    # Per-category summary (sorted by final accuracy)
    print("\n" + "-" * 40)
    print("Category Accuracy (Final Round):")
    print("-" * 40)
    
    sorted_categories = sorted(
        category_stats.keys(),
        key=lambda c: category_stats[c].get(round_indices[-1], 0),
        reverse=True
    )
    
    for category in sorted_categories:
        # Count operators in this category
        count = sum(1 for op in all_ops if get_category_for_op(op, category_to_ops) == category)
        final_acc = category_stats[category].get(round_indices[-1], 0) * 100
        print(f"  {category:35} ({count:3} ops): {final_acc:5.1f}%")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results by API category and plot accuracy curves.'
    )
    parser.add_argument(
        'eval_dir',
        type=str,
        help='Path to the evaluation directory containing verification/log_X subdirectories'
    )
    parser.add_argument(
        '--categories',
        type=str,
        default=None,
        help='Path to custom categories JSON file (overrides --categories-type)'
    )
    parser.add_argument(
        '--categories-type',
        type=str,
        default='api',
        choices=['api', 'difficult'],
        help='Type of categories to use: "api" for api_categories.json, "difficult" for difficult_categories.json (default: api)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for plots and statistics (default: same as eval_dir)'
    )
    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help='Skip generating the heatmap plot'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=12,
        help='Number of top categories to show in the accuracy curve plot (default: 12)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    eval_dir = Path(args.eval_dir).resolve()
    
    # Check if verification subdirectory exists
    verification_dir = eval_dir / 'verification'
    if verification_dir.exists():
        log_dir = verification_dir
    else:
        log_dir = eval_dir
    
    # Find categories file
    if args.categories:
        categories_path = Path(args.categories).resolve()
        categories_type = 'custom'
    else:
        # Use categories-type to determine which file to use
        script_dir = Path(__file__).parent
        categories_type = args.categories_type
        
        if categories_type == 'api':
            categories_path = script_dir / 'api_categories.json'
        elif categories_type == 'difficult':
            categories_path = script_dir / 'difficult_categories.json'
        else:
            categories_path = script_dir / 'api_categories.json'
        
        if not categories_path.exists():
            categories_path = eval_dir / f'{categories_type}_categories.json'
    
    if not categories_path.exists():
        print(f"Error: Categories file not found at {categories_path}")
        return
    
    # Set output directory
    output_dir = Path(args.output) if args.output else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output file prefix based on categories type
    output_prefix = f"{categories_type}_" if categories_type != 'api' else ""
    
    print(f"Loading categories from: {categories_path}")
    print(f"Categories type: {categories_type}")
    print(f"Loading results from: {log_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    category_to_ops = load_api_categories(categories_path)
    round_results, round_indices = load_all_round_results(log_dir)
    
    if not round_results:
        print("Error: No results found in log directories")
        return
    
    print(f"\nFound {len(round_indices)} rounds: {round_indices}")
    
    # Compute statistics
    category_stats, overall_stats, all_ops = compute_category_stats(
        round_results, round_indices, category_to_ops
    )
    
    # Generate title suffix from directory name and categories type
    categories_label = f" ({categories_type.title()} Categories)" if categories_type != 'api' else ""
    title_suffix = f"{categories_label} - {eval_dir.name}"
    
    # Generate plots with appropriate file names
    plot_category_accuracy(
        category_stats, overall_stats, round_indices,
        output_dir / f'{output_prefix}category_accuracy.png',
        title_suffix,
        top_n=args.top_n
    )
    
    if not args.no_heatmap:
        plot_category_heatmap(
            category_stats, round_indices,
            output_dir / f'{output_prefix}category_heatmap.png',
            title_suffix
        )
    
    # Save statistics
    save_statistics(
        category_stats, overall_stats, round_indices, all_ops, category_to_ops,
        output_dir / f'{output_prefix}category_statistics.json'
    )
    
    # Print summary
    print_summary(category_stats, overall_stats, round_indices, all_ops, category_to_ops)


if __name__ == '__main__':
    main()
