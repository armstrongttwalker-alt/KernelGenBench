#!/usr/bin/env python3
"""
Analyze speedup data from txt files.

This script reads speedup data from txt files (e.g., test_report_aten::sort_speedup.txt)
and calculates average speedup and other statistics.

Usage:
    python scripts/analyze_speedup_txt.py <speedup_txt_file>

Example:
    python scripts/analyze_speedup_txt.py output/fixed_operators/test_results/qwen_next_sort_test/log_0/test_report_aten::sort_speedup.txt
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict


def load_speedup_txt(txt_file: Path) -> List[Dict]:
    """
    Load speedup data from txt file.

    Each line in the file should be a Python dictionary with the format:
    {'ref_time': ..., 'res_time': ..., 'speedup': ..., 'params': {...}}

    Args:
        txt_file: Path to the speedup txt file

    Returns:
        List of dictionaries containing speedup data
    """
    if not txt_file.exists():
        print(f"Error: File does not exist: {txt_file}", file=sys.stderr)
        return []

    speedup_data = []

    try:
        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse the line as a Python dictionary
                    data = ast.literal_eval(line)
                    if isinstance(data, dict) and 'speedup' in data:
                        speedup_data.append(data)
                    else:
                        print(f"Warning: Line {line_num} does not contain 'speedup' field", file=sys.stderr)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    continue

    except Exception as e:
        print(f"Error: Failed to read file {txt_file}: {e}", file=sys.stderr)
        return []

    return speedup_data


def compute_statistics(speedups: List[float]) -> Dict:
    """
    Compute statistics for a list of speedups.

    Args:
        speedups: List of speedup values

    Returns:
        Dictionary containing mean, median, min, max, and count
    """
    if not speedups:
        return {}

    speedups_sorted = sorted(speedups)
    n = len(speedups)

    return {
        "mean": sum(speedups) / n,
        "min": min(speedups),
        "max": max(speedups),
        "median": speedups_sorted[n // 2] if n % 2 == 1 else
                  (speedups_sorted[n // 2 - 1] + speedups_sorted[n // 2]) / 2,
        "count": n
    }


def format_statistics(stats: Dict, filename: str) -> str:
    """
    Format statistics as a readable string.

    Args:
        stats: Statistics dictionary
        filename: Name of the analyzed file

    Returns:
        Formatted string with statistics
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"Speedup Analysis for: {filename}")
    lines.append("=" * 70)
    lines.append("")

    if not stats:
        lines.append("No speedup data found.")
        return "\n".join(lines)

    lines.append(f"Total measurements:  {stats['count']}")
    lines.append("")
    lines.append(f"Average speedup:     {stats['mean']:.6f}")
    lines.append(f"Median speedup:      {stats['median']:.6f}")
    lines.append(f"Min speedup:         {stats['min']:.6f}")
    lines.append(f"Max speedup:         {stats['max']:.6f}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def analyze_by_dtype(speedup_data: List[Dict]) -> Dict[str, Dict]:
    """
    Analyze speedup data grouped by dtype.

    Args:
        speedup_data: List of speedup data dictionaries

    Returns:
        Dictionary mapping dtype to statistics
    """
    dtype_groups = {}

    for data in speedup_data:
        params = data.get('params', {})
        dtype = params.get('dtype', 'unknown')

        if dtype not in dtype_groups:
            dtype_groups[dtype] = []

        dtype_groups[dtype].append(data['speedup'])

    # Compute statistics for each dtype
    dtype_stats = {}
    for dtype, speedups in dtype_groups.items():
        dtype_stats[dtype] = compute_statistics(speedups)

    return dtype_stats


def format_dtype_statistics(dtype_stats: Dict[str, Dict]) -> str:
    """
    Format dtype statistics as a readable string.

    Args:
        dtype_stats: Dictionary mapping dtype to statistics

    Returns:
        Formatted string with dtype statistics
    """
    if not dtype_stats:
        return ""

    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("Speedup Analysis by Data Type")
    lines.append("=" * 70)
    lines.append("")

    # Sort by dtype name
    for dtype in sorted(dtype_stats.keys()):
        stats = dtype_stats[dtype]
        lines.append(f"Data Type: {dtype}")
        lines.append("-" * 70)
        lines.append(f"  Total measurements:  {stats['count']}")
        lines.append(f"  Average speedup:     {stats['mean']:.6f}")
        lines.append(f"  Median speedup:      {stats['median']:.6f}")
        lines.append(f"  Min speedup:         {stats['min']:.6f}")
        lines.append(f"  Max speedup:         {stats['max']:.6f}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_speedup_txt.py <speedup_txt_file>")
        print("\nExample:")
        print("  python scripts/analyze_speedup_txt.py output/fixed_operators/test_results/qwen_next_sort_test/log_0/test_report_aten::sort_speedup.txt")
        sys.exit(1)

    txt_file = Path(sys.argv[1])

    if not txt_file.exists():
        print(f"Error: File does not exist: {txt_file}", file=sys.stderr)
        sys.exit(1)

    if not txt_file.is_file():
        print(f"Error: Not a file: {txt_file}", file=sys.stderr)
        sys.exit(1)

    # Load speedup data
    print(f"Loading speedup data from: {txt_file}")
    speedup_data = load_speedup_txt(txt_file)

    if not speedup_data:
        print("No valid speedup data found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(speedup_data)} speedup measurements")
    print()

    # Extract speedup values
    speedups = [data['speedup'] for data in speedup_data]

    # Compute overall statistics
    stats = compute_statistics(speedups)

    # Print overall statistics
    result = format_statistics(stats, txt_file.name)
    print(result)

    # Analyze by dtype if possible
    dtype_stats = analyze_by_dtype(speedup_data)
    if dtype_stats:
        dtype_result = format_dtype_statistics(dtype_stats)
        print(dtype_result)


if __name__ == "__main__":
    main()
