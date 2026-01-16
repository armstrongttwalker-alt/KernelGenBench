#!/usr/bin/env python3
"""
Analyze speedup data from verification results.

This script processes verification results from pass@k experiments,
extracting speedup statistics for successful operators.

Usage:
    python scripts/analyze_speedup.py <verification_dir>

Example:
    python scripts/analyze_speedup.py output/pass_at_k/pass_at_10_gpt-5_triton_reflection_20251226-184155/verification_rerun
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_result_json(log_dir: Path) -> List[Dict]:
    """Load result.json from a log directory."""
    result_file = log_dir / "result.json"
    if not result_file.exists():
        return []

    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {result_file}: {e}", file=sys.stderr)
        return []


def extract_avg_speedup(speedup_list: List[Dict]) -> float:
    """Extract average speedup from speedup list."""
    if not speedup_list:
        return None

    # Find the entry with "params": "avg"
    for entry in speedup_list:
        if entry.get("params") == "avg":
            return entry.get("speedup")

    return None


def analyze_verification_dir(verification_dir: Path) -> Dict[str, Dict]:
    """
    Analyze all log directories in verification_dir.

    Returns:
        Dict mapping operator names to their speedup statistics across rounds.
    """
    # Collect data: op_name -> list of (round_num, avg_speedup)
    operator_data = defaultdict(list)

    # Find all log_* directories
    log_dirs = sorted([d for d in verification_dir.iterdir()
                      if d.is_dir() and d.name.startswith("log_")])

    if not log_dirs:
        print(f"Warning: No log_* directories found in {verification_dir}", file=sys.stderr)
        return {}

    print(f"Found {len(log_dirs)} log directories", file=sys.stderr)

    for log_dir in log_dirs:
        # Extract round number from directory name (e.g., "log_0" -> 0)
        try:
            round_num = int(log_dir.name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse round number from {log_dir.name}", file=sys.stderr)
            continue

        # Load result.json
        results = load_result_json(log_dir)

        # Process each operator result
        for result in results:
            if not result.get("success"):
                continue

            op_name = result.get("op_name")
            if not op_name:
                continue

            speedup_list = result.get("speedup", [])
            avg_speedup = extract_avg_speedup(speedup_list)

            if avg_speedup is not None:
                operator_data[op_name].append((round_num, avg_speedup))

    return operator_data


def compute_statistics(speedups: List[float]) -> Dict:
    """Compute statistics for a list of speedups."""
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


def generate_summary(operator_data: Dict[str, List[Tuple[int, float]]]) -> Dict:
    """
    Generate summary statistics for each operator.

    Returns:
        Dict with operator statistics and summary table.
    """
    summary = {
        "operators": {},
        "total_operators": len(operator_data),
        "total_successful_rounds": sum(len(rounds) for rounds in operator_data.values())
    }

    for op_name, rounds_data in operator_data.items():
        # Extract just the speedup values
        speedups = [speedup for _, speedup in rounds_data]

        # Compute statistics
        stats = compute_statistics(speedups)

        # Add round information
        stats["rounds"] = sorted([round_num for round_num, _ in rounds_data])
        stats["speedups_by_round"] = {
            round_num: speedup for round_num, speedup in rounds_data
        }

        summary["operators"][op_name] = stats

    return summary


def format_markdown_table(summary: Dict) -> str:
    """Format summary as a markdown table."""
    lines = []
    lines.append("# Speedup Analysis Summary")
    lines.append("")
    lines.append(f"**Total Operators**: {summary['total_operators']}")
    lines.append(f"**Total Successful Rounds**: {summary['total_successful_rounds']}")
    lines.append("")
    lines.append("## Speedup Statistics by Operator")
    lines.append("")
    lines.append("| Operator | Mean Speedup | Median Speedup | Min Speedup | Max Speedup | Count | Rounds |")
    lines.append("|----------|--------------|----------------|-------------|-------------|-------|--------|")

    # Sort operators by mean speedup (descending)
    sorted_ops = sorted(
        summary["operators"].items(),
        key=lambda x: x[1].get("mean", 0),
        reverse=True
    )

    for op_name, stats in sorted_ops:
        mean = stats.get("mean", 0)
        median = stats.get("median", 0)
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 0)
        count = stats.get("count", 0)
        rounds = stats.get("rounds", [])
        rounds_str = ",".join(map(str, rounds))

        lines.append(
            f"| {op_name} | {mean:.4f} | {median:.4f} | {min_val:.4f} | "
            f"{max_val:.4f} | {count} | {rounds_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_results(verification_dir: Path, summary: Dict, markdown_table: str):
    """Save analysis results to files in the verification directory."""
    # Save JSON summary
    json_output = verification_dir / "speedup_summary.json"
    with open(json_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to: {json_output}")

    # Save markdown table
    md_output = verification_dir / "speedup_summary.md"
    with open(md_output, 'w') as f:
        f.write(markdown_table)
    print(f"Saved markdown summary to: {md_output}")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_speedup.py <verification_dir>")
        print("\nExample:")
        print("  python scripts/analyze_speedup.py output/pass_at_k/pass_at_10_gpt-5_triton_reflection_20251226-184155/verification_rerun")
        sys.exit(1)

    verification_dir = Path(sys.argv[1])

    if not verification_dir.exists():
        print(f"Error: Directory does not exist: {verification_dir}", file=sys.stderr)
        sys.exit(1)

    if not verification_dir.is_dir():
        print(f"Error: Not a directory: {verification_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing verification results in: {verification_dir}")
    print()

    # Analyze the verification directory
    operator_data = analyze_verification_dir(verification_dir)

    if not operator_data:
        print("No successful operators found with speedup data.", file=sys.stderr)
        sys.exit(1)

    # Generate summary
    summary = generate_summary(operator_data)

    # Format as markdown table
    markdown_table = format_markdown_table(summary)

    # Print to stdout
    print()
    print(markdown_table)

    # Save results
    print()
    save_results(verification_dir, summary, markdown_table)


if __name__ == "__main__":
    main()
