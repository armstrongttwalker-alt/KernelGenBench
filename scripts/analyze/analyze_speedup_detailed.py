#!/usr/bin/env python3
"""
Analyze speedup data and identify concerning patterns
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

def load_speedup_data(filepath: str) -> Dict:
    """Load speedup summary JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def categorize_operators(data: Dict) -> Tuple[List, List, List]:
    """Categorize operators by speedup ranges"""
    high_speedup = []  # > 1.3
    low_speedup = []   # < 0.1
    normal_speedup = []  # 0.1 <= speedup <= 1.3

    for op_name, stats in data['operators'].items():
        mean_speedup = stats['mean']

        if mean_speedup > 1.3:
            high_speedup.append((op_name, stats))
        elif mean_speedup < 0.1:
            low_speedup.append((op_name, stats))
        else:
            normal_speedup.append((op_name, stats))

    # Sort by speedup value
    high_speedup.sort(key=lambda x: x[1]['mean'], reverse=True)
    low_speedup.sort(key=lambda x: x[1]['mean'])

    return high_speedup, low_speedup, normal_speedup

def analyze_operator_pattern(op_name: str, stats: Dict) -> Dict:
    """Analyze patterns for a single operator"""
    analysis = {
        'operator': op_name,
        'mean_speedup': stats['mean'],
        'min_speedup': stats['min'],
        'max_speedup': stats['max'],
        'count': stats['count'],
        'variance': stats['max'] - stats['min'],
        'rounds': stats['rounds'],
    }

    # Determine operator category
    if '::' in op_name:
        analysis['category'] = op_name.split('::')[1]

    # Check for consistency across rounds
    if stats['count'] > 1:
        analysis['consistent'] = stats['max'] / stats['min'] < 2.0 if stats['min'] > 0 else False
    else:
        analysis['consistent'] = True

    return analysis

def main():
    speedup_file = "/share/project/tj/workspace/flag-bench/output/pass_at_k/pass_at_10_gpt-5_triton_reflection_20251226-184155/verification_rerun/speedup_summary.json"

    print("Loading speedup data...")
    data = load_speedup_data(speedup_file)

    print(f"Total operators: {data['total_operators']}")
    print(f"Total successful rounds: {data['total_successful_rounds']}")
    print()

    high_speedup, low_speedup, normal_speedup = categorize_operators(data)

    print("=" * 80)
    print("HIGH SPEEDUP OPERATORS (> 1.3x)")
    print("=" * 80)
    for op_name, stats in high_speedup:
        analysis = analyze_operator_pattern(op_name, stats)
        print(f"\n{op_name}:")
        print(f"  Mean speedup: {stats['mean']:.4f}x")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Count: {stats['count']} rounds")
        print(f"  Rounds: {stats['rounds']}")

    print("\n" + "=" * 80)
    print("LOW SPEEDUP OPERATORS (< 0.1x - MAJOR SLOWDOWN)")
    print("=" * 80)
    for op_name, stats in low_speedup:
        analysis = analyze_operator_pattern(op_name, stats)
        print(f"\n{op_name}:")
        print(f"  Mean speedup: {stats['mean']:.4f}x ({1/stats['mean']:.1f}x SLOWER)")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Count: {stats['count']} rounds")
        print(f"  Rounds: {stats['rounds']}")

        # Consistency check
        if stats['count'] > 1:
            if analysis['consistent']:
                print(f"  ⚠️  Consistently slow across rounds")
            else:
                print(f"  ⚠️  Variable performance across rounds")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"High speedup (>1.3x):     {len(high_speedup)} operators")
    print(f"Normal (0.1x-1.3x):       {len(normal_speedup)} operators")
    print(f"Low speedup (<0.1x):      {len(low_speedup)} operators")
    print(f"Total:                    {data['total_operators']} operators")

    # Calculate percentage
    if data['total_operators'] > 0:
        print(f"\nPercentage with major slowdown: {len(low_speedup)/data['total_operators']*100:.1f}%")
        print(f"Percentage with speedup:        {len(high_speedup)/data['total_operators']*100:.1f}%")

if __name__ == '__main__':
    main()
