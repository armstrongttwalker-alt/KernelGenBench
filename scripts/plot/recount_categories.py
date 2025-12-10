#!/usr/bin/env python3
"""
Script to recount statistics in category JSON files.

This script reads a categories JSON file (like api_categories.json or difficult_categories.json),
recalculates the count statistics, and writes the updated JSON back.

Statistics recalculated:
- total_categories: Number of categories
- total_ops: Total number of ops across all categories (including duplicates)
- total_unique_ops: Total number of unique ops across all categories
- For each category:
  - count: Number of ops in the category (including duplicates)
  - count_unique: Number of unique ops in the category
"""

import json
import argparse
from pathlib import Path


def recount_categories(data: dict) -> dict:
    """
    Recount all statistics in the categories data.
    
    Args:
        data: The original categories dictionary
        
    Returns:
        Updated dictionary with recounted statistics
    """
    categories = data.get('categories', {})
    
    total_ops = 0
    all_ops_list = []
    
    # Process each category
    for category_name, category_info in categories.items():
        ops = category_info.get('ops', [])
        
        # Count ops (including duplicates)
        count = len(ops)
        
        # Count unique ops
        count_unique = len(set(ops))
        
        # Update category stats
        category_info['count'] = count
        category_info['count_unique'] = count_unique
        
        # Accumulate for totals
        total_ops += count
        all_ops_list.extend(ops)
    
    # Calculate totals
    total_categories = len(categories)
    total_unique_ops = len(set(all_ops_list))
    
    # Update top-level stats
    data['total_categories'] = total_categories
    data['total_ops'] = total_ops
    data['total_unique_ops'] = total_unique_ops
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Recount statistics in category JSON files.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input categories JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output file (default: overwrite input file)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the result without writing to file'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file).resolve()
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Read the input file
    print(f"Reading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Store original values for comparison
    original_total_categories = data.get('total_categories', 'N/A')
    original_total_ops = data.get('total_ops', 'N/A')
    original_total_unique_ops = data.get('total_unique_ops', 'N/A')
    
    # Recount
    updated_data = recount_categories(data)
    
    # Print summary
    print("\n" + "=" * 50)
    print("RECOUNT SUMMARY")
    print("=" * 50)
    print(f"\nTotal Categories: {original_total_categories} -> {updated_data['total_categories']}")
    print(f"Total Ops: {original_total_ops} -> {updated_data['total_ops']}")
    print(f"Total Unique Ops: {original_total_unique_ops} -> {updated_data['total_unique_ops']}")
    
    print("\n" + "-" * 50)
    print("Per-Category Counts:")
    print("-" * 50)
    
    for category_name, category_info in updated_data['categories'].items():
        count = category_info['count']
        count_unique = category_info['count_unique']
        print(f"  {category_name:35}: count={count:4}, count_unique={count_unique:4}")
    
    print("=" * 50)
    
    if args.dry_run:
        print("\n[Dry run] No file written.")
        return
    
    # Determine output path
    output_path = Path(args.output).resolve() if args.output else input_path
    
    # Write the updated file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nWritten to: {output_path}")


if __name__ == '__main__':
    main()
