#!/usr/bin/env python3
"""
Script to reclassify operators based on keyword matching.

This script reads a categories JSON file, finds operators matching certain keywords,
and moves them to the appropriate category.

Example: Move all ops containing 'fft' to 'core fft' category.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def find_ops_with_keyword(categories: dict, keyword: str) -> Dict[str, List[str]]:
    """
    Find all operators containing the keyword across all categories.
    
    Args:
        categories: The categories dictionary
        keyword: The keyword to search for (case-insensitive)
        
    Returns:
        Dictionary mapping category name to list of matching ops
    """
    matches = defaultdict(list)
    keyword_lower = keyword.lower()
    
    for category_name, category_info in categories.items():
        ops = category_info.get('ops', [])
        for op in ops:
            if keyword_lower in op.lower():
                matches[category_name].append(op)
    
    return dict(matches)


def reclassify_ops(
    data: dict,
    keyword: str,
    target_category: str,
    exclude_categories: List[str] = None
) -> Tuple[dict, Dict[str, List[str]]]:
    """
    Move operators matching a keyword to a target category.
    
    Args:
        data: The original categories dictionary
        keyword: The keyword to match (case-insensitive)
        target_category: The category to move matching ops to
        exclude_categories: Categories to exclude from moving (e.g., the target itself)
        
    Returns:
        Tuple of (updated_data, moved_ops_dict)
    """
    categories = data.get('categories', {})
    exclude_categories = exclude_categories or []
    
    # Always exclude the target category itself
    if target_category not in exclude_categories:
        exclude_categories.append(target_category)
    
    # Find matching ops
    matches = find_ops_with_keyword(categories, keyword)
    
    # Track what we moved
    moved_ops = {}
    
    keyword_lower = keyword.lower()
    
    for category_name, matching_ops in matches.items():
        if category_name in exclude_categories:
            continue
        
        if not matching_ops:
            continue
        
        # Get the current ops list
        current_ops = categories[category_name].get('ops', [])
        
        # Remove matching ops from this category
        new_ops = [op for op in current_ops if keyword_lower not in op.lower()]
        
        # Get the ops we're moving
        ops_to_move = [op for op in current_ops if keyword_lower in op.lower()]
        
        if ops_to_move:
            moved_ops[category_name] = ops_to_move
            categories[category_name]['ops'] = new_ops
    
    # Add moved ops to target category
    if target_category not in categories:
        categories[target_category] = {'ops': [], 'count': 0, 'count_unique': 0}
    
    target_ops = categories[target_category].get('ops', [])
    for ops_list in moved_ops.values():
        for op in ops_list:
            if op not in target_ops:  # Avoid duplicates
                target_ops.append(op)
    
    categories[target_category]['ops'] = sorted(set(target_ops))  # Sort and dedupe
    
    return data, moved_ops


def recount_categories(data: dict) -> dict:
    """Recount all statistics after reclassification."""
    categories = data.get('categories', {})
    
    total_ops = 0
    all_ops_list = []
    
    # Remove empty categories
    empty_categories = [name for name, info in categories.items() 
                       if not info.get('ops', [])]
    for name in empty_categories:
        del categories[name]
    
    # Process each category
    for category_name, category_info in categories.items():
        ops = category_info.get('ops', [])
        category_info['count'] = len(ops)
        category_info['count_unique'] = len(set(ops))
        total_ops += len(ops)
        all_ops_list.extend(ops)
    
    data['total_categories'] = len(categories)
    data['total_ops'] = total_ops
    data['total_unique_ops'] = len(set(all_ops_list))
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Reclassify operators based on keyword matching.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input categories JSON file'
    )
    parser.add_argument(
        '-k', '--keyword',
        type=str,
        required=True,
        help='Keyword to search for in operator names (case-insensitive)'
    )
    parser.add_argument(
        '-t', '--target',
        type=str,
        required=True,
        help='Target category to move matching operators to'
    )
    parser.add_argument(
        '-e', '--exclude',
        type=str,
        nargs='*',
        default=[],
        help='Categories to exclude from reclassification'
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
        help='Print the changes without writing to file'
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
    
    # Show current matches before reclassification
    print(f"\nSearching for ops containing '{args.keyword}'...")
    matches = find_ops_with_keyword(data.get('categories', {}), args.keyword)
    
    print("\n" + "=" * 60)
    print(f"OPERATORS MATCHING '{args.keyword.upper()}'")
    print("=" * 60)
    
    total_matches = 0
    for category, ops in sorted(matches.items()):
        print(f"\n  [{category}] ({len(ops)} ops):")
        for op in ops:
            print(f"    - {op}")
        total_matches += len(ops)
    
    print(f"\nTotal matches: {total_matches}")
    print("=" * 60)
    
    # Reclassify
    print(f"\nReclassifying to '{args.target}'...")
    updated_data, moved_ops = reclassify_ops(
        data, 
        args.keyword, 
        args.target,
        exclude_categories=args.exclude
    )
    
    # Recount statistics
    updated_data = recount_categories(updated_data)
    
    # Print what was moved
    print("\n" + "-" * 60)
    print("RECLASSIFICATION SUMMARY")
    print("-" * 60)
    
    if moved_ops:
        total_moved = 0
        for from_category, ops in sorted(moved_ops.items()):
            print(f"\n  From [{from_category}] -> [{args.target}]:")
            for op in ops:
                print(f"    - {op}")
            total_moved += len(ops)
        print(f"\nTotal ops moved: {total_moved}")
    else:
        print("\nNo operators were moved.")
    
    # Show target category after reclassification
    target_ops = updated_data['categories'].get(args.target, {}).get('ops', [])
    print(f"\n[{args.target}] now has {len(target_ops)} ops:")
    for op in sorted(target_ops):
        print(f"  - {op}")
    
    print("-" * 60)
    
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
