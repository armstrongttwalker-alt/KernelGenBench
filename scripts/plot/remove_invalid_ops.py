#!/usr/bin/env python3
"""
Script to remove non-operator entries from category JSON files.

This script reads a categories JSON file and removes entries that match
certain patterns (like __doc__, __name__, etc.) that are clearly not operators.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def find_invalid_ops(categories: dict, patterns: List[str] = None) -> Dict[str, List[str]]:
    """
    Find all operators matching invalid patterns across all categories.
    
    Args:
        categories: The categories dictionary
        patterns: List of regex patterns to match invalid ops
        
    Returns:
        Dictionary mapping category name to list of invalid ops
    """
    if patterns is None:
        patterns = [
            r'^__.*__$',  # Dunder methods like __doc__, __name__
        ]
    
    compiled_patterns = [re.compile(p) for p in patterns]
    
    matches = defaultdict(list)
    
    for category_name, category_info in categories.items():
        ops = category_info.get('ops', [])
        for op in ops:
            for pattern in compiled_patterns:
                if pattern.match(op):
                    matches[category_name].append(op)
                    break
    
    return dict(matches)


def remove_invalid_ops(
    data: dict,
    patterns: List[str] = None,
    exclude_categories: List[str] = None
) -> Tuple[dict, Dict[str, List[str]]]:
    """
    Remove operators matching invalid patterns.
    
    Args:
        data: The original categories dictionary
        patterns: List of regex patterns to match invalid ops
        exclude_categories: Categories to exclude from processing
        
    Returns:
        Tuple of (updated_data, removed_ops_dict)
    """
    if patterns is None:
        patterns = [
            r'^__.*__$',  # Dunder methods like __doc__, __name__
        ]
    
    compiled_patterns = [re.compile(p) for p in patterns]
    
    categories = data.get('categories', {})
    exclude_categories = exclude_categories or []
    
    removed_ops = {}
    
    for category_name, category_info in categories.items():
        if category_name in exclude_categories:
            continue
        
        ops = category_info.get('ops', [])
        ops_to_remove = []
        
        for op in ops:
            for pattern in compiled_patterns:
                if pattern.match(op):
                    ops_to_remove.append(op)
                    break
        
        if ops_to_remove:
            removed_ops[category_name] = ops_to_remove
            # Remove invalid ops
            new_ops = [op for op in ops if op not in ops_to_remove]
            categories[category_name]['ops'] = new_ops
    
    return data, removed_ops


def recount_categories(data: dict) -> dict:
    """Recount all statistics after removal."""
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
        description='Remove non-operator entries from category JSON files.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input categories JSON file'
    )
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        nargs='*',
        default=None,
        help='Regex patterns to match invalid ops (default: ^__.*__$ for dunder methods)'
    )
    parser.add_argument(
        '-e', '--exclude',
        type=str,
        nargs='*',
        default=[],
        help='Categories to exclude from processing'
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
    
    # Default patterns
    patterns = args.pattern if args.pattern else [r'^__.*__$']
    
    # Read the input file
    print(f"Reading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Show current matches before removal
    print(f"\nSearching for ops matching patterns: {patterns}")
    matches = find_invalid_ops(data.get('categories', {}), patterns)
    
    print("\n" + "=" * 60)
    print("INVALID OPERATORS FOUND")
    print("=" * 60)
    
    total_matches = 0
    for category, ops in sorted(matches.items()):
        print(f"\n  [{category}] ({len(ops)} ops):")
        for op in sorted(ops):
            print(f"    - {op}")
        total_matches += len(ops)
    
    if total_matches == 0:
        print("\n  No invalid operators found.")
    else:
        print(f"\nTotal invalid ops: {total_matches}")
    print("=" * 60)
    
    if total_matches == 0:
        print("\nNothing to remove.")
        return
    
    # Remove invalid ops
    print(f"\nRemoving invalid operators...")
    updated_data, removed_ops = remove_invalid_ops(
        data, 
        patterns=patterns,
        exclude_categories=args.exclude
    )
    
    # Recount statistics
    updated_data = recount_categories(updated_data)
    
    # Print what was removed
    print("\n" + "-" * 60)
    print("REMOVAL SUMMARY")
    print("-" * 60)
    
    if removed_ops:
        total_removed = 0
        for category, ops in sorted(removed_ops.items()):
            print(f"\n  From [{category}]:")
            for op in sorted(ops):
                print(f"    - {op}")
            total_removed += len(ops)
        print(f"\nTotal ops removed: {total_removed}")
    else:
        print("\nNo operators were removed.")
    
    print("-" * 60)
    
    # Print updated totals
    print(f"\nUpdated totals:")
    print(f"  Total categories: {updated_data['total_categories']}")
    print(f"  Total ops: {updated_data['total_ops']}")
    print(f"  Total unique ops: {updated_data['total_unique_ops']}")
    
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
