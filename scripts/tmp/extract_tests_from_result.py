#!/usr/bin/env python3
"""
Script to extract test functions for specified ops from result.json
and write them to a new test file.

Usage:
    python extract_tests_from_result.py <sampled_ops_json> <result_json> <output_test_file>

Example:
    python extract_tests_from_result.py \
        cache/runs/.../sampled_from_passed_ops.json \
        cache/runs/.../result.json \
        src/flagbench/accuracy/test_v2_ops.py
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_sampled_ops(sampled_path: Path) -> List[str]:
    """
    Load the list of operator names from sampled_from_passed_ops.json.
    
    Args:
        sampled_path: Path to sampled_from_passed_ops.json
        
    Returns:
        List of operator names
    """
    with open(sampled_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ops = []
    for cat_info in data.get('categories', {}).values():
        ops.extend(cat_info.get('ops', []))
    
    return ops


def load_result_json(result_path: Path) -> Dict[str, dict]:
    """
    Load result.json and create a mapping from op_name to result entry.
    
    Args:
        result_path: Path to result.json
        
    Returns:
        Dictionary mapping op_name (without aten:: prefix) to result entry
    """
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    op_map = {}
    for entry in data:
        op_name = entry.get('op_name', '')
        # Remove aten:: prefix if present
        clean_name = op_name.replace('aten::', '')
        op_map[clean_name] = entry
    
    return op_map


def clean_test_func(test_func: str) -> str:
    """
    Clean up the test function code:
    - Remove duplicate imports
    - Fix any formatting issues
    
    Args:
        test_func: Raw test function code
        
    Returns:
        Cleaned test function code
    """
    if not test_func:
        return ""
    
    lines = test_func.split('\n')
    cleaned_lines = []
    
    # Track which imports we've seen (we'll handle imports separately in the header)
    import_lines = set()
    in_imports = True
    
    for line in lines:
        stripped = line.strip()
        
        # Skip common import lines that will be in the header
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.add(stripped)
            continue
        
        # Skip empty lines at the start
        if not cleaned_lines and not stripped:
            continue
        
        cleaned_lines.append(line)
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def extract_test_functions(
    sampled_ops: List[str],
    result_map: Dict[str, dict]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract test functions for the specified operators.
    
    Args:
        sampled_ops: List of operator names to extract
        result_map: Mapping from op_name to result entry
        
    Returns:
        Tuple of (test_functions, found_ops, missing_ops)
    """
    test_functions = []
    found_ops = []
    missing_ops = []
    
    for op_name in sampled_ops:
        if op_name in result_map:
            entry = result_map[op_name]
            test_func = entry.get('test_func')
            
            if test_func:
                cleaned_func = clean_test_func(test_func)
                if cleaned_func:
                    test_functions.append(f"# ========== {op_name} ==========\n{cleaned_func}")
                    found_ops.append(op_name)
                else:
                    missing_ops.append(f"{op_name} (empty test_func)")
            else:
                missing_ops.append(f"{op_name} (no test_func field)")
        else:
            missing_ops.append(f"{op_name} (not in result.json)")
    
    return test_functions, found_ops, missing_ops


def generate_test_file(test_functions: List[str], output_path: Path):
    """
    Generate the test file with all extracted test functions.
    
    Args:
        test_functions: List of cleaned test function code blocks
        output_path: Path to write the output file
    """
    header = '''#!/usr/bin/env python3
"""
Auto-generated test file for v2 operators.

This file contains test functions extracted from evaluation results.
"""

import torch
import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS


'''

    content = header + '\n\n'.join(test_functions)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description='Extract test functions for specified ops from result.json.'
    )
    parser.add_argument(
        'sampled_ops_json',
        type=str,
        help='Path to sampled_from_passed_ops.json'
    )
    parser.add_argument(
        'result_json',
        type=str,
        help='Path to result.json containing test_func fields'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output test file (e.g., test_v2_ops.py)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print summary without writing file'
    )
    
    args = parser.parse_args()
    
    sampled_path = Path(args.sampled_ops_json).resolve()
    result_path = Path(args.result_json).resolve()
    output_path = Path(args.output_file).resolve()
    
    # Validate inputs
    if not sampled_path.exists():
        print(f"Error: Sampled ops file not found: {sampled_path}")
        return
    
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        return
    
    print(f"Loading sampled ops from: {sampled_path}")
    sampled_ops = load_sampled_ops(sampled_path)
    print(f"Found {len(sampled_ops)} ops to extract")
    
    print(f"\nLoading results from: {result_path}")
    result_map = load_result_json(result_path)
    print(f"Loaded {len(result_map)} entries from result.json")
    
    # Extract test functions
    test_functions, found_ops, missing_ops = extract_test_functions(sampled_ops, result_map)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    
    print(f"\nSuccessfully extracted: {len(found_ops)} ops")
    for op in found_ops:
        print(f"  ✓ {op}")
    
    if missing_ops:
        print(f"\nMissing or empty: {len(missing_ops)} ops")
        for op in missing_ops:
            print(f"  ✗ {op}")
    
    print("=" * 60)
    
    if args.dry_run:
        print("\n[Dry run] No file written.")
        return
    
    # Generate output file
    print(f"\nWriting to: {output_path}")
    generate_test_file(test_functions, output_path)
    print(f"Done! Generated test file with {len(test_functions)} test blocks.")


if __name__ == '__main__':
    main()
