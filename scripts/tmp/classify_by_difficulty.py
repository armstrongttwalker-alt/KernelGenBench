#!/usr/bin/env python3
# filepath: /share/project/tj/workspace/flag-bench/scripts/plot/classify_by_difficulty.py
"""
Script to classify operators by difficulty based on pass@k evaluation results.

This script:
1. Reads evaluation results from multiple rounds (log_X directories)
2. Reads the target operators from sampled_from_passed_ops.json
3. Classifies operators into difficulty levels:
   - level1: Passed in at least one round (pass@k success)
   - level2: Failed in all rounds (pass@k failure)
4. Outputs results in the same format as difficult_categories.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Set, List, Any


def load_sampled_ops(sampled_ops_path: Path) -> Set[str]:
    """
    Load the target operator names from sampled_from_passed_ops.json.
    
    Args:
        sampled_ops_path: Path to sampled_from_passed_ops.json
        
    Returns:
        Set of operator names
    """
    with open(sampled_ops_path, 'r') as f:
        data = json.load(f)
    
    all_ops = set()
    for category_info in data.get('categories', {}).values():
        ops = category_info.get('ops', [])
        all_ops.update(ops)
    
    return all_ops


def load_evaluation_results(eval_dir: Path) -> Dict[str, bool]:
    """
    Load evaluation results from all rounds and compute pass@k status for each operator.
    
    An operator passes if it succeeded in at least one round.
    
    Args:
        eval_dir: Path to the evaluation directory
        
    Returns:
        Dictionary mapping operator name to pass status (True if passed in any round)
    """
    verification_dir = eval_dir / 'verification'
    if not verification_dir.exists():
        verification_dir = eval_dir
    
    # Find all log directories
    log_dirs = sorted(
        [d for d in verification_dir.iterdir() if d.is_dir() and d.name.startswith('log_')],
        key=lambda x: int(x.name.split('_')[1])
    )
    
    if not log_dirs:
        raise ValueError(f"No log directories found in {verification_dir}")
    
    print(f"Found {len(log_dirs)} evaluation rounds")
    
    # Track success status for each operator across all rounds
    op_success: Dict[str, bool] = {}
    
    for log_dir in log_dirs:
        result_file = log_dir / 'result.json'
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping")
            continue
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        for result in results:
            op_name = result.get('op_name', '')
            success = result.get('success', False)
            
            # If operator succeeded in any round, mark as passed
            if op_name not in op_success:
                op_success[op_name] = False
            
            if success:
                op_success[op_name] = True
    
    return op_success


def normalize_op_name(op_name: str) -> str:
    """
    Normalize operator name by removing 'aten::' prefix if present.
    
    Args:
        op_name: Operator name (may have 'aten::' prefix)
        
    Returns:
        Normalized operator name without prefix
    """
    if op_name.startswith('aten::'):
        return op_name[6:]  # Remove 'aten::' prefix
    return op_name


def classify_operators(
    target_ops: Set[str],
    eval_results: Dict[str, bool]
) -> Dict[str, List[str]]:
    """
    Classify operators into difficulty levels.
    
    Args:
        target_ops: Set of target operator names (without 'aten::' prefix)
        eval_results: Dictionary mapping operator name to pass status (may have 'aten::' prefix)
        
    Returns:
        Dictionary with 'level1' and 'level2' lists
    """
    level1 = []  # Passed in at least one round
    level2 = []  # Failed in all rounds
    not_found = []  # Not found in evaluation results
    
    # Build a normalized lookup dict: normalized_name -> pass_status
    # This handles both 'aten::xxx' and 'xxx' formats in eval_results
    normalized_eval_results: Dict[str, bool] = {}
    for op_name, passed in eval_results.items():
        normalized_name = normalize_op_name(op_name)
        # If same op appears with different formats, use OR logic (passed in any)
        if normalized_name in normalized_eval_results:
            normalized_eval_results[normalized_name] = normalized_eval_results[normalized_name] or passed
        else:
            normalized_eval_results[normalized_name] = passed
    
    for op in sorted(target_ops):
        normalized_op = normalize_op_name(op)
        if normalized_op in normalized_eval_results:
            if normalized_eval_results[normalized_op]:
                level1.append(op)
            else:
                level2.append(op)
        else:
            print(f"Warning: Operator '{op}' not found in evaluation results")
            not_found.append(op)
    
    # Add not_found operators to level2 (conservative approach)
    level2.extend(not_found)
    
    return {
        'level1': sorted(level1),
        'level2': sorted(level2)
    }


def create_output_json(classification: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create output JSON in the same format as difficult_categories.json.
    
    Args:
        classification: Dictionary with 'level1' and 'level2' lists
        
    Returns:
        Formatted output dictionary
    """
    all_ops = classification['level1'] + classification['level2']
    
    output = {
        "total_categories": 2,
        "total_ops": len(all_ops),
        "total_unique_ops": len(set(all_ops)),
        "description": {
            "level1": "Operators that passed in at least one evaluation round (pass@k success)",
            "level2": "Operators that failed in all evaluation rounds (pass@k failure)"
        },
        "categories": {
            "level1": {
                "count": len(classification['level1']),
                "count_unique": len(classification['level1']),
                "ops": classification['level1']
            },
            "level2": {
                "count": len(classification['level2']),
                "count_unique": len(classification['level2']),
                "ops": classification['level2']
            }
        }
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Classify operators by difficulty based on pass@k evaluation results.'
    )
    parser.add_argument(
        'eval_dir',
        type=str,
        help='Path to the evaluation results directory'
    )
    parser.add_argument(
        'sampled_ops',
        type=str,
        help='Path to sampled_from_passed_ops.json'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: scripts/plot/difficult_categories_v2.json)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print results without writing to file'
    )
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir).resolve()
    sampled_ops_path = Path(args.sampled_ops).resolve()
    
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        return 1
    
    if not sampled_ops_path.exists():
        print(f"Error: Sampled ops file not found: {sampled_ops_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / 'difficult_categories_v2.json'
    
    print(f"Evaluation directory: {eval_dir}")
    print(f"Sampled ops file: {sampled_ops_path}")
    print(f"Output file: {output_path}")
    print()
    
    # Load data
    print("Loading target operators...")
    target_ops = load_sampled_ops(sampled_ops_path)
    print(f"Found {len(target_ops)} target operators")
    
    print("\nLoading evaluation results...")
    eval_results = load_evaluation_results(eval_dir)
    print(f"Found results for {len(eval_results)} operators")
    
    # Classify operators
    print("\nClassifying operators by difficulty...")
    classification = classify_operators(target_ops, eval_results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Level 1 (passed pass@k): {len(classification['level1'])} operators")
    print(f"Level 2 (failed pass@k): {len(classification['level2'])} operators")
    print()
    
    print("Level 1 operators:")
    for op in classification['level1']:
        print(f"  - {op}")
    
    print("\nLevel 2 operators:")
    for op in classification['level2']:
        print(f"  - {op}")
    
    # Create output
    output_data = create_output_json(classification)
    
    if args.dry_run:
        print("\n[Dry-run mode] Would write the following JSON:")
        print(json.dumps(output_data, indent=2))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())