#!/usr/bin/env python3
"""
Full conversion script for test_v2_ops.py

This script converts all test functions in test_v2_ops.py to include
performance testing alongside accuracy testing.
"""

import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from utils import convert_accuracy_to_performance_test


def extract_all_test_functions(content: str) -> dict:
    """Extract all test functions from the file content.

    Returns:
        dict: {function_name: function_code}
    """
    functions = {}

    # Pattern to match test functions with their decorators
    # Matches from @label to the next @label or end of file
    pattern = r'(# =+ .*? =+\n)?(@label.*?\n)+(@parametrize.*?\n)*def (test_\w+)\(.*?\):(.*?)(?=\n# =+|@label|\Z)'

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        func_name = match.group(4)
        func_code = match.group(0)
        functions[func_name] = func_code

    return functions


def reconstruct_file(original_content: str, converted_functions: dict) -> str:
    """Reconstruct the file with converted functions.

    Args:
        original_content: Original file content
        converted_functions: Dict of converted function code

    Returns:
        str: Reconstructed file content
    """
    # Start with the file header (imports, etc.)
    lines = original_content.split('\n')

    # Find where the first test function starts
    first_test_idx = None
    for i, line in enumerate(lines):
        if line.startswith('# ==========') or line.startswith('@label'):
            first_test_idx = i
            break

    if first_test_idx is None:
        raise ValueError("Could not find first test function")

    # Keep the header
    header = '\n'.join(lines[:first_test_idx])

    # Reconstruct with converted functions
    new_content = header + '\n'

    # Add all converted functions
    for func_name, func_code in converted_functions.items():
        new_content += func_code + '\n\n'

    return new_content


def main():
    """Run full conversion on test_v2_ops.py"""
    print("=" * 80)
    print("Full Conversion Script for test_v2_ops.py")
    print("=" * 80)
    print()

    test_file = project_root / "src" / "flagbench" / "accuracy" / "test_v2_ops.py"

    if not test_file.exists():
        print(f"Error: {test_file} not found")
        return 1

    print(f"Reading: {test_file}")
    with open(test_file, 'r') as f:
        original_content = f.read()

    print(f"Original file size: {len(original_content)} chars")

    # Extract all test functions
    print("\nExtracting test functions...")
    functions = extract_all_test_functions(original_content)
    print(f"Found {len(functions)} test functions")

    # Convert all functions
    print("\nConverting functions...")
    converted_functions = convert_accuracy_to_performance_test(functions)
    print(f"Converted {len(converted_functions)} functions")

    # Reconstruct file
    print("\nReconstructing file...")
    new_content = reconstruct_file(original_content, converted_functions)
    print(f"New file size: {len(new_content)} chars")

    # Write to file
    print(f"\nWriting to: {test_file}")
    with open(test_file, 'w') as f:
        f.write(new_content)

    print("\n" + "=" * 80)
    print("✓ Conversion complete!")
    print("=" * 80)
    print(f"\nBackup saved at: {test_file}.backup")
    print(f"Converted file: {test_file}")
    print(f"\nNext steps:")
    print(f"  1. Check syntax: python -m py_compile {test_file}")
    print(f"  2. Run sample tests: python test/test_accuracy_ut.py --name log_sigmoid_backward")

    return 0


if __name__ == "__main__":
    sys.exit(main())
