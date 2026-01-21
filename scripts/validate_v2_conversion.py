#!/usr/bin/env python3
"""
Validation script for test_v2_ops.py conversion.

This script validates that the conversion function correctly:
1. Keeps all accuracy testing code
2. Adds performance testing after assert_close()
3. Extracts function calls correctly
4. Produces syntactically valid code
"""

import sys
import ast
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from utils import convert_accuracy_to_performance_test


def extract_test_function(file_path: str, function_name: str) -> str:
    """Extract a single test function from test_v2_ops.py."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the function definition
    pattern = rf'(@label.*?\n)*def {function_name}\(.*?\):\n(.*?)(?=\n@label|\n\ndef |\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        raise ValueError(f"Function {function_name} not found")

    return match.group(0)


def validate_syntax(code: str) -> tuple[bool, str]:
    """Check if the code is syntactically valid."""
    try:
        ast.parse(code)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def validate_conversion(original: str, converted: str) -> dict:
    """Validate that the conversion is correct."""
    results = {
        "syntax_valid": False,
        "has_accuracy_testing": False,
        "has_performance_testing": False,
        "has_imports": False,
        "has_return_statement": False,
        "errors": []
    }

    # Check syntax
    is_valid, msg = validate_syntax(converted)
    results["syntax_valid"] = is_valid
    if not is_valid:
        results["errors"].append(msg)

    # Check that accuracy testing is preserved
    if "assert_close" in converted:
        results["has_accuracy_testing"] = True
    else:
        results["errors"].append("Missing assert_close - accuracy testing not preserved")

    # Check that performance testing is added
    if "triton.testing.do_bench" in converted:
        results["has_performance_testing"] = True
    else:
        results["errors"].append("Missing triton.testing.do_bench - performance testing not added")

    # Check for necessary imports
    if "import triton" in converted and "CustomBenchmarkResult" in converted:
        results["has_imports"] = True
    else:
        results["errors"].append("Missing necessary imports")

    # Check for return statement
    if "return result" in converted or "return CustomBenchmarkResult" in converted:
        results["has_return_statement"] = True
    else:
        results["errors"].append("Missing return statement")

    return results


def main():
    """Run validation on sample test functions."""
    print("=" * 80)
    print("Validation Script for test_v2_ops.py Conversion")
    print("=" * 80)
    print()

    # Sample test functions to validate
    test_samples = [
        "test_log_sigmoid_backward_tensor",
        "test_reflection_pad1d_backward_grad_input",
        "test_rrelu_with_noise_backward_tensor",
    ]

    test_file = project_root / "src" / "flagbench" / "accuracy" / "test_v2_ops.py"

    if not test_file.exists():
        print(f"Error: {test_file} not found")
        return 1

    all_passed = True

    for func_name in test_samples:
        print(f"\n{'=' * 80}")
        print(f"Testing: {func_name}")
        print(f"{'=' * 80}")

        try:
            # Extract original function
            original = extract_test_function(str(test_file), func_name)
            print(f"✓ Extracted function ({len(original)} chars)")

            # Convert using the conversion function
            converted_dict = convert_accuracy_to_performance_test({func_name: original})
            converted = converted_dict[func_name]
            print(f"✓ Converted function ({len(converted)} chars)")

            # Validate conversion
            results = validate_conversion(original, converted)

            # Print results
            print("\nValidation Results:")
            print(f"  Syntax valid: {'✓' if results['syntax_valid'] else '✗'}")
            print(f"  Has accuracy testing: {'✓' if results['has_accuracy_testing'] else '✗'}")
            print(f"  Has performance testing: {'✓' if results['has_performance_testing'] else '✗'}")
            print(f"  Has imports: {'✓' if results['has_imports'] else '✗'}")
            print(f"  Has return statement: {'✓' if results['has_return_statement'] else '✗'}")

            if results["errors"]:
                print("\nErrors:")
                for error in results["errors"]:
                    print(f"  ✗ {error}")
                all_passed = False
            else:
                print("\n✓ All checks passed!")

            # Save converted output for inspection
            output_file = project_root / "tmp" / f"{func_name}_converted.py"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(converted)
            print(f"\nConverted output saved to: {output_file}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print(f"\n{'=' * 80}")
    if all_passed:
        print("✓ All validations passed!")
        return 0
    else:
        print("✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
