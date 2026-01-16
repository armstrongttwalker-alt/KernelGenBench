#!/usr/bin/env python3
"""
Convert FlagGems test functions to flagbench format.

This tool parses FlagGems test files and converts them to flagbench format,
handling decorator changes, import statements, and context manager replacements.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import logging

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FLAGGEMS_DIR = PROJECT_ROOT / "FlagGems"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestFunction:
    """Represents a parsed test function."""

    def __init__(
        self,
        name: str,
        decorators: List[ast.expr],
        args: ast.arguments,
        body: List[ast.stmt],
        op_marks: List[str],
        source_file: Path
    ):
        self.name = name
        self.decorators = decorators
        self.args = args
        self.body = body
        self.op_marks = op_marks  # List of operator marks like ['add', 'inplace']
        self.source_file = source_file


class TestParser:
    """Parse FlagGems test files and extract test functions."""

    def __init__(self, flaggems_tests_dir: Path):
        self.tests_dir = flaggems_tests_dir

    def find_test_files(self) -> List[Path]:
        """Find all test files in FlagGems tests directory."""
        return list(self.tests_dir.glob("test_*.py"))

    def parse_test_file(self, file_path: Path) -> Tuple[ast.Module, str]:
        """Parse a test file and return AST and source code."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
        return tree, source

    def extract_test_functions(self, tree: ast.Module, source_file: Path) -> List[TestFunction]:
        """Extract all test functions from AST."""
        test_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Extract operator marks from decorators
                op_marks = self._extract_op_marks(node.decorator_list)

                test_func = TestFunction(
                    name=node.name,
                    decorators=node.decorator_list,
                    args=node.args,
                    body=node.body,
                    op_marks=op_marks,
                    source_file=source_file
                )
                test_functions.append(test_func)

        return test_functions

    def _extract_op_marks(self, decorators: List[ast.expr]) -> List[str]:
        """Extract operator marks from pytest.mark decorators."""
        op_marks = []

        for decorator in decorators:
            # Handle @pytest.mark.{op_name}
            if isinstance(decorator, ast.Attribute):
                if (isinstance(decorator.value, ast.Attribute) and
                    isinstance(decorator.value.value, ast.Name) and
                    decorator.value.value.id == 'pytest' and
                    decorator.value.attr == 'mark'):
                    op_marks.append(decorator.attr)

        return op_marks

    def extract_module_constants(self, tree: ast.Module, source: str) -> Dict[str, str]:
        """Extract module-level constant definitions from AST.

        Returns a dictionary mapping constant names to their source code.
        """
        constants = {}

        for node in tree.body:
            # Extract simple assignments (e.g., CONST = value)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        const_name = target.id
                        # Only extract UPPERCASE constants (convention)
                        if const_name.isupper() or const_name.startswith('_'):
                            try:
                                # Get the source code for this assignment
                                const_code = ast.unparse(node)
                                constants[const_name] = const_code
                            except Exception as e:
                                logger.warning(f"Failed to extract constant {const_name}: {e}")

        return constants

    def extract_helper_functions(self, tree: ast.Module, source: str) -> Dict[str, str]:
        """Extract helper functions (non-test functions) from AST.

        Returns a dictionary mapping function names to their source code.
        """
        helper_functions = {}

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Extract non-test functions (not starting with 'test_' or '_')
                if not node.name.startswith('test_') and not node.name.startswith('_'):
                    try:
                        # Get the source code for this function
                        func_code = ast.unparse(node)
                        helper_functions[node.name] = func_code
                    except Exception as e:
                        logger.warning(f"Failed to extract helper function {node.name}: {e}")

        return helper_functions

    def find_operator_tests(self, operator_name: str) -> List[TestFunction]:
        """Find all test functions for a specific operator.

        Uses exact matching - 'add' will only match 'add', not 'add_'.
        To convert inplace variants, specify them explicitly (e.g., 'add_').
        """
        all_tests = []

        for test_file in self.find_test_files():
            try:
                tree, source = self.parse_test_file(test_file)
                functions = self.extract_test_functions(tree, test_file)

                # Filter functions that have the exact operator mark
                operator_tests = [
                    func for func in functions
                    if operator_name in func.op_marks
                ]
                all_tests.extend(operator_tests)

            except Exception as e:
                logger.warning(f"Failed to parse {test_file}: {e}")

        return all_tests


class TestConverter:
    """Convert FlagGems test functions to flagbench format."""

    def __init__(self):
        self.import_replacements = {
            'flag_gems': 'flagbench',
        }

    def extract_names_from_constant(self, const_code: str) -> Set[str]:
        """Extract all names referenced in a constant definition.

        Args:
            const_code: The source code of the constant definition

        Returns:
            Set of names referenced in the constant
        """
        try:
            tree = ast.parse(const_code)
            names = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)

            return names
        except Exception as e:
            logger.warning(f"Failed to extract names from constant: {e}")
            return set()

    def resolve_constant_dependencies(self, const_name: str, all_constants: Dict[str, str],
                                     resolved: Optional[Set[str]] = None) -> Set[str]:
        """Recursively resolve all dependencies of a constant.

        Args:
            const_name: Name of the constant to resolve
            all_constants: Dictionary of all available constants
            resolved: Set of already resolved constants (for cycle detection)

        Returns:
            Set of all constant names that this constant depends on
        """
        if resolved is None:
            resolved = set()

        if const_name in resolved:
            return set()

        if const_name not in all_constants:
            return set()

        resolved.add(const_name)
        dependencies = {const_name}

        # Extract names from this constant's definition
        const_code = all_constants[const_name]
        referenced_names = self.extract_names_from_constant(const_code)

        # Recursively resolve dependencies
        for name in referenced_names:
            if name in all_constants and name not in resolved:
                deps = self.resolve_constant_dependencies(name, all_constants, resolved)
                dependencies.update(deps)

        return dependencies

    def convert_test_function(self, test_func: TestFunction, source: str) -> str:
        """Convert a single test function to flagbench format."""
        # Extract the function source code
        func_source = self._extract_function_source(test_func, source)

        # Apply conversions
        converted = func_source
        converted = self._convert_decorators(converted)
        converted = self._convert_code_body(converted)

        return converted

    def _extract_function_source(self, test_func: TestFunction, full_source: str) -> str:
        """Extract the source code of a specific function."""
        lines = full_source.split('\n')

        # Find the function definition line
        func_pattern = rf'^\s*def\s+{re.escape(test_func.name)}\s*\('
        start_line = None

        for i, line in enumerate(lines):
            if re.match(func_pattern, line):
                start_line = i
                break

        if start_line is None:
            raise ValueError(f"Could not find function {test_func.name}")

        # Find decorators before the function
        # Search upward until we hit an empty line (decorators are usually continuous)
        decorator_start = start_line
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()

            # Stop at empty line (decorators are continuous without empty lines)
            if not line:
                decorator_start = i + 1
                break

            # Include this line (decorator, comment, or part of multi-line decorator)
            decorator_start = i
        else:
            # Reached start of file without hitting empty line
            decorator_start = 0

        # Find the end of the function signature (the line with ':')
        signature_end = start_line
        for i in range(start_line, len(lines)):
            if ':' in lines[i]:
                signature_end = i
                break

        # Find the end of the function body (next function or end of file)
        end_line = len(lines)
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())

        # Start searching from after the signature ends
        for i in range(signature_end + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                # If we find a line with same or less indentation, function ends
                if current_indent <= indent_level and not line.strip().startswith('#'):
                    end_line = i
                    break

        func_lines = lines[decorator_start:end_line]
        return '\n'.join(func_lines)

    def _convert_decorators(self, code: str) -> str:
        """Convert pytest decorators to flagbench format."""
        lines = code.split('\n')
        converted_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Convert @pytest.mark.{op_name} to @label("{op_name}")
            if '@pytest.mark.' in line:
                if '@pytest.mark.parametrize' in line:
                    # Convert @pytest.mark.parametrize to @parametrize
                    line = line.replace('@pytest.mark.parametrize', '@parametrize')
                    converted_lines.append(line)
                    i += 1
                elif '@pytest.mark.skipif' in line:
                    # Skip the entire skipif decorator (including multi-line arguments)
                    # Find the matching closing parenthesis
                    paren_count = line.count('(') - line.count(')')
                    i += 1
                    while i < len(lines) and paren_count > 0:
                        paren_count += lines[i].count('(') - lines[i].count(')')
                        i += 1
                    # Don't append anything - we're removing the skipif decorator
                elif '@pytest.mark.inplace' in line:
                    # Convert @pytest.mark.inplace to @label("inplace")
                    indent = line[:len(line) - len(line.lstrip())]
                    line = f'{indent}@label("inplace")'
                    converted_lines.append(line)
                    i += 1
                else:
                    # Extract the mark name and convert to @label
                    match = re.search(r'@pytest\.mark\.(\w+)', line)
                    if match:
                        mark_name = match.group(1)
                        indent = line[:len(line) - len(line.lstrip())]
                        line = f'{indent}@label("{mark_name}")'
                    converted_lines.append(line)
                    i += 1
            else:
                converted_lines.append(line)
                i += 1

        return '\n'.join(converted_lines)

    def _replace_pytest_skip(self, code: str) -> str:
        """Replace pytest.skip() calls with return statements.

        Uses bracket matching to handle multi-line calls and strings with parentheses.
        """
        result = []
        i = 0
        while i < len(code):
            # Look for pytest.skip(
            if code[i:i+12] == 'pytest.skip(':
                # Found pytest.skip(, now find the matching closing parenthesis
                paren_count = 1
                j = i + 12
                in_string = False
                string_char = None

                while j < len(code) and paren_count > 0:
                    char = code[j]

                    # Handle string literals
                    if char in ('"', "'") and (j == 0 or code[j-1] != '\\'):
                        if not in_string:
                            in_string = True
                            string_char = char
                        elif char == string_char:
                            in_string = False
                            string_char = None

                    # Count parentheses only outside strings
                    if not in_string:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1

                    j += 1

                # Replace the entire pytest.skip(...) with return
                result.append('return')
                i = j
            else:
                result.append(code[i])
                i += 1

        return ''.join(result)

    def _convert_code_body(self, code: str) -> str:
        """Convert code body to flagbench format."""
        # Replace flag_gems.device with device
        code = re.sub(r'\bflag_gems\.device\b', 'device', code)

        # Replace flag_gems.use_gems() with flagbench.use_gems(REGISTERED_OPS)
        code = re.sub(
            r'with\s+flag_gems\.use_gems\(\)\s*:',
            'with flagbench.use_gems(REGISTERED_OPS):',
            code
        )

        # Replace import flag_gems with import flagbench
        code = re.sub(r'\bimport\s+flag_gems\b', 'import flagbench', code)

        # Convert @parametrize list-style param names to comma-separated string
        # Match: @parametrize(\n    ["param1", "param2", ...],
        # Replace with: @parametrize(\n    "param1,param2,...",
        def convert_param_list(match):
            # Extract the list content
            list_content = match.group(1)
            # Find all quoted strings in the list
            params = re.findall(r'"([^"]+)"', list_content)
            # Join with commas
            return f'@parametrize(\n    "{",".join(params)}",'

        code = re.sub(
            r'@parametrize\(\s*\n\s*\[([^\]]+)\],',
            convert_param_list,
            code
        )

        # Replace pytest.skip() with return (test framework doesn't support pytest skip mechanism)
        code = self._replace_pytest_skip(code)

        return code

    def generate_imports(self, test_functions: List[TestFunction], original_source: str,
                        extracted_constants: Optional[Dict[str, str]] = None,
                        extracted_helpers: Optional[Dict[str, str]] = None) -> str:
        """Generate import statements for the converted file.

        Args:
            test_functions: List of test functions to convert
            original_source: Original source code
            extracted_constants: Dictionary of extracted constant definitions
            extracted_helpers: Dictionary of extracted helper functions
        """
        imports = []

        if extracted_constants is None:
            extracted_constants = {}
        if extracted_helpers is None:
            extracted_helpers = {}

        # Check if numpy or random are needed
        needs_numpy = 'np.' in original_source or 'numpy.' in original_source
        needs_random = 'random.' in original_source or 'import random' in original_source
        needs_os = 'os.' in original_source or 'os.environ' in original_source
        needs_itertools = 'itertools.' in original_source
        needs_logging = 'logging.' in original_source or 'import logging' in original_source
        needs_pytest = 'pytest.' in original_source or 'import pytest' in original_source

        # Check if flag_gems is needed (for vendor-specific constants)
        needs_flag_gems = False
        if extracted_constants:
            for const_code in extracted_constants.values():
                if 'flag_gems' in const_code:
                    needs_flag_gems = True
                    break

        # Check if typing imports are needed (for helper functions with type hints)
        typing_imports = set()
        common_typing_names = ['List', 'Dict', 'Optional', 'Union', 'Tuple', 'Set', 'Any', 'Callable']

        if extracted_helpers:
            for helper_code in extracted_helpers.values():
                for typing_name in common_typing_names:
                    # Check if the typing name is used as a type hint (followed by [ or ,)
                    if f'{typing_name}[' in helper_code or f', {typing_name}' in helper_code or f': {typing_name}' in helper_code:
                        typing_imports.add(typing_name)

        # Standard imports
        if typing_imports:
            imports.append(f"from typing import {', '.join(sorted(typing_imports))}")
        if needs_numpy:
            imports.append("import numpy as np")
        if needs_random:
            imports.append("import random")
        if needs_os:
            imports.append("import os")
        if needs_itertools:
            imports.append("import itertools")
        if needs_logging:
            imports.append("import logging")
        if needs_pytest:
            imports.append("import pytest")
        imports.append("import torch")
        if needs_flag_gems:
            imports.append("import flag_gems  # Needed for vendor-specific constants")
        imports.append("import flagbench")

        # Check if we need specific imports based on the test functions
        needs_parametrize = any('@parametrize' in self.convert_test_function(tf, original_source)
                                 for tf in test_functions)
        needs_label = any('@label' in self.convert_test_function(tf, original_source)
                         for tf in test_functions)

        if needs_parametrize or needs_label:
            imports.append("from sandbox.verifier.test_parametrize import parametrize, label")

        # Import from sandbox
        imports.append("from sandbox.config import DEVICE as device, QUICK_MODE, TO_CPU")
        imports.append("from sandbox.register import REGISTERED_OPS")

        # Check what accuracy utils are needed
        if 'gems_assert_close' in original_source or 'gems_assert_equal' in original_source:
            utils_imports = []

            # Check for commonly used functions from accuracy_utils
            common_utils_functions = [
                'gems_assert_close', 'gems_assert_equal', 'to_reference', 'to_cpu',
                'init_seed', 'SkipVersion',
                'unsqueeze_tuple', 'unsqueeze_tensor'
            ]

            for func_name in common_utils_functions:
                if func_name in original_source:
                    utils_imports.append(func_name)

            # Check for dtype/shape constants from accuracy_utils
            dtype_constants = ['FLOAT_DTYPES', 'INT_DTYPES', 'ALL_FLOAT_DTYPES', 'ALL_INT_DTYPES',
                               'BOOL_TYPES', 'PRIMARY_FLOAT_DTYPES', 'COMPLEX_DTYPES']
            shape_constants = ['POINTWISE_SHAPES', 'SPECIAL_SHAPES', 'REDUCTION_SHAPES', 'REDUCTION_SMALL_SHAPES',
                               'STACK_SHAPES', 'SCALARS', 'UT_SHAPES_1D', 'UT_SHAPES_2D', 'KRON_SHAPES',
                               'DISTRIBUTION_SHAPES', 'SHAPE_STRIDES', 'CONTIGUOUS_SHAPE_STRIDES_2D',
                               'CONTIGUOUS_SHAPE_STRIDES_3D', 'CONTIGUOUS_SHAPE_STRIDES_4D',
                               'IRREGULAR_SHAPE_STRIDES', 'UPSAMPLE_SHAPES', 'STACK_DIM_LIST', 'ARANGE_START']

            for const in dtype_constants + shape_constants:
                if const in original_source:
                    utils_imports.append(const)

            if utils_imports:
                imports.append(f"from sandbox.utils.accuracy_utils import {', '.join(sorted(set(utils_imports)))}")

        # Merge hardcoded fallback constants with extracted constants
        fallback_constants = {
            'DIMS_LIST': 'DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]',
            'DIM_LIST': 'DIM_LIST = [1] if QUICK_MODE else [0, 1]',
            'KEEPDIM_DIMS_SHAPE': (
                'KEEPDIM_DIMS_SHAPE = (\n'
                '    [(True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if QUICK_MODE\n'
                '    else list(zip([True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))\n'
                ')'
            ),
            'KIND_KEEPDIM_DIMS_SHAPE': (
                'KIND_KEEPDIM_DIMS_SHAPE = (\n'
                '    [("normal", True, DIMS_LIST[0], REDUCTION_SHAPES[0])] if QUICK_MODE\n'
                '    else list(zip(["normal", "allTrue"] * 2, [True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))\n'
                ')'
            ),
            'KEEPDIM_DIMS': (
                'KEEPDIM_DIMS = (\n'
                '    [(True, DIMS_LIST[0])] if QUICK_MODE\n'
                '    else list(zip([True, False] * 2, DIMS_LIST))\n'
                ')'
            ),
            'KEEPDIM_DIM': (
                'KEEPDIM_DIM = (\n'
                '    [(True, DIM_LIST[0])] if QUICK_MODE\n'
                '    else list(zip([True, False], DIM_LIST))\n'
                ')'
            ),
            'EMPTY_SHAPES': 'EMPTY_SHAPES = [(0, 5), (3, 0, 4), (2, 5, 0), (0,)]',
        }

        # Use extracted constants, fallback to hardcoded if not found
        # Note: We don't apply conversion to extracted constants because they may
        # contain vendor-specific code (like flag_gems.vendor_name) that should be preserved
        all_constants = {**fallback_constants, **extracted_constants}

        # Use dependency analysis to find which constants are needed
        # Step 1: Find constants that are directly used in the source
        used_constants = set()
        for const_name in all_constants.keys():
            if const_name in original_source and const_name not in utils_imports:
                used_constants.add(const_name)

        # Step 2: Resolve dependencies for each used constant
        all_needed_constants = set()
        for const_name in used_constants:
            dependencies = self.resolve_constant_dependencies(const_name, all_constants)
            all_needed_constants.update(dependencies)

        # Step 3: Build the list of constant definitions in dependency order
        # We need to ensure constants are defined before they're used
        needed_constants = []
        added_constants = set()

        def add_constant_with_deps(const_name):
            """Add a constant and its dependencies in the correct order."""
            if const_name in added_constants or const_name not in all_constants:
                return

            # Mark as added BEFORE processing dependencies to prevent infinite recursion
            added_constants.add(const_name)

            # First, add dependencies
            const_code = all_constants[const_name]
            referenced_names = self.extract_names_from_constant(const_code)
            for ref_name in referenced_names:
                if ref_name in all_needed_constants and ref_name not in added_constants:
                    add_constant_with_deps(ref_name)

            # Then add this constant's definition
            needed_constants.append(all_constants[const_name])

        # Add all needed constants in dependency order
        for const_name in sorted(all_needed_constants):
            add_constant_with_deps(const_name)

        if needed_constants:
            imports.append('\n# Additional constant definitions from FlagGems')
            imports.extend(needed_constants)

        return '\n'.join(imports)


class TestWriter:
    """Write converted test functions to flagbench format."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_test_file(
        self,
        operators: List[str],
        test_functions: List[TestFunction],
        converter: TestConverter,
        original_sources: Dict[Path, str],
        output_file: Path,
        dry_run: bool = False,
        extracted_constants: Optional[Dict[str, str]] = None,
        extracted_helpers: Optional[Dict[str, str]] = None
    ):
        """Write converted test functions to a file.

        Args:
            operators: List of operator names
            test_functions: List of test functions to convert
            converter: TestConverter instance
            original_sources: Dictionary of original source files
            output_file: Output file path
            dry_run: If True, only preview without writing
            extracted_constants: Dictionary of extracted constant definitions
            extracted_helpers: Dictionary of extracted helper functions
        """
        if extracted_constants is None:
            extracted_constants = {}
        if extracted_helpers is None:
            extracted_helpers = {}
        if not test_functions:
            logger.warning(f"No test functions found for operators: {operators}")
            return

        # Group functions by source file to get original source
        functions_by_source = {}
        for func in test_functions:
            if func.source_file not in functions_by_source:
                functions_by_source[func.source_file] = []
            functions_by_source[func.source_file].append(func)

        # Generate file header
        header = self._generate_header(operators)

        # Generate imports (check all source files)
        all_sources = '\n'.join(original_sources.values())
        imports = converter.generate_imports(test_functions, all_sources,
                                             extracted_constants, extracted_helpers)

        # Prepare helper functions section
        helper_functions_code = []
        if extracted_helpers:
            helper_functions_code.append('\n# Helper functions from FlagGems')
            for helper_name, helper_code in sorted(extracted_helpers.items()):
                helper_functions_code.append(helper_code)

        # Convert all test functions
        converted_functions = []
        conversion_errors = []

        for func in test_functions:
            try:
                source = original_sources[func.source_file]
                converted = converter.convert_test_function(func, source)
                converted_functions.append((func.name, converted))
            except Exception as e:
                logger.error(f"Failed to convert {func.name}: {e}")
                conversion_errors.append((func.name, str(e)))

        if dry_run:
            # Dry run mode: display information without writing
            logger.info("\n" + "="*60)
            logger.info("DRY RUN MODE - Preview of conversion")
            logger.info("="*60)
            logger.info(f"Output file: {output_file}")
            logger.info(f"Operators: {', '.join(operators)}")
            logger.info(f"Total test functions: {len(test_functions)}")
            logger.info(f"Successfully converted: {len(converted_functions)}")
            if conversion_errors:
                logger.info(f"Conversion errors: {len(conversion_errors)}")

            logger.info("\nTest functions to be converted:")
            for i, (func_name, _) in enumerate(converted_functions, 1):
                logger.info(f"  {i}. {func_name}")

            if conversion_errors:
                logger.info("\nConversion errors:")
                for func_name, error in conversion_errors:
                    logger.info(f"  ✗ {func_name}: {error}")

            logger.info("\nFile structure preview:")
            logger.info(f"  - Header: {len(header.split(chr(10)))} lines")
            logger.info(f"  - Imports: {len(imports.split(chr(10)))} lines")
            logger.info(f"  - Helper functions: {len(extracted_helpers)}")
            logger.info(f"  - Test functions: {len(converted_functions)}")

            total_lines = (len(header.split('\n')) +
                          len(imports.split('\n')) +
                          sum(len(code.split('\n')) for code in helper_functions_code) +
                          sum(len(code.split('\n')) for _, code in converted_functions))
            logger.info(f"  - Total lines: ~{total_lines}")

            logger.info("\n" + "="*60)
            logger.info("No files were written (dry run mode)")
            logger.info("="*60)
        else:
            # Normal mode: write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write('\n\n')
                f.write(imports)
                f.write('\n\n')
                # Write helper functions
                if helper_functions_code:
                    f.write('\n\n'.join(helper_functions_code))
                    f.write('\n\n')
                f.write('\n\n'.join(code for _, code in converted_functions))
                f.write('\n')

            logger.info(f"Wrote {len(converted_functions)} test functions to {output_file}")
            if conversion_errors:
                logger.warning(f"Failed to convert {len(conversion_errors)} functions")

    def _generate_header(self, operators: List[str]) -> str:
        """Generate file header."""
        ops_str = ', '.join(operators)
        return f'''#!/usr/bin/env python3
"""
Converted test file for operators: {ops_str}

This file contains test functions converted from FlagGems tests.
Auto-generated by convert_flaggems_tests.py
"""'''


def main():
    parser = argparse.ArgumentParser(
        description="Convert FlagGems test functions to flagbench format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert tests for specific operators
  python scripts/convert_flaggems_tests.py --operators add,mul,sub

  # Read operators from file
  python scripts/convert_flaggems_tests.py --operators-file operators.txt

  # Specify output file
  python scripts/convert_flaggems_tests.py --operators add --output test_add.py
        """
    )

    parser.add_argument(
        '--operators',
        type=str,
        help='Comma-separated list of operator names to convert'
    )

    parser.add_argument(
        '--operators-file',
        type=Path,
        help='File containing operator names (one per line)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output file path (default: src/flagbench/accuracy/test_converted_ops.py)'
    )

    parser.add_argument(
        '--flaggems-dir',
        type=Path,
        default=FLAGGEMS_DIR,
        help='Path to FlagGems directory'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be converted without writing files'
    )

    args = parser.parse_args()

    # Get operator list
    operators = []
    if args.operators:
        operators = [op.strip() for op in args.operators.split(',')]
    elif args.operators_file:
        with open(args.operators_file, 'r') as f:
            operators = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Either --operators or --operators-file must be specified")

    # Set output path
    if args.output:
        output_file = args.output
    else:
        output_file = PROJECT_ROOT / "src" / "flagbench" / "accuracy" / "test_converted_ops.py"

    logger.info(f"Converting tests for operators: {operators}")
    logger.info(f"FlagGems directory: {args.flaggems_dir}")
    logger.info(f"Output file: {output_file}")

    # Initialize components
    flaggems_tests_dir = args.flaggems_dir / "tests"
    if not flaggems_tests_dir.exists():
        logger.error(f"FlagGems tests directory not found: {flaggems_tests_dir}")
        sys.exit(1)

    parser_obj = TestParser(flaggems_tests_dir)
    converter = TestConverter()
    writer = TestWriter(output_file.parent)

    # Find and convert test functions
    all_test_functions = []
    original_sources = {}

    for operator in operators:
        logger.info(f"Searching for tests for operator: {operator}")
        test_functions = parser_obj.find_operator_tests(operator)

        if not test_functions:
            logger.warning(f"No tests found for operator: {operator}")
            continue

        logger.info(f"Found {len(test_functions)} test functions for {operator}")
        all_test_functions.extend(test_functions)

        # Load original sources
        for func in test_functions:
            if func.source_file not in original_sources:
                with open(func.source_file, 'r', encoding='utf-8') as f:
                    original_sources[func.source_file] = f.read()

    # Extract constants and helper functions from all source files
    all_constants = {}
    all_helpers = {}

    for source_file, source_code in original_sources.items():
        try:
            tree, _ = parser_obj.parse_test_file(source_file)

            # Extract constants
            constants = parser_obj.extract_module_constants(tree, source_code)
            all_constants.update(constants)

            # Extract helper functions
            helpers = parser_obj.extract_helper_functions(tree, source_code)
            all_helpers.update(helpers)

            logger.info(f"Extracted {len(constants)} constants and {len(helpers)} helper functions from {source_file.name}")
        except Exception as e:
            logger.warning(f"Failed to extract from {source_file}: {e}")

    logger.info(f"Total extracted: {len(all_constants)} constants, {len(all_helpers)} helper functions")

    # Write converted tests
    if all_test_functions:
        writer.write_test_file(
            operators,
            all_test_functions,
            converter,
            original_sources,
            output_file,
            dry_run=args.dry_run,
            extracted_constants=all_constants,
            extracted_helpers=all_helpers
        )
        if not args.dry_run:
            logger.info(f"Conversion complete! Output: {output_file}")
    else:
        logger.error("No test functions found for any of the specified operators")
        sys.exit(1)


if __name__ == "__main__":
    main()
