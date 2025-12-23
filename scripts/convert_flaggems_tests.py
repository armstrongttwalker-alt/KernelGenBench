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
        decorator_start = start_line
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('@'):
                decorator_start = i
            elif line and not line.startswith('#'):
                break

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

        for line in lines:
            # Convert @pytest.mark.{op_name} to @label("{op_name}")
            if '@pytest.mark.' in line:
                if '@pytest.mark.parametrize' in line:
                    # Convert @pytest.mark.parametrize to @parametrize
                    line = line.replace('@pytest.mark.parametrize', '@parametrize')
                elif '@pytest.mark.inplace' in line:
                    # Convert @pytest.mark.inplace to @label("inplace")
                    indent = line[:len(line) - len(line.lstrip())]
                    line = f'{indent}@label("inplace")'
                else:
                    # Extract the mark name and convert to @label
                    match = re.search(r'@pytest\.mark\.(\w+)', line)
                    if match:
                        mark_name = match.group(1)
                        indent = line[:len(line) - len(line.lstrip())]
                        line = f'{indent}@label("{mark_name}")'

            converted_lines.append(line)

        return '\n'.join(converted_lines)

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

        return code

    def generate_imports(self, test_functions: List[TestFunction], original_source: str) -> str:
        """Generate import statements for the converted file."""
        imports = []

        # Check if numpy or random are needed
        needs_numpy = 'np.' in original_source or 'numpy.' in original_source
        needs_random = 'random.' in original_source or 'import random' in original_source

        # Standard imports
        if needs_numpy:
            imports.append("import numpy as np")
        if needs_random:
            imports.append("import random")
        imports.append("import torch")
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
            if 'gems_assert_close' in original_source:
                utils_imports.append('gems_assert_close')
            if 'gems_assert_equal' in original_source:
                utils_imports.append('gems_assert_equal')
            if 'to_reference' in original_source:
                utils_imports.append('to_reference')

            # Check for dtype/shape constants
            dtype_constants = ['FLOAT_DTYPES', 'INT_DTYPES', 'ALL_FLOAT_DTYPES', 'ALL_INT_DTYPES', 'BOOL_TYPES']
            shape_constants = ['POINTWISE_SHAPES', 'SPECIAL_SHAPES', 'REDUCTION_SHAPES', 'REDUCTION_SMALL_SHAPES',
                               'STACK_SHAPES', 'SCALARS', 'UT_SHAPES_1D', 'UT_SHAPES_2D', 'KRON_SHAPES',
                               'DISTRIBUTION_SHAPES']

            for const in dtype_constants + shape_constants:
                if const in original_source:
                    utils_imports.append(const)

            if utils_imports:
                imports.append(f"from sandbox.utils.accuracy_utils import {', '.join(sorted(set(utils_imports)))}")

        # Add hardcoded definitions for constants not in accuracy_utils
        constant_definitions = {
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

        # Check which constants are used but not imported (in dependency order)
        needed_constants = []
        for const_name in ['DIMS_LIST', 'DIM_LIST', 'KEEPDIM_DIMS_SHAPE', 'KIND_KEEPDIM_DIMS_SHAPE',
                           'KEEPDIM_DIMS', 'KEEPDIM_DIM', 'EMPTY_SHAPES']:
            if const_name in original_source and const_name not in utils_imports:
                needed_constants.append(constant_definitions[const_name])

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
        dry_run: bool = False
    ):
        """Write converted test functions to a file.

        Args:
            operators: List of operator names
            test_functions: List of test functions to convert
            converter: TestConverter instance
            original_sources: Dictionary of original source files
            output_file: Output file path
            dry_run: If True, only preview without writing
        """
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
        imports = converter.generate_imports(test_functions, all_sources)

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
            logger.info(f"  - Test functions: {len(converted_functions)}")

            total_lines = (len(header.split('\n')) +
                          len(imports.split('\n')) +
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

    # Write converted tests
    if all_test_functions:
        writer.write_test_file(
            operators,
            all_test_functions,
            converter,
            original_sources,
            output_file,
            dry_run=args.dry_run
        )
        if not args.dry_run:
            logger.info(f"Conversion complete! Output: {output_file}")
    else:
        logger.error("No test functions found for any of the specified operators")
        sys.exit(1)


if __name__ == "__main__":
    main()
