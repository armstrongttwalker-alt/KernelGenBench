#!/bin/bash
# Example script demonstrating how to test qwen_next operators

# Set the project root
PROJECT_ROOT="/share/project/tj/workspace/flag-bench"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Qwen Next Operator Testing Examples"
echo "=========================================="

# Example 1: Test a single operator
echo -e "\n[Example 1] Testing a single operator..."
echo "Command: python tests/test_qwen_next_operator.py output/pass_at_k/round_0/aten::add.py"
# Uncomment to run:
# python tests/test_qwen_next_operator.py output/pass_at_k/round_0/aten::add.py

# Example 2: Test all operators in a directory
echo -e "\n[Example 2] Testing all operators in round_0..."
echo "Command: python tests/test_qwen_next_batch.py output/pass_at_k/round_0"
# Uncomment to run:
# python tests/test_qwen_next_batch.py output/pass_at_k/round_0

# Example 3: Test with parallel execution
echo -e "\n[Example 3] Testing with 4 parallel workers..."
echo "Command: python tests/test_qwen_next_batch.py output/pass_at_k/round_0 --max-workers 4"
# Uncomment to run:
# python tests/test_qwen_next_batch.py output/pass_at_k/round_0 --max-workers 4

# Example 4: Test specific pattern
echo -e "\n[Example 4] Testing only 'add' related operators..."
echo "Command: python tests/test_qwen_next_batch.py output/pass_at_k/round_0 --pattern '*add*.py'"
# Uncomment to run:
# python tests/test_qwen_next_batch.py output/pass_at_k/round_0 --pattern '*add*.py'

echo -e "\n=========================================="
echo "To run these examples, uncomment the commands in this script"
echo "=========================================="
