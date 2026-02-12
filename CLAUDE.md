# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Rules

**CRITICAL: File Modification Policy**
- **DO NOT** modify, edit, or write to any files unless the user explicitly says to start development or make changes
- **WAIT** for clear user confirmation before making any code changes
- When analyzing bugs or issues, provide analysis and solutions but DO NOT implement them until explicitly instructed
- User must explicitly say "start development", "begin implementation", "make the changes", or similar clear instructions before any file modifications

## Project Overview

FlagBench is a benchmark framework for Triton kernel generation and verification. It supports automatic test case generation, accuracy validation, and performance testing for PyTorch operators implemented in Triton.

**Key Capabilities:**
- Generate Triton kernel implementations from PyTorch operator specifications
- Verify kernel accuracy against PyTorch reference implementations
- Benchmark kernel performance
- Support Pass@K testing for iterative kernel improvement
- Convert FlagGems tests to flagbench format

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Core Architecture

### Three-Layer Structure

1. **Generator Layer** (`src/generator/`)
   - `TritonKernelGenerator`: Generates Triton kernel code using LLM
   - `TestFuncGenerator`: Generates accuracy test functions
   - `BenchmarkFuncGenerator`: Generates performance benchmark functions
   - `TorchKernelGenerator`: Generates PyTorch implementations

2. **Sandbox/Verifier Layer** (`src/sandbox/`)
   - `Verifier`: Executes and validates generated kernels
   - `Register`: Manages operator registration with PyTorch's dispatch system
   - Runs tests in isolated processes with timeout protection
   - Collects accuracy metrics and performance benchmarks

3. **Benchmark Layer** (`src/flagbench/`)
   - `accuracy/`: Test definitions for accuracy validation
   - `perfermance/`: Performance benchmark definitions
   - `dataset/`: Operator lists and test case specifications
   - Uses `@parametrize` decorator for test case generation

### Key Concepts

**Operator Registration:**
- Uses PyTorch's `torch.library.Library` to override ATen operators
- `DISPATCH_TORCH_LIB` environment variable controls whether to dispatch to custom implementations
- `Register` class manages operator registration lifecycle

**Test Parametrization:**
- `@parametrize` decorator generates test cases from parameter specifications
- `@label` decorator marks tests for selective execution
- Test functions use `Param` objects to define input variations

**Verification Flow:**
1. Load generated kernel code dynamically
2. Register kernel with PyTorch dispatch
3. Run parametrized test cases
4. Compare results against PyTorch reference (with tolerance)
5. Collect pass/fail statistics and error traces

## Common Commands

### Running Tests

**Test single operator accuracy:**
```bash
python test/test_accuracy_ut.py --name <operator_name>
```

**Test multiple operators:**
```bash
python test/test_accuracy_ut.py --name abs,mul,div
```

**Test all operators:**
```bash
python test/test_accuracy_ut.py --name all
```

**With GPU control:**
```bash
python test/test_accuracy_ut.py --name abs --device-count 8 --timeout 300
```

### Generating Kernels

**Generate accuracy test functions:**
```bash
python scripts/generate_ut_sample.py
```

**Generate Triton kernels from test results:**
```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 python scripts/generate_sample.py \
    --test-func-result-path <result_path>
```

**Pass@K workflow (generate and verify iteratively):**
```bash
python scripts/generate_ut_and_verify.py \
    --name aten \
    --test-type accuracy \
    --max-rounds 10
```

### Verification

**Verify generated kernels:**
```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 \
python scripts/eval_from_path_with_test_func.py \
    --path <triton_code_dir> \
    --num-samples 10 \
    --device-count 8 \
    --timeout 300 \
    --test-func-path <test_func_path>
```

**Test updated accuracy tests:**
```bash
python scripts/test_updated_accuracy_ut.py \
    --path <path_from_generation> \
    --device-count 8
```

### Converting FlagGems Tests

```bash
python scripts/convert_flaggems_tests.py \
    --operator <operator_name> \
    --output-dir <output_directory>
```

## Important Environment Variables

- `FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1`: Enable dynamic implementation info for kernel generation
- `FLAGBENCH_SKIP_BOTH_TEST=1`: Skip redundant double testing during verification
- `DISPATCH_TORCH_LIB=0`: Disable custom operator dispatch (use PyTorch reference)
- `FLAGBENCH_UPCAST=1`: Use float64 for reference computations (improves tolerance matching)
- `FLAGBENCH_ENABLE_DEVICE_CONSTRAINTS=1`: Enable device-specific prompt constraints for NPU/MUSA (default: enabled)
- `GEMS_VENDOR`: Override device vendor detection (values: `nvidia`, `ascend`, `mthreads`)

## Directory Structure

```
output/          # Generated Triton kernels
output_ut/       # Generated unit tests
runs/            # Verification logs and results
cache/           # Cached intermediate results
FlagGems/        # FlagGems submodule (reference implementations)
```

## Working with Qwen Next Operators

The repository includes special support for Qwen model operators:

**Test single Qwen operator:**
```bash
python tests/test_qwen_next_operator.py <kernel_file> \
    --device-count 1 \
    --timeout 300
```

**Operator definitions:**
- `src/flagbench/accuracy/test_qwen_next_ops.py`: Test cases for 94 Qwen operators
- `src/flagbench/dataset/kernel_list.py`: Contains `QWEN_NEXT_OPERATORS` dictionary

**Batch testing:**
```bash
python tests/test_qwen_next_batch.py <directory> --device-count 8
```

## Key Implementation Patterns

### Triton Kernel Requirements

**Complete ATen Interface Coverage:**
- Implement ALL overload variants (check with `torch.ops.aten.<op>.overloads()`)
- Common variants: `default`, `out`, `Scalar`, `Tensor`, `dimname`
- Example: `add` needs `add_Tensor`, `add_Scalar`, `add_out`, etc.

**Scalar Handling:**
- Pass scalars directly as kernel parameters (not as tensors)
- Prevents precision loss from tensor conversion
- Need separate kernels for: tensor⊗tensor, tensor⊗scalar, scalar⊗tensor

**Type Promotion:**
```python
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND

computation_dtype, result_dtype = elementwise_dtypes(
    self, other,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT  # or INT_TO_FLOAT
)
```

**Precision Control:**
- Use `allow_tf32=False` for matrix operations to maintain FP32 precision
- Accumulate in float32 for float16/bfloat16 inputs
- Example: `acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)`

### Common Pitfalls

1. **Recursion in overrides**: Calling `torch.op()` inside override triggers infinite recursion
   - Solution: Use CPU fallback or implement all logic in Triton

2. **Missing kernel variants**: Forgetting scalar⊗tensor variant
   - Solution: Always implement 3 variants for binary ops

3. **TF32 precision loss**: Default `tl.dot()` uses TF32 mode
   - Solution: Explicitly set `allow_tf32=False`

4. **Incomplete ATen interfaces**: Missing `_out` or `_dimname` variants
   - Solution: Check all overloads and implement each one

## FlagGems Integration

FlagGems is included as a submodule at `FlagGems/`. It provides:
- Reference implementations for complex operators
- Algorithm patterns for Triton kernels
- Validation targets for accuracy testing

**Using FlagGems as reference (when needed):**
```python
import sys
sys.path.insert(0, '/path/to/FlagGems/src')
from flag_gems.ops.<op> import <op> as flaggems_<op>
```

**Note:** The goal is FlagGems-free implementations, but FlagGems can be used for:
- Understanding algorithm structure
- Validating target accuracy
- Complex operators where custom implementation is impractical

## Testing Strategy

1. **Verify target is achievable**: Test FlagGems wrapper first to confirm 100% pass rate is possible
2. **Implement all variants**: Ensure complete ATen interface coverage
3. **Handle edge cases**: Empty tensors, broadcasting, different dtypes
4. **Validate incrementally**: Test after each major change
5. **Check all dtypes**: float16, bfloat16, float32 (and int types where applicable)

## Output Interpretation

**Verification results** are in `runs/` with structure:
```
runs/<run_name>/
├── log_0/result.json    # Detailed test results
├── log_1/result.json
└── ...
```

**Result JSON format:**
```json
{
  "op_name": "add",
  "success": true,
  "passed_tests": 650,
  "failed_tests": 0,
  "total_tests": 650,
  "traceback": "..."  // if failed
}
```
