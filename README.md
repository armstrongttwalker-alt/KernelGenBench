# KernelGenBench

A benchmark framework for evaluating LLM and agent-based Triton kernel generation on NVIDIA GPUs.

## Overview

KernelGenBench provides two evaluation tracks:

1. **LLM Track** (`scripts/generate_kernel_and_verify.py`): Evaluates LLMs on generating Triton kernels via Pass@K testing.
2. **Agent Track** (`agent_bench/test_ops.sh`): Evaluates coding agents on iteratively generating and verifying Triton kernels.

The benchmark covers 210 operators across three categories:
- **PyTorch ATen operators** (v2.1 subset, 110 ops)
- **vLLM operators** (50 ops)
- **cuBLAS operators** (50 ops)

## Requirements

- Python 3.10+
- PyTorch 2.x
- Triton
- NVIDIA GPU

```bash
pip install -r requirements.txt
pip install -e .
```

## LLM Track

Generate and verify Triton kernels using Pass@K:

```bash
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench \
    --test-type triton \
    --max-rounds 10 \
    --model-name gpt-4o \
    --device-count 8
```

Key arguments:
- `--dataset`: `KernelGenBench` (default, covers all 210 ops)
- `--max-rounds`: number of attempts per operator (Pass@K)
- `--model-name`: LLM model to use
- `--server-type`: API server type
- `--op-name`: test a single operator (e.g. `aten::add`)

## Agent Track

Run the agent benchmark:

```bash
cd agent_bench

# Test all operators
./test_ops.sh -d KernelGenBench

# Test a single operator
./test_ops.sh aten__add -d KernelGenBench

# Use a specific agent method
./test_ops.sh -d KernelGenBench -m naive_cc
```

### Configuration

Copy and edit the config file:

```bash
cp config.example.yaml config.yaml
```

Key fields in `config.yaml`:
- `agent.bin`: path to the Claude Code executable (`claude`)
- `agent.budget`: per-operator token budget (USD)
- `device.gpu_ids`: list of GPU IDs to use, or `null` for auto-detect

### Agent Methods

- `naive_cc`: single-shot generation with Claude Code (default)
- `normal_cc`: generation with tool use and verification feedback

## Results

Results are saved to `agent_bench/runs/<run_name>/`:
- `result.json`: per-operator pass/fail with error traces
- `kernels/`: generated Triton kernel files

## Project Structure

```
scripts/                    # LLM track scripts
  generate_kernel_and_verify.py
agent_bench/                # Agent track
  test_ops.sh               # Entry point
  run.py                    # Agent runner
  verify.py                 # Kernel verifier
  methods/                  # Agent methods
src/
  kernelgenbench/                # Benchmark dataset and test cases
    accuracy/               # Test functions
    dataset/                # Operator lists
  generator/                # LLM prompt builders and generators
  sandbox/                  # Kernel execution and verification
```
