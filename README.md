# KernelGenBench

A benchmark framework for evaluating LLM and agent-based Triton kernel generation across multiple hardware platforms.

## Features

- **210 operators** across three sources: ATen (110), vLLM (50), cuBLAS (50)
- **Multi-chip support**: NVIDIA, Ascend NPU, MUSA, Hygon DCU, Iluvatar, MetaX
- **Two evaluation tracks**: LLM Track (Pass@K) and Agent Track (iterative generation)
- **Multiple agent methods**: Claude Code, OpenCode, AutoKernel, AKO4ALL, cuda-optimized-skill
- **Automatic verification**: accuracy testing with tolerance-based comparison

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Configure API credentials:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=your_key

# OpenAI / OpenAI-compatible
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://your-endpoint/v1  # optional, for custom endpoints
```

## Supported Devices

Device type is auto-detected. Override with `GEMS_VENDOR` environment variable if needed.

| Device | Type | Visibility Env Var | `GEMS_VENDOR` |
|--------|------|-------------------|---------------|
| NVIDIA GPU | `cuda` | `CUDA_VISIBLE_DEVICES` | `nvidia` |
| Ascend NPU | `npu` | `ASCEND_RT_VISIBLE_DEVICES` | `ascend` |
| MUSA (Moore Threads) | `musa` | `MUSA_VISIBLE_DEVICES` | `mthreads` |
| Hygon DCU | `cuda` (HIP) | `HIP_VISIBLE_DEVICES` | `hygon` |
| Iluvatar GPU | `cuda` | `CUDA_VISIBLE_DEVICES` | `iluvatar` |
| MetaX (MUXI) GPU | `cuda` | `MACA_VISIBLE_DEVICES` | `muxi` |

All chips use the same commands — the framework handles device differences automatically.

## Datasets

| Dataset | Operators | Description |
|---------|-----------|-------------|
| `KernelGenBench` | 210 | Full set (ATen + vLLM + cuBLAS) |
| `KernelGenBench-aten` | 110 | ATen operators only |
| `KernelGenBench-vllm` | 50 | vLLM operators only |
| `KernelGenBench-cublas` | 50 | cuBLAS operators only (NVIDIA-only) |

On non-NVIDIA chips, the default dataset is automatically set to `KernelGenBench-aten` (cuBLAS operators require NVIDIA GPUs).

## LLM Track

Evaluate an LLM on generating Triton kernels with Pass@K metric:

```bash
# Single operator test
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3

# Full benchmark (all 210 operators)
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3

# Non-NVIDIA chips (ATen only)
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-aten \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--op-name` | Single operator to test (e.g., `aten::add`, `vllm13::rms_norm`) | All operators |
| `--single-test` | Randomly pick 1 operator for quick testing | Off |
| `--dataset` | Dataset to use (`KernelGenBench`, `KernelGenBench-aten`, `-vllm`, `-cublas`) | Auto-detect |
| `--server-type` | LLM provider (`openai`, `anthropic`) | `openai` |
| `--model-name` | Model name | `gpt-4o` |
| `--max-rounds` | Number of Pass@K rounds | 10 |
| `--device-count` | Number of GPUs for verification | 8 |
| `--timeout` | Timeout per operator (seconds) | 300 |
| `--temperature` | Sampling temperature | 0.8 |
| `--reflection` | Use previous round's errors as feedback | Off |
| `--resume-from` | Resume from existing checkpoint directory | - |
| `--debug` | Debug mode (only 8 operators) | Off |

## Agent Track

Evaluate coding agents that iteratively generate, verify, and fix kernels.

### Setup

```bash
cp agent_bench/config.example.yaml agent_bench/config.yaml
# Edit config.yaml:
#   - paths.python: path to Python with torch installed
#   - agent.bin: path to agent executable
```

### Methods

| Method | Description | Command |
|--------|-------------|---------|
| `naive_cc` | Single Claude Code call | `bash test_ops.sh add -m naive_cc` |
| `normal_cc` | Claude Code + self-verification loop | `bash test_ops.sh add -m normal_cc` |
| `naive_opencode` | Single OpenCode call | `bash test_ops.sh add -m naive_opencode` |
| `normal_opencode` | OpenCode + self-verification loop | `bash test_ops.sh add -m normal_opencode` |
| AutoKernel | Automated kernel optimization pipeline | `bash test_autokernel.sh add` |
| AKO4ALL | Kernel optimization for all operators | `bash test_ako4all.sh add` |
| cuda-optimized-skill | CUDA optimization with strategy memory | `bash test_cuda_optimized_skill.sh add` |

### Running

```bash
cd agent_bench

# Single operator
bash test_ops.sh add --device-count 1

# Multiple operators
bash test_ops.sh add,softmax,mul --device-count 4

# Full benchmark
bash test_ops.sh --device-count 8

# Specialized methods
bash test_autokernel.sh add --device-count 1
bash test_ako4all.sh add --device-count 1
bash test_cuda_optimized_skill.sh add --device-count 1
```

### Parameters (`test_ops.sh`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `[operators]` | Comma-separated operator names (positional) | All operators |
| `-d, --dataset` | Dataset to use | `KernelGenBench` |
| `-m, --method` | Agent method (`naive_cc`, `normal_cc`, `naive_opencode`, `normal_opencode`) | `normal_cc` |
| `--device-count` | Number of GPUs for verification | 8 |
| `--timeout` | Timeout per operator (seconds) | 600 |
| `--skip-gen` | Skip prompt generation step | Off |
| `--skip-verify` | Skip verification (only generate kernels) | Off |
| `-v, --verbose` | Enable verbose output | Off |

### Results

Results are saved to `agent_bench/runs/<run_name>/`:
- `progress.json` — real-time progress tracking
- `kernels/` — generated kernel files
- `results.json` — verification results

## Analyzing Results

```bash
# LLM track
python scripts/analyze/analyze.py output/pass_at_k/<run_dir>/

# Agent track
python scripts/analyze/analyze.py agent_bench/runs/<run_dir>/
```

## Project Structure

```
agent_bench/           # Agent Track framework
  methods/             # Agent methods (naive_cc, normal_cc, opencode, ...)
  templates/           # Prompt templates (generic + per-chip)
  tools/               # Verification tools
  config.example.yaml  # Configuration template
sota_agents/           # Specialized kernel generation agents
  AutoKernel/          # Automated kernel optimization
  AKO4ALL/             # Kernel optimization for all operators
  cuda-optimized-skill/  # CUDA optimization with strategy memory
src/
  kernelgenbench/      # Core package (accuracy tests, dataset, framework)
  generator/           # LLM prompt builders and samplers
  sandbox/             # Kernel verifier and anti-hack
  runtime/             # Device detection and constraints
scripts/               # LLM Track entry points and analysis tools
```

## Evaluating Custom Operators

To benchmark your own operators, add test cases to `src/kernelgenbench/accuracy/` and register them in the dataset. See [CONTRIBUTING.md](CONTRIBUTING.md) for step-by-step instructions.

## Contributing

We welcome contributions! You can:
- **Add new operators** — expand the benchmark with new test cases
- **Add new chip backends** — extend support to additional hardware
- **Add new agents** — integrate coding tools like Codex, Trae, Cursor
- **Add new agentic methods** — contribute specialized optimization pipelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guides.

## Related Projects

KernelGenBench is part of the [FlagOS](https://github.com/flagos-ai) open-source ecosystem:

| Project | Description |
|---------|-------------|
| [FlagGems](https://github.com/flagos-ai/FlagGems) | High-performance Triton operator library for multiple hardware backends |
| [FlagTree](https://github.com/flagos-ai/FlagTree) | Triton language extensions and multi-chip compilation infrastructure |
| [KernelGen](https://github.com/flagos-ai/KernelGen) | LLM/Agent-based Triton kernel generation method |

- **FlagGems** provides reference operator implementations and algorithm patterns used in KernelGenBench's accuracy tests.
- **FlagTree** extends Triton to support diverse hardware, enabling the multi-chip capability of KernelGenBench.
- **KernelGen** is our proposed agent method for automated kernel generation, evaluated on KernelGenBench.

## Citation

If you find KernelGenBench useful in your research or evaluation, please cite:

```bibtex
@software{kernelgenbench2026,
  title={KernelGenBench: A Benchmark for LLM and Agent-Based Triton Kernel Generation},
  author={KernelGen Team},
  url={https://github.com/flagos-ai/KernelGenBench},
  year={2026}
}
```

## License

This project is licensed under the MIT License.
