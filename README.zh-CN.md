# KernelGenBench

跨多种硬件平台评测 LLM 和 Agent 生成 Triton kernel 能力的基准框架。

## 特性

- **210 个算子**，涵盖三个来源：ATen (110)、vLLM (50)、cuBLAS (50)
- **多芯片支持**：NVIDIA、昇腾 NPU、MUSA、海光 DCU、天数智芯、沐曦
- **两条评测路径**：LLM Track（Pass@K）和 Agent Track（迭代生成）
- **多种 Agent 方法**：Claude Code、OpenCode、AutoKernel、AKO4ALL、cuda-optimized-skill
- **自动化验证**：基于容差的精度测试

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

配置 API 密钥：

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=your_key

# OpenAI / OpenAI 兼容接口
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://your-endpoint/v1  # 可选，自定义端点
```

## 支持的设备

设备类型自动检测。可通过 `GEMS_VENDOR` 环境变量手动指定。

| 设备 | 类型 | 可见性环境变量 | `GEMS_VENDOR` |
|------|------|--------------|---------------|
| NVIDIA GPU | `cuda` | `CUDA_VISIBLE_DEVICES` | `nvidia` |
| 昇腾 NPU | `npu` | `ASCEND_RT_VISIBLE_DEVICES` | `ascend` |
| MUSA（摩尔线程） | `musa` | `MUSA_VISIBLE_DEVICES` | `mthreads` |
| 海光 DCU | `cuda` (HIP) | `HIP_VISIBLE_DEVICES` | `hygon` |
| 天数智芯 GPU | `cuda` | `CUDA_VISIBLE_DEVICES` | `iluvatar` |
| 沐曦 GPU | `cuda` | `MACA_VISIBLE_DEVICES` | `muxi` |

所有芯片使用相同的命令，框架自动处理设备差异。

## 数据集

| 数据集 | 算子数 | 说明 |
|--------|--------|------|
| `KernelGenBench` | 210 | 完整集（ATen + vLLM + cuBLAS） |
| `KernelGenBench-aten` | 110 | 仅 ATen 算子 |
| `KernelGenBench-vllm` | 50 | 仅 vLLM 算子 |
| `KernelGenBench-cublas` | 50 | 仅 cuBLAS 算子（需要 NVIDIA GPU） |

在非 NVIDIA 芯片上，默认数据集自动设置为 `KernelGenBench-aten`（cuBLAS 算子需要 NVIDIA GPU）。

## LLM Track

使用 Pass@K 指标评测 LLM 生成 Triton kernel 的能力：

```bash
# 单算子测试
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3

# 完整测试（全部 210 个算子）
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3

# 非 NVIDIA 芯片（仅 ATen）
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-aten \
    --server-type openai \
    --model-name your-model-name \
    --max-rounds 3
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--op-name` | 指定单个算子（如 `aten::add`、`vllm13::rms_norm`） | 全部算子 |
| `--single-test` | 随机选 1 个算子快速测试 | 关闭 |
| `--dataset` | 数据集（`KernelGenBench`、`KernelGenBench-aten`、`-vllm`、`-cublas`） | 自动检测 |
| `--server-type` | LLM 提供商（`openai`、`anthropic`） | `openai` |
| `--model-name` | 模型名称 | `gpt-4o` |
| `--max-rounds` | Pass@K 轮数 | 10 |
| `--device-count` | 验证使用的 GPU 数量 | 8 |
| `--timeout` | 单算子超时时间（秒） | 300 |
| `--temperature` | 采样温度 | 0.8 |
| `--reflection` | 使用上一轮错误作为反馈 | 关闭 |
| `--resume-from` | 从已有检查点恢复 | - |
| `--debug` | 调试模式（仅 8 个算子） | 关闭 |

## Agent Track

评测编程 Agent 迭代生成、验证、修复 kernel 的能力。

### 配置

```bash
cp agent_bench/config.example.yaml agent_bench/config.yaml
# 编辑 config.yaml：
#   - paths.python: 已安装 torch 的 Python 路径
#   - agent.bin: Agent 可执行文件路径
```

### 方法

| 方法 | 说明 | 命令 |
|------|------|------|
| `naive_cc` | 单次 Claude Code 调用 | `bash test_ops.sh add -m naive_cc` |
| `normal_cc` | Claude Code + 自验证循环 | `bash test_ops.sh add -m normal_cc` |
| `naive_opencode` | 单次 OpenCode 调用 | `bash test_ops.sh add -m naive_opencode` |
| `normal_opencode` | OpenCode + 自验证循环 | `bash test_ops.sh add -m normal_opencode` |
| AutoKernel | 自动化 kernel 优化流水线 | `bash test_autokernel.sh add` |
| AKO4ALL | 全算子 kernel 优化 | `bash test_ako4all.sh add` |
| cuda-optimized-skill | 基于策略记忆的 CUDA 优化 | `bash test_cuda_optimized_skill.sh add` |

### 运行

```bash
cd agent_bench

# 单算子
bash test_ops.sh add --device-count 1

# 多算子
bash test_ops.sh add,softmax,mul --device-count 4

# 完整测试
bash test_ops.sh --device-count 8

# 专用方法
bash test_autokernel.sh add --device-count 1
bash test_ako4all.sh add --device-count 1
bash test_cuda_optimized_skill.sh add --device-count 1
```

### 参数说明 (`test_ops.sh`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `[operators]` | 逗号分隔的算子名（位置参数） | 全部算子 |
| `-d, --dataset` | 数据集 | `KernelGenBench` |
| `-m, --method` | Agent 方法（`naive_cc`、`normal_cc`、`naive_opencode`、`normal_opencode`） | `normal_cc` |
| `--device-count` | 验证使用的 GPU 数量 | 8 |
| `--timeout` | 单算子超时（秒） | 600 |
| `--skip-gen` | 跳过 prompt 生成步骤 | 关闭 |
| `--skip-verify` | 跳过验证（仅生成 kernel） | 关闭 |
| `-v, --verbose` | 详细输出 | 关闭 |

### 结果

结果保存在 `agent_bench/runs/<run_name>/`：
- `progress.json` — 实时进度
- `kernels/` — 生成的 kernel 文件
- `results.json` — 验证结果

## 结果分析

```bash
# LLM track
python scripts/analyze/analyze.py output/pass_at_k/<run_dir>/

# Agent track
python scripts/analyze/analyze.py agent_bench/runs/<run_dir>/
```

## 项目结构

```
agent_bench/           # Agent Track 框架
  methods/             # Agent 方法 (naive_cc, normal_cc, opencode, ...)
  templates/           # Prompt 模板（通用 + 各芯片专用）
  tools/               # 验证工具
  config.example.yaml  # 配置模板
sota_agents/           # 专用 kernel 生成 Agent
  AutoKernel/          # 自动化 kernel 优化
  AKO4ALL/             # 全算子 kernel 优化
  cuda-optimized-skill/  # 基于策略记忆的 CUDA 优化
src/
  kernelgenbench/      # 核心包（精度测试、数据集、框架）
  generator/           # LLM prompt 构建器和采样器
  sandbox/             # Kernel 验证器和反作弊
  runtime/             # 设备检测和约束
scripts/               # LLM Track 入口和分析工具
```

## 评测自定义算子

如需评测自己的算子，在 `src/kernelgenbench/accuracy/` 添加测试用例并注册到数据集。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 贡献指南

欢迎社区贡献！你可以：
- **添加新算子** — 扩展评测集
- **添加新芯片** — 支持更多硬件平台
- **添加新 Agent** — 集成 Codex、Trae、Cursor 等编程工具
- **添加新 Agentic 方法** — 贡献专用优化流水线

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 相关项目

KernelGenBench 是 [FlagOS](https://github.com/flagos-ai) 开源生态的一部分：

| 项目 | 说明 |
|------|------|
| [FlagGems](https://github.com/flagos-ai/FlagGems) | 高性能 Triton 算子库，支持多硬件后端 |
| [FlagTree](https://github.com/flagos-ai/FlagTree) | Triton 语言扩展与多芯片编译基础设施 |
| [KernelGen](https://github.com/flagos-ai/KernelGen) | 基于 LLM/Agent 的 Triton kernel 自动生成方法 |
| [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation) | AI 驱动 kernel 生成研究综述 |

- **FlagGems** 提供参考算子实现和算法模式，用于 KernelGenBench 的精度测试。
- **FlagTree** 扩展 Triton 以支持多种硬件，为 KernelGenBench 的多芯片能力提供基础。
- **KernelGen** 是我们提出的 Agent 自动 kernel 生成方法，在 KernelGenBench 上进行评测。

## 引用

如果 KernelGenBench 对您的研究或评测有帮助，请引用：

```bibtex
@software{kernelgenbench2026,
  title={KernelGenBench: A Benchmark for LLM and Agent-Based Triton Kernel Generation},
  author={KernelGen Team},
  url={https://github.com/flagos-ai/KernelGenBench},
  year={2026}
}
```

## 许可证

本项目采用 MIT 许可证。
