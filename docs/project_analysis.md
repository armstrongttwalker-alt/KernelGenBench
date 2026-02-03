# FlagBench 项目分析

## 项目定位

FlagBench 是一个 Triton kernel 生成和验证的基准测试框架，核心目标是使用 LLM 自动生成 PyTorch ATen 算子和 CuPy/cuBLAS 算子的 Triton 实现，并通过自动化测试验证其正确性和性能。支持 Pass@K 迭代测试流程。

## 项目架构

```
flag-bench/
├── src/                              # 源代码目录
│   ├── flagbench/                   # 核心 benchmark 功能
│   │   ├── accuracy/                # 准确性测试定义
│   │   ├── perfermance/             # 性能测试定义
│   │   ├── dataset/                 # 算子列表和数据集
│   │   ├── framework/               # 框架适配器
│   │   └── processing/              # 数据处理工具
│   ├── generator/                   # 代码生成器
│   │   ├── generator.py             # BaseGenerator 基类
│   │   ├── triton_kernel_generator.py
│   │   ├── test_func_generator.py
│   │   ├── benchmark_func_generator.py
│   │   ├── torch_kernel_generator.py
│   │   ├── prompt_builder.py        # PromptBuilder 基类
│   │   ├── torch_prompt_builder.py  # PyTorch 算子 Prompt 构建
│   │   ├── cupy_prompt_builder.py   # CuPy 算子 Prompt 构建
│   │   └── sampler/                 # LLM 采样器
│   └── sandbox/                     # 沙箱验证器
│       ├── verifier/                # 验证器核心
│       ├── register.py              # 算子注册
│       └── utils/                   # 工具函数
├── scripts/                          # 工具脚本
│   ├── generate_kernel_and_verify.py # 主要工作流脚本
│   ├── generate_ut_and_verify.py    # Pass@K 测试脚本
│   └── analyze/                     # 分析脚本
├── test/                             # 测试入口
├── FlagGems/                         # FlagGems 子模块（参考实现）
├── output/                           # Triton kernel 生成输出
├── output_ut/                        # 单元测试生成输出
└── runs/                             # 验证运行结果
```

## 核心概念

### 三层架构

| 层级 | 模块 | 说明 |
|------|------|------|
| **Generator** | `src/generator/` | 使用 LLM 生成 Triton kernel、测试函数、性能测试函数 |
| **Sandbox/Verifier** | `src/sandbox/` | 执行和验证生成的 kernel，管理算子注册 |
| **Benchmark** | `src/flagbench/` | 测试用例定义、算子列表、参数化测试 |

### 核心数据结构

**VerifyResult**（验证结果）:
```python
class VerifyResult(BaseModel):
    op_name: str = None             # 算子名称
    success: Optional[bool] = None  # 是否成功
    traceback: Optional[str] = None # 错误追踪
    params: Optional[dict] = None   # 测试参数
    speedup: Optional[List] = None  # 性能数据
    info: Optional[dict] = None     # 附加信息
    code: Optional[str] = None      # 生成的代码
    test_func: Optional[str] = None # 测试函数代码
```

**VerifyConfig**（验证配置）:
```python
@dataclass
class VerifyConfig:
    run_name: str                   # 运行名称
    test_type: str = "accuracy"     # 测试类型: accuracy, performance, both
    run_dir: str = "runs"           # 运行结果目录
    store_type: str = "local"       # 存储类型
    strict_check: bool = False      # 严格检查模式
    seed: int = 42                  # 随机种子
    sample_id: int = 0              # 样本 ID
    save_log: bool = True           # 是否保存日志
    acc_timeout: int = 300          # 准确性测试超时（秒）
    perf_timeout: int = 600         # 性能测试超时（秒）
```

### 算子注册机制

使用 PyTorch 的 `torch.library.Library` 覆盖 ATen 算子：
- `DISPATCH_TORCH_LIB=1`: 使用生成的 Triton 实现
- `DISPATCH_TORCH_LIB=0`: 使用 PyTorch 原生实现（用于 baseline 对比）

### 测试参数化

使用 `@parametrize` 和 `@label` 装饰器生成测试用例：
```python
@label("add")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_add(shape, dtype):
    ...
```

## 数据集（算子）

### 数据集版本

| 数据集 | 说明 | 算子数量 |
|--------|------|----------|
| `V1_OPERATORS` | 基础算子集 | 40 |
| `V2_OPERATORS` | 扩展算子集 | 50 |
| `QWEN_NEXT_OPERATORS` | Qwen 模型专用算子 | 20 |
| `NON_FLAGGEMS_OPERATORS` | FlagGems 不支持的算子 | 10 |
| `CUPY_OPERATORS` | CuPy/cuBLAS 算子 | 47 |

### 算子分类

基于 `src/flagbench/accuracy/` 目录下的测试文件：

| 类别 | 测试文件 | 说明 |
|------|----------|------|
| 二元逐点运算 | `test_binary_pointwise_ops.py` | add, sub, mul, div 等 |
| 一元逐点运算 | `test_unary_pointwise_ops.py` | abs, cos, sin, exp 等 |
| 归约运算 | `test_reduction_ops.py` | sum, mean, max, min 等 |
| 通用归约运算 | `test_general_reduction_ops.py` | argmax, argmin, all, any 等 |
| BLAS 运算 | `test_blas_ops.py` | mm, bmm, addmm 等 |
| 归一化运算 | `test_norm_ops.py` | batch_norm, layer_norm 等 |
| 注意力运算 | `test_attention_ops.py` | scaled_dot_product_attention |
| 分布运算 | `test_distribution_ops.py` | dropout, exponential_ 等 |
| 张量构造 | `test_tensor_constructor_ops.py` | arange, zeros, ones 等 |
| 特殊运算 | `test_special_ops.py` | embedding, pad, sort 等 |
| V2 扩展算子 | `test_v2_ops.py` | V2_OPERATORS 中的算子测试 |
| Qwen 算子 | `test_qwen_next_ops.py` | Qwen 模型专用算子测试 |
| 非 FlagGems 算子 | `test_non_flaggems_ops.py` | FlagGems 不支持的算子 |

### 算子命名规则

格式: `{namespace}::{op_name}` 或 `torch.ops.{namespace}.{op_name}`

| 格式 | 示例 | 说明 |
|------|------|------|
| `aten::add` | PyTorch ATen 算子 | 标准 PyTorch 算子 |
| `cupy::sgemm` | CuPy cuBLAS 算子 | cuBLAS 线性代数算子 |

### IMPL_INFO 结构

记录算子的所有 overload 变体：
```python
IMPL_INFO = {
    "add": [("add.Tensor", Autograd.disable)],
    "div": [
        ("div.Tensor", Autograd.disable),
        ("div.Scalar", Autograd.disable),
        ("div.Tensor_mode", Autograd.disable),
        ("div.Scalar_mode", Autograd.disable),
    ],
    ...
}
```

## 评测流程

### 评测标准

评测逻辑位于 `src/sandbox/utils/accuracy_utils.py`。

**容差配置**（RESOLUTION）:
```python
RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1e-5,
    torch.float64: 1e-5,
    torch.bfloat16: 0.016,
    ...
}
```

**断言函数**:
```python
def assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1):
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol, equal_nan=equal_nan)
```

### 验证流程

1. **代码检查**: 验证生成代码的语法正确性，检查是否包含所有 overload 函数
2. **算子注册**: 使用 `torch.library.Library` 注册生成的实现
3. **参数化测试**: 运行 `@parametrize` 定义的所有测试用例
4. **结果对比**: 与 PyTorch 原生实现对比，检查容差
5. **结果汇总**: 统计通过/失败数量，记录错误信息

### 验证状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | `True` | 所有测试通过 |
| `success` | `False` | 存在失败测试 |
| `success` | `None` | 未找到测试用例 |

### 输出结构

```
runs/<run_name>/
├── log_0/
│   ├── result.json              # 详细测试结果
│   ├── summary.json             # 汇总统计
│   └── test_report_*.json       # 单算子测试报告
└── log_1/
    └── ...
```

**result.json 格式**:
```json
{
  "op_name": "add",
  "success": true,
  "info": {
    "total": 650,
    "failed": 0,
    "success": 650
  }
}
```

## Kernel Generator Pipeline

### 架构

```
Operator ──▶ Prompt ──▶ LLM ──▶ Code ──▶ Verify ──▶ Result
                ▲                                      │
                └─────── Reflection Prompt ◀───────────┘
```

### 生成器类型

| 生成器 | 类 | 说明 |
|--------|---|------|
| Triton Kernel | `TritonKernelGenerator` | 生成 Triton kernel 实现 |
| Test Function | `TestFuncGenerator` | 生成准确性测试函数 |
| Benchmark Function | `BenchmarkFuncGenerator` | 生成性能测试函数 |
| Torch Kernel | `TorchKernelGenerator` | 生成 PyTorch 实现 |

### Pass@K 测试流程

**工作流程**（`scripts/generate_kernel_and_verify.py`）:
1. 第一轮: 为所有算子生成代码
2. 验证: 运行测试，收集结果
3. 后续轮次: 仅对失败算子重新生成
4. 汇总: 统计 Pass@K 结果

**支持模式**:
- `--test-type triton`: 生成 Triton kernel
- `--test-type accuracy`: 生成准确性测试函数
- `--test-type performance`: 生成性能测试函数

### Prompt 构成

**初始 Prompt**（`TorchPromptBuilder.build_new`）:
- 示例代码（add kernel）
- PyTorch 函数代码
- 输入输出参数说明
- ATen overload 变体列表
- 关键要求（广播、非连续张量处理）
- Wiki 参考实现（可选）

**修复/优化 Prompt**（`TorchPromptBuilder.build_fix`）:
- 原始 Prompt 内容
- 当前生成的代码
- 错误信息和测试参数
- 修复建议

### Reflection 模式

启用 `--reflection` 时，使用上一轮的失败结果作为下一轮的反馈：
- 失败代码和错误追踪
- 失败的测试参数
- AI 建议（`TritonKernelAdviceGenerator`）

### 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-rounds` | Pass@K 轮数 | 10 |
| `--device-count` | 并行 GPU 数量 | 8 |
| `--timeout` | 单测试超时（秒） | 300 |
| `--model-name` | LLM 模型 | gpt-4o-mini |
| `--temperature` | 采样温度 | 0.8 |
| `--dataset` | 数据集版本（pytorch/gems/v1/v2/qwen_next/cupy） | v2 |

### 使用示例

```bash
# 生成 Triton kernel 并验证
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 python scripts/generate_kernel_and_verify.py \
    --name aten \
    --dataset v2 \
    --max-rounds 10 \
    --device-count 8

# 启用 Reflection 模式
python scripts/generate_kernel_and_verify.py \
    --name aten \
    --test-type triton \
    --reflection \
    --max-rounds 10
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `FLAGBENCH_USE_DYNAMIC_IMPL_INFO` | 动态获取算子 overload 信息 | 0 |
| `FLAGBENCH_SKIP_BOTH_TEST` | 跳过重复测试 | 0 |
| `DISPATCH_TORCH_LIB` | 使用自定义算子实现 | 1 |
| `FLAGBENCH_UPCAST` | 使用 float64 计算参考值 | 1 |

## FlagGems 集成

FlagGems 作为子模块提供参考实现：
- 路径: `FlagGems/`
- 用途: 算法参考、验证目标、复杂算子 fallback
- 测试转换: `scripts/convert_flaggems_tests.py`
