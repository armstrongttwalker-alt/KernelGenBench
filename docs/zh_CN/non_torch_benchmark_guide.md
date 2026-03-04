# 非 Torch API Benchmark 构建指南

本文档说明如何为非 PyTorch API（如 CuPy、cuBLAS、cuDNN 等）构建 benchmark 题目。

## 概述

非 Torch API 的 benchmark 包括两个核心部分：
1. **Baseline 实现**：放在 `src/flagbench/dataset/baseline/cupy/` 目录
2. **测试文件**：放在 `src/flagbench/accuracy/` 目录

系统会自动发现测试函数并自动加载 baseline 实现。

---

## 第一部分：创建 Baseline 实现

### 1.1 命名规范

使用 `::` 作为分类符：
- 格式：`<category>::<operator_name>`
- 示例：`cupy::add`、`cupy::matmul`

### 1.2 文件位置

将 baseline 实现放在：
```
src/flagbench/dataset/baseline/cupy/<operator_name>.py
```

**示例：** `cupy::add` 的 baseline 应放在：
```
src/flagbench/dataset/baseline/cupy/add.py
```

### 1.3 Baseline 代码结构

```python
import torch

def cupy_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    CuPy add baseline implementation

    Args:
        x: Input tensor
        y: Input tensor

    Returns:
        Result tensor
    """
    # 使用 CuPy 实现
    import cupy as cp

    x_cp = cp.asarray(x.detach().cpu().numpy())
    y_cp = cp.asarray(y.detach().cpu().numpy())
    result_cp = cp.add(x_cp, y_cp)

    return torch.from_numpy(cp.asnumpy(result_cp)).to(x.device)
```

**注意事项：**
- 函数名格式：`<category>_<operator_name>`（用 `_` 连接）
- 必须接受 torch.Tensor 作为输入
- 必须返回 torch.Tensor
- 内部可以使用任何第三方库（CuPy、cuBLAS 等）

---

## 第二部分：创建测试文件

### 2.1 文件位置

在 `src/flagbench/accuracy/` 目录创建测试文件：
```
src/flagbench/accuracy/test_<category>_<operator>_ops.py
```

**示例：**
```
src/flagbench/accuracy/test_cupy_add_ops.py
```

### 2.2 测试代码结构

```python
import torch
import flagbench
from sandbox.verifier.test_parametrize import parametrize, label
from torch.testing import assert_close
from sandbox.config import DEVICE as device
from sandbox.config import QUICK_MODE

# 测试参数配置
SHAPES = [(2, 3), (128, 256)] if QUICK_MODE else [(2, 3), (128, 256), (512, 512)]
DTYPES = [torch.float32] if QUICK_MODE else [torch.float32, torch.float16, torch.bfloat16]

@label("cupy::add")
@parametrize("shape", SHAPES)
@parametrize("dtype", DTYPES)
def test_accuracy_cupy_add(shape, dtype):
    """Test cupy::add: compare baseline vs triton implementation"""
    x = torch.randn(shape, dtype=dtype, device=device)
    y = torch.randn(shape, dtype=dtype, device=device)

    # 调用 baseline 实现（自动加载）
    baseline_result = flagbench.baseline.cupy_add(x, y)

    # 调用 triton 实现
    triton_result = flagbench.triton.cupy_add(x, y)

    # 比较 baseline vs triton
    assert_close(triton_result, baseline_result, rtol=1e-3, atol=1e-3)
```

**关键要素：**
- `@label("cupy::add")`：使用 `::` 格式标记算子
- `@parametrize`：定义测试参数组合
- `flagbench.baseline.<func_name>`：调用 baseline（自动加载）
- `flagbench.triton.<func_name>`：调用 triton 实现
- `assert_close`：比较结果

### 2.3 注册测试模块

在 `src/flagbench/__init__.py` 中注册测试模块：

```python
accuracy_modules = [
    # ... 其他模块 ...
    "flagbench.accuracy.test_cupy_add_ops",  # 添加新模块
]
```

---

## 完整示例：cupy::matmul

### Step 1: 创建 baseline

**文件：** `src/flagbench/dataset/baseline/cupy/matmul.py`

```python
import torch
import cupy as cp

def cupy_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CuPy matmul baseline implementation"""
    a_cp = cp.asarray(a.detach().cpu().numpy())
    b_cp = cp.asarray(b.detach().cpu().numpy())
    result_cp = cp.matmul(a_cp, b_cp)
    return torch.from_numpy(cp.asnumpy(result_cp)).to(a.device)
```

### Step 2: 创建测试文件

**文件：** `src/flagbench/accuracy/test_cupy_matmul_ops.py`

```python
import torch
import flagbench
from sandbox.verifier.test_parametrize import parametrize, label
from torch.testing import assert_close
from sandbox.config import DEVICE as device

SHAPES = [(32, 64, 128), (128, 256, 512)]
DTYPES = [torch.float32, torch.float16]

@label("cupy::matmul")
@parametrize("m,k,n", SHAPES)
@parametrize("dtype", DTYPES)
def test_accuracy_cupy_matmul(m, k, n, dtype):
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    baseline_result = flagbench.baseline.cupy_matmul(a, b)
    triton_result = flagbench.triton.cupy_matmul(a, b)

    assert_close(triton_result, baseline_result, rtol=1e-3, atol=1e-3)
```

### Step 3: 注册模块

在 `src/flagbench/__init__.py` 添加：
```python
"flagbench.accuracy.test_cupy_matmul_ops",
```

---

## 自动发现机制

系统会自动：
1. **发现测试函数**：从 `accuracy/` 目录加载所有 `@label` 标记的测试
2. **加载 baseline**：根据 `@label("cupy::matmul")` 自动加载 `baseline/cupy/matmul.py`
3. **注册命名空间**：
   - `flagbench.baseline.cupy_matmul` → 总是调用 baseline
   - `flagbench.triton.cupy_matmul` → 根据 `DISPATCH_TORCH_LIB` 调用 triton 或 baseline

---

## 环境变量控制

- `DISPATCH_TORCH_LIB=1`（默认）：`flagbench.triton.XXX` 调用 triton 实现
- `DISPATCH_TORCH_LIB=0`：`flagbench.triton.XXX` 调用 baseline 实现
- `flagbench.baseline.XXX` 不受环境变量影响，总是调用 baseline

---

## 运行测试

> **注意**: `test_accuracy_ut.py` 用于验证**测试函数本身**是否正常工作，而非测试算子实现的正确性。

```bash
# 列出 cupy 测试集的所有算子
python test/test_accuracy_ut.py --list-ops cupy

# 测试单个算子
python test/test_accuracy_ut.py --test-set cupy --name sgemm

# 测试多个算子
python test/test_accuracy_ut.py --test-set cupy --name saxpy,sgemm,dgemm

# 测试所有 cupy 算子
python test/test_accuracy_ut.py --test-set cupy --name all

# 使用多 GPU
python test/test_accuracy_ut.py --test-set cupy --name sgemm --device-count 8
```

---

## 注意事项

1. **命名一致性**：
   - Label 使用 `::`：`@label("cupy::add")`
   - 函数名使用 `_`：`def cupy_add(...)`
   - 文件路径使用 `/`：`baseline/cupy/add.py`

2. **Baseline 要求**：
   - 必须是正确的参考实现
   - 性能不重要，正确性最重要
   - 可以使用任何第三方库

3. **测试参数**：
   - 使用 `QUICK_MODE` 支持快速测试
   - 覆盖不同 shape、dtype 组合
   - 考虑边界情况（空张量、标量等）

4. **容差设置**：
   - 根据算子特性调整 `rtol` 和 `atol`
   - 浮点运算通常使用 `rtol=1e-3, atol=1e-3`
   - 整数运算可以使用更严格的容差
