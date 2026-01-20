# TODO: 添加非 PyTorch 算子支持（cuBLAS 等）

## 目标
支持非 PyTorch 算子（cuBLAS、cuDNN 等）的测试，通过 `flagbench.baseline.XXX` 和 `flagbench.triton.XXX` 调用不同实现并比较结果。

## 核心设计

**关键区别**：
- PyTorch 算子：使用 `torch.library.Library` dispatch，调用 `torch.xxx()` 自动路由
- 非 PyTorch 算子：仅命名空间注册，显式调用 `flagbench.baseline/triton.xxx()`

**算子类型识别**：通过 `IMPL_INFO.get(api)` 判断
- 存在 → PyTorch 算子
- 不存在 → 非 PyTorch 算子

## 关键区别表

| 特性 | PyTorch 算子 | 非 PyTorch 算子 |
|------|-------------|----------------|
| 注册机制 | torch.library.Library | 仅命名空间注册 |
| 调用方式 | torch.xxx() (隐式) | flagbench.baseline/triton.xxx() (显式) |
| Dispatch | 自动 | 手动 |
| IMPL_INFO | 必须存在 | 可以不存在 |
| namespace 参数 | 可选 | 必须（baseline/triton） |

---

## 实施步骤

### 阶段 1：修改 `src/sandbox/register.py`

#### 1.1 修改 `Register.register_impl()` 方法

**位置**：`src/sandbox/register.py:95-122`

**背景说明**：
- `@register()` 装饰器（lines 17-56）已经处理了函数注册到 `flagbench` 模块的 `setattr` 操作，包括 namespace 支持（lines 47-53）
- 注释代码（lines 104-108）是之前的方案，但其 `setattr` 操作与 `@register()` 重复，且不支持 namespace，无需恢复
- `register_impl()` 的职责是将 PyTorch 算子注册到 `torch.library`，非 PyTorch 算子不需要此步骤

**修改内容**：
1. 将 `impl_info = IMPL_INFO.get(api)` 检查移到方法最前面（在 try 块内）
2. 如果 `impl_info` 为 None（非 PyTorch 算子）：
   - 记录 info 级别日志：`logging.info(f"Non-PyTorch operator {key} (api={api}), skip torch.library registration")`
   - 直接 return，跳过后续的 `self.lib.impl()` 调用
3. 将 `self.all_ops.append(key)` 从 line 100 移到 IMPL_INFO 检查之后
   - 这样 `all_ops` 只记录注册到 `torch.library` 的 PyTorch 算子
4. 保持 PyTorch 算子的原有 `torch.library` 注册逻辑不变

**测试要点**：
- [ ] PyTorch 算子仍然正常注册到 torch.library
- [ ] 非 PyTorch 算子不会尝试调用 `self.lib.impl()`
- [ ] 非 PyTorch 算子不会被添加到 `self.all_ops`（只有 PyTorch 算子在 all_ops 中）
- [ ] 非 PyTorch 算子可以通过 `flagbench.baseline.XXX` 或 `flagbench.triton.XXX` 调用（由 @register() 装饰器完成）

#### 1.2 验证 `register()` 装饰器

**位置**：`src/sandbox/register.py:17-56`

**结论**：现有 namespace 参数已支持需求，无需修改

---

### 阶段 2：修改 `src/sandbox/verifier/verifier.py`

#### 2.1 修改 `_check_code()` 方法

**位置**：`src/sandbox/verifier/verifier.py:235-278`

**修改内容**：

**步骤 2.1.1：添加算子类型判断**
1. 添加 `is_pytorch_op = IMPL_INFO.get(name) is not None` 判断
2. PyTorch 算子：检查所有 overload 变体（保持现有逻辑）
3. 非 PyTorch 算子：只检查主函数名，创建 `ops = [(name, None)]` 统一处理

**步骤 2.1.2：修改装饰器添加逻辑**
1. 统一处理 ops 列表：`op_name = op[0] if isinstance(op, tuple) else op`
2. 为每个算子添加 `@register` 装饰器，传递 namespace 参数
3. **对于非 PyTorch 算子，根据 DISPATCH_TORCH_LIB 环境变量调整 namespace**：
   - `DISPATCH_TORCH_LIB=0`：将 namespace="baseline" 的代码注册到 "triton" 空间
   - `DISPATCH_TORCH_LIB=1`：保持 namespace="triton" 不变
   - 这样测试函数可以统一调用 `flagbench.triton.XXX`，通过环境变量切换实现

**测试要点**：
- [ ] PyTorch 算子的 overload 检查正常
- [ ] 非 PyTorch 算子不会因 IMPL_INFO 缺失报错
- [ ] 装饰器正确添加

#### 2.2 验证 `_verify()` 方法

**位置**：`src/sandbox/verifier/verifier.py:543-602`

**结论**：现有实现已支持多个 Source，无需修改

**使用说明**：
对于非 PyTorch 算子，调用 verifier 时需要传入**两个 Source**：
1. baseline 实现：`Source(source=baseline_code, function_name=name, namespace="baseline")`
2. triton 实现：`Source(source=triton_code, function_name=name, namespace="triton")`

**DISPATCH_TORCH_LIB 行为**：
- `DISPATCH_TORCH_LIB=0`：baseline 代码注册到 `flagbench.triton.XXX`（用于测试 baseline）
- `DISPATCH_TORCH_LIB=1`：triton 代码注册到 `flagbench.triton.XXX`（用于测试 triton）
- 测试函数统一调用 `flagbench.triton.XXX()`，通过环境变量切换实现

#### 2.3 处理 DISPATCH_TORCH_LIB 环境变量

**位置**：`src/sandbox/verifier/verifier.py:564` 和 `_check_code()` 方法

**新的 DISPATCH_TORCH_LIB 语义**：
对于非 PyTorch 算子，DISPATCH_TORCH_LIB 控制使用哪个实现：
- `DISPATCH_TORCH_LIB=0`：使用 baseline 实现，注册到 `flagbench.triton.XXX`
- `DISPATCH_TORCH_LIB=1`：使用 triton 实现，注册到 `flagbench.triton.XXX`

**修改内容**：
1. 在 `_verify()` 或 `_check_code()` 中，根据 DISPATCH_TORCH_LIB 过滤 Source：
   - 检查是否是非 PyTorch 算子（通过 IMPL_INFO）
   - `DISPATCH_TORCH_LIB=0`：只处理 namespace="baseline" 的 Source，将其 namespace 改为 "triton"
   - `DISPATCH_TORCH_LIB=1`：只处理 namespace="triton" 的 Source
2. 确保 `_check_code()` 在有非 PyTorch 算子时总是被调用

**测试要点**：
- [ ] `DISPATCH_TORCH_LIB=1` 时，triton 实现注册到 `flagbench.triton.XXX`
- [ ] `DISPATCH_TORCH_LIB=0` 时，baseline 实现注册到 `flagbench.triton.XXX`
- [ ] 测试函数统一调用 `flagbench.triton.XXX()`，结果随环境变量切换

---

### 阶段 3：添加示例和测试

#### 3.1 创建 non_torch_prelu 示例

**目标**：使用 prelu 作为非 PyTorch 算子示例，验证完整流程

**已创建的文件**：
- `src/flagbench/dataset/baseline/example/baseline_prelu.py` - baseline 实现
- `src/flagbench/dataset/baseline/example/triton_prelu.py` - Triton kernel 实现
- `src/flagbench/dataset/baseline/example/test_prelu.py` - 准确性测试函数

**baseline_prelu.py 内容**：
```python
import torch

def non_torch_prelu(self: torch.Tensor, weight: torch.Tensor):
    """Baseline implementation using torch operations"""
    # Handle scalar weight
    if weight.numel() == 1 or weight.dim() == 0:
        return torch.where(self > 0, self, weight * self)

    # Handle per-channel weight
    # weight shape: (C,), self shape: (N, C, ...)
    if self.dim() < 2:
        raise ValueError("prelu: per-channel weight requires input with at least 2 dims (N, C, ...)")
    if weight.numel() != self.size(1):
        raise ValueError(f"prelu: weight.numel() ({weight.numel()}) must equal input.size(1) ({self.size(1)})")

    weight = weight.view(1, -1, *([1] * (self.dim() - 2)))
    return torch.where(self > 0, self, weight * self)
```

**注意**：不需要添加 `@register` 装饰器，Verifier 会根据传入的 namespace 参数自动添加

**triton_prelu.py 内容**：
用户提供的完整 Triton kernel 实现，需要修改函数名：
- 将函数名 `prelu` 改为 `non_torch_prelu`
- **不需要**添加 `@register` 装饰器，Verifier 会自动添加

**test_prelu.py 内容**：
```python
import torch
import flagbench
from flagbench.verifier.test_parametrize import label, parametrize
from torch.testing import assert_close

device = "cuda"

@label("non_torch_prelu")
@parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("weight_kind", ["scalar", "per_channel"])
def test_non_torch_prelu(shape, dtype, weight_kind):
    x = torch.randn(shape, dtype=dtype, device=device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=device)

    # 统一调用 flagbench.triton.non_torch_prelu
    # DISPATCH_TORCH_LIB=0 时调用 baseline
    # DISPATCH_TORCH_LIB=1 时调用 triton
    result = flagbench.triton.non_torch_prelu(x, w)

    # 与 PyTorch 参考实现比较
    ref = torch.nn.functional.prelu(x, w)
    assert_close(result, ref, rtol=1e-3, atol=1e-3)
```

**Verifier 调用示例**：
```python
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
import os

# 读取文件内容
baseline_code = open("src/flagbench/dataset/baseline/example/baseline_prelu.py").read()
triton_code = open("src/flagbench/dataset/baseline/example/triton_prelu.py").read()
test_code = open("src/flagbench/dataset/baseline/example/test_prelu.py").read()

# 创建 verifier
config = VerifyConfig(run_name="non_torch_prelu_test", test_type="accuracy")
verifier = Verifier(config)

# 测试 baseline（DISPATCH_TORCH_LIB=0）
os.environ["DISPATCH_TORCH_LIB"] = "0"
result_baseline = verifier.only_verify(
    name_source_map=[
        VerifyRequest(
            source=[
                Source(source=baseline_code, function_name="non_torch_prelu", namespace="baseline"),
                Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
            ],
            test_func=[test_code],
            test_func_mark="non_torch_prelu"
        )
    ]
)
print("Baseline test:", result_baseline)

# 测试 triton（DISPATCH_TORCH_LIB=1）
os.environ["DISPATCH_TORCH_LIB"] = "1"
result_triton = verifier.only_verify(
    name_source_map=[
        VerifyRequest(
            source=[
                Source(source=baseline_code, function_name="non_torch_prelu", namespace="baseline"),
                Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
            ],
            test_func=[test_code],
            test_func_mark="non_torch_prelu"
        )
    ]
)
print("Triton test:", result_triton)
```

**预期结果**：
- `DISPATCH_TORCH_LIB=0`：baseline 实现通过所有测试用例
- `DISPATCH_TORCH_LIB=1`：triton 实现通过所有测试用例
- 验证非 PyTorch 算子的注册、调用、测试流程正常工作

#### 3.2 创建单元测试

**文件**：`tests/test_non_pytorch_ops.py`

**测试内容**：
- 测试非 PyTorch 算子的注册
- 测试 Verifier 对非 PyTorch 算子的支持
- 测试混合 PyTorch 和非 PyTorch 算子

#### 3.3 测试检查清单

- [ ] 运行 `pytest tests/test_non_pytorch_ops.py -v`
- [ ] 验证 baseline 和 triton 命名空间正确创建
- [ ] 验证函数可以通过 `flagbench.baseline.XXX` 和 `flagbench.triton.XXX` 调用

---

### 阶段 4：集成测试和文档

#### 4.1 运行完整的 GEMM 示例

**测试步骤**：
1. 准备测试环境（安装依赖）
2. 运行准确性测试，验证 baseline 和 triton 结果一致
3. 运行性能测试，验证性能数据正确生成

**预期结果**：
- 所有测试用例通过
- 日志保存在 `runs/gemm_example/` 目录

#### 4.2 更新文档

**需要更新的文件**：
- `CLAUDE.md` - 添加非 PyTorch 算子的使用说明
- `examples/cublas/README.md` - 创建 GEMM 示例文档

**文档内容**：
- 使用方式说明
- 关键区别表格
- 示例代码引用

---

## 测试计划

### 单元测试

#### 测试 1：register() 装饰器
- [ ] 测试 PyTorch 算子注册（保持向后兼容）
- [ ] 测试非 PyTorch 算子注册到 baseline 命名空间
- [ ] 测试非 PyTorch 算子注册到 triton 命名空间
- [ ] 验证命名空间正确创建
- [ ] 验证函数可通过 `flagbench.baseline.XXX` 调用

#### 测试 2：Register.register_impl()
- [ ] 测试 PyTorch 算子调用 `torch.library.Library.impl()`
- [ ] 测试非 PyTorch 算子跳过 `torch.library` 注册
- [ ] 测试非 PyTorch 算子仍然添加到 `all_ops` 列表
- [ ] 测试错误处理（已注册的算子等）

#### 测试 3：Verifier._check_code()
- [ ] 测试 PyTorch 算子的 overload 检查
- [ ] 测试非 PyTorch 算子的函数名检查
- [ ] 测试 `@register` 装饰器自动添加
- [ ] 测试 namespace 参数正确传递
- [ ] 测试代码编译和保存

#### 测试 4：Verifier._verify()
- [ ] 测试单个 Source 的注册
- [ ] 测试多个 Source 的注册（baseline + triton）
- [ ] 测试 DISPATCH_TORCH_LIB 环境变量处理
- [ ] 测试混合 PyTorch 和非 PyTorch 算子

### 集成测试

#### 测试 5：完整的 GEMM 示例
- [ ] 运行 baseline_gemm.py 和 triton_gemm.py
- [ ] 运行 test_gemm.py 验证准确性
- [ ] 运行 benchmark_gemm.py 测试性能
- [ ] 验证日志和结果保存正确

#### 测试 6：与现有 PyTorch 算子的兼容性
- [ ] 运行现有的 PyTorch 算子测试
- [ ] 确保没有回归
- [ ] 验证 DISPATCH_TORCH_LIB 仍然正常工作

### 性能测试

#### 测试 7：性能基准
- [ ] 测试 GEMM 性能（不同矩阵大小）
- [ ] 比较 baseline 和 triton 的性能
- [ ] 生成性能报告

---

## 注意事项和最佳实践

### 1. 算子命名规范

**PyTorch 算子**：
- API 名称：`torch.ops.aten.add`
- Key：`add`（或具体的 overload 如 `add.Tensor`）
- Namespace：`None`

**非 PyTorch 算子**：
- API 名称：`cublas.gemm`、`cudnn.conv` 等
- Key：函数名（如 `gemm`、`conv`）
- Namespace：`baseline` 或 `triton`

### 2. IMPL_INFO 管理

- PyTorch 算子必须在 `IMPL_INFO` 中定义
- 非 PyTorch 算子不需要在 `IMPL_INFO` 中定义
- 如果需要，可以为非 PyTorch 算子添加 `IMPL_INFO` 条目以支持更复杂的场景

### 3. 测试函数编写

- PyTorch 算子：使用 `torch.xxx()` 隐式调用
- 非 PyTorch 算子：显式调用 `flagbench.baseline.xxx()` 和 `flagbench.triton.xxx()`

### 4. 环境变量

- `DISPATCH_TORCH_LIB=1`：启用 PyTorch dispatch（默认）
- `DISPATCH_TORCH_LIB=0`：禁用 PyTorch dispatch
- 对于非 PyTorch 算子，该环境变量不影响注册行为

### 5. 错误处理

**常见错误**：
1. **函数名不匹配**：确保代码中的函数名与 key 参数一致
2. **命名空间缺失**：非 PyTorch 算子必须指定 namespace
3. **IMPL_INFO 缺失**：PyTorch 算子必须在 IMPL_INFO 中定义
4. **导入错误**：确保 `from flagbench import register` 在代码中

**调试技巧**：
- 检查 `REGISTERED_OPS` 字典内容
- 使用 `hasattr(flagbench, "baseline")` 验证命名空间
- 使用 `hasattr(flagbench.baseline, "gemm")` 验证函数注册
- 查看 verifier 日志中的注册信息

### 6. 性能优化

- Baseline 实现应该是高效的参考实现
- Triton 实现应该针对 GPU 优化
- 使用合适的 block size 和 grid 配置
- 考虑不同数据类型的性能差异

---

## 实施检查清单

### 阶段 1：修改 register.py
- [ ] 1.1 修改 `Register.register_impl()` 方法
  - [ ] 添加 `impl_info` 检查
  - [ ] 对非 PyTorch 算子跳过 `torch.library` 注册
  - [ ] 保持 PyTorch 算子逻辑不变
  - [ ] 测试验证
- [ ] 1.2 验证 `register()` 装饰器
  - [ ] 确认 namespace 参数支持
  - [ ] 可选：添加文档字符串

### 阶段 2：修改 verifier.py
- [ ] 2.1 修改 `_check_code()` - 算子类型识别
  - [ ] 添加 `is_pytorch_op` 判断
  - [ ] 实现 PyTorch 算子检查逻辑
  - [ ] 实现非 PyTorch 算子检查逻辑
  - [ ] 测试验证
- [ ] 2.1.2 修改 `_check_code()` - 装饰器添加
  - [ ] 统一处理 ops 列表
  - [ ] 为所有算子添加 @register 装饰器
  - [ ] 测试验证
- [ ] 2.2 验证 `_verify()` 方法
  - [ ] 确认多 Source 支持
  - [ ] 测试多 Source 注册
- [ ] 2.3 处理 DISPATCH_TORCH_LIB 环境变量
  - [ ] 添加非 PyTorch 算子检测
  - [ ] 修改条件判断逻辑
  - [ ] 测试不同环境变量配置

### 阶段 3：添加示例和测试
- [ ] 3.1 创建 cuBLAS GEMM 示例
  - [ ] 3.1.1 创建 baseline_gemm.py
  - [ ] 3.1.2 创建 triton_gemm.py
  - [ ] 3.1.3 创建 test_gemm.py
  - [ ] 3.1.4 创建 benchmark_gemm.py
- [ ] 3.2 创建单元测试
  - [ ] 创建 tests/test_non_pytorch_ops.py
  - [ ] 实现所有测试用例
  - [ ] 运行测试验证
- [ ] 3.3 测试检查清单
  - [ ] 运行 pytest
  - [ ] 验证命名空间创建
  - [ ] 验证函数调用
  - [ ] 验证测试比较

### 阶段 4：集成测试和文档
- [ ] 4.1 运行完整的 GEMM 示例
  - [ ] 4.1.1 准备测试环境
  - [ ] 4.1.2 运行准确性测试
  - [ ] 4.1.3 运行性能测试
  - [ ] 验证结果正确
- [ ] 4.2 更新文档
  - [ ] 4.2.1 更新 CLAUDE.md
  - [ ] 4.2.2 创建示例文档
  - [ ] 审查文档完整性

### 最终验证
- [ ] 运行所有单元测试
- [ ] 运行所有集成测试
- [ ] 验证向后兼容性（现有 PyTorch 算子测试）
- [ ] 性能测试和基准
- [ ] 代码审查
- [ ] 文档审查

---

## 预期成果

完成本 TODO 后，FlagBench 将具备以下能力：

1. **支持非 PyTorch 算子**：可以测试 cuBLAS、cuDNN 等非 PyTorch 算子
2. **双命名空间架构**：通过 `flagbench.baseline.XXX` 和 `flagbench.triton.XXX` 区分实现
3. **向后兼容**：现有 PyTorch 算子的功能完全保留
4. **完整示例**：提供 cuBLAS GEMM 的完整示例代码
5. **测试覆盖**：单元测试和集成测试覆盖所有新功能
6. **文档完善**：使用说明和最佳实践文档

---

## 参考资料

- PyTorch Library API: https://pytorch.org/docs/stable/library.html
- Triton Documentation: https://triton-lang.org/
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- FlagBench 现有文档: CLAUDE.md

