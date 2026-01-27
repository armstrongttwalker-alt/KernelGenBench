# TODO List

## 框架适配器架构重构

### 背景

当前的 `generate_kernel_and_verify.py` 脚本完全为 torch API 设计，包括：
- `create_triton_generate_args` 函数假设所有算子都是 `torch.ops`
- `TritonKernelGenerateArgs` 的结构是 torch 特定的
- prompt template 针对 torch API

随着需要支持更多框架（cupy, numpy, jax 等），需要一个更通用和可扩展的架构。

### 目标

设计并实现一个框架无关的架构，使得：
1. 添加新框架支持时不需要修改核心逻辑
2. 每个框架的特定逻辑封装在独立的模块中
3. 易于维护和测试
4. 向后兼容现有的 torch 支持

---

## 方案：适配器模式（推荐）

### 核心思想

创建框架适配器层，将不同框架的差异封装起来。每个框架实现统一的接口，核心逻辑通过适配器与具体框架交互。

### 架构设计

#### 1. 基础接口层

```python
# src/flagbench/framework/adapter.py

from abc import ABC, abstractmethod
from typing import Any, Dict

class FrameworkAdapter(ABC):
    """框架适配器基类，定义统一接口"""

    @abstractmethod
    def get_operator_function(self, op_name: str) -> Any:
        """
        获取算子函数对象

        Args:
            op_name: 算子名称，格式如 "aten::add" 或 "cupy::caxpy"

        Returns:
            函数对象
        """
        pass

    @abstractmethod
    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """
        获取函数签名信息

        Args:
            func: 函数对象
            op_name: 算子名称

        Returns:
            签名信息字典
        """
        pass

    @abstractmethod
    def create_generate_args(self, op_name: str, func: Any, impl_info: Any):
        """
        创建生成参数对象

        Args:
            op_name: 算子名称
            func: 函数对象
            impl_info: 实现信息

        Returns:
            GenerateArgs 对象
        """
        pass

    @abstractmethod
    def get_reference_code(self, func: Any, op_name: str) -> str:
        """
        生成参考代码

        Args:
            func: 函数对象
            op_name: 算子名称

        Returns:
            参考代码字符串
        """
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """框架名称"""
        pass
```

#### 2. Torch 适配器

```python
# src/flagbench/framework/torch_adapter.py

class TorchAdapter(FrameworkAdapter):
    """PyTorch 框架适配器"""

    @property
    def framework_name(self) -> str:
        return "torch"

    def get_operator_function(self, op_name: str):
        """从 torch.ops 获取算子函数"""
        namespace, kernel_name = op_name.split("::", 1)
        torch_op_namespace = getattr(torch.ops, namespace)
        return getattr(torch_op_namespace, kernel_name)

    def get_signature_info(self, func, op_name: str):
        """获取 torch API 的签名信息"""
        return get_torch_api_signature(op_name, func)

    def create_generate_args(self, op_name: str, func, impl_info):
        """创建 TritonKernelGenerateArgs"""
        # 封装现有的 create_triton_generate_args 逻辑
        return TritonKernelGenerateArgs(...)

    def get_reference_code(self, func, op_name: str) -> str:
        """生成 torch 参考代码"""
        namespace, kernel_name = op_name.split("::", 1)
        return f"""
# Reference PyTorch implementation
import torch

{kernel_name} = torch.ops.{namespace}.{kernel_name}
""".strip()
```

#### 3. Cupy 适配器

```python
# src/flagbench/framework/cupy_adapter.py

class CupyAdapter(FrameworkAdapter):
    """CuPy 框架适配器"""

    @property
    def framework_name(self) -> str:
        return "cupy"

    def get_operator_function(self, op_name: str):
        """从 CUPY_OPERATORS 获取 baseline 函数"""
        from flagbench.dataset import CUPY_OPERATORS
        return CUPY_OPERATORS[op_name]

    def get_signature_info(self, func, op_name: str):
        """获取 cupy baseline 的签名信息"""
        import inspect
        sig = inspect.signature(func)
        # 转换为统一格式
        return {
            'parameters': sig.parameters,
            'return_annotation': sig.return_annotation,
            'doc': func.__doc__
        }

    def create_generate_args(self, op_name: str, func, impl_info):
        """创建 CupyGenerateArgs"""
        import inspect
        # 确定 BLAS 操作类型
        blas_type = self._determine_blas_type(op_name)

        return CupyGenerateArgs(
            cupy_kernel_name=op_name,
            baseline_func=func,
            baseline_code=inspect.getsource(func),
            func_desc=func.__doc__ or f"cuBLAS {op_name} operation",
            blas_operation_type=blas_type,
            impl_info=impl_info
        )

    def _determine_blas_type(self, op_name: str) -> str:
        """根据函数名确定 BLAS 操作类型"""
        kernel_name = op_name.split("::")[-1]
        # Level 1: asum, axpy, dot, nrm2, scal
        if any(op in kernel_name for op in ['asum', 'axpy', 'dot', 'nrm2', 'scal']):
            return "Level 1"
        # Level 2: gemv, ger, sbmv
        elif any(op in kernel_name for op in ['gemv', 'ger', 'sbmv']):
            return "Level 2"
        # Level 3: gemm, syrk
        elif any(op in kernel_name for op in ['gemm', 'syrk']):
            return "Level 3"
        # Extensions: geam, dgmm
        else:
            return "Extension"

    def get_reference_code(self, func, op_name: str) -> str:
        """读取 cupy baseline 源代码作为参考"""
        import inspect
        return inspect.getsource(func)
```

#### 4. GenerateArgs 层次结构

**重要说明：**
- 现有代码中已有 `BaseGenerateArgs` 和 `TritonKernelGenerateArgs` 在 `src/generator/sampler/generate_samples.py`
- 重构时将在 `src/flagbench/framework/generate_args.py` 创建新的基类
- 现有的 `TritonKernelGenerateArgs` 将继承新的基类
- 通过别名保持向后兼容

```python
# src/flagbench/framework/generate_args.py (新文件)

from pydantic import BaseModel
from typing import Optional, Any, List
from abc import ABC, abstractmethod

class BaseGenerateArgs(BaseModel, ABC):
    """生成参数基类 - 只负责存储和提供算子信息"""

    # 通用字段（从现有 BaseGenerateArgs 继承）
    from_mcp: bool = False
    user_advice: Optional[str] = None
    check_result: Optional[Any] = None  # VerifyResult
    old_code: Optional[str] = None
    sample_id: int = 0
    wiki_reference: Optional[Any] = None

    @property
    @abstractmethod
    def op_name(self):
        """子类必须实现此属性"""
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """框架名称"""
        pass

    class Config:
        arbitrary_types_allowed = True

# Torch API 的生成参数（对应现有的 TritonKernelGenerateArgs）
class TritonKernelGenerateArgs(BaseGenerateArgs):
    """Torch API 的生成参数 - 用于从 torch API 生成 Triton kernel"""

    # Torch 特定字段（保持与现有 TritonKernelGenerateArgs 一致）
    triton_kernel_name: str
    func_desc: str
    torch_kernel_code: str
    input_args: Any = None  # List[InputArg] | None | dict
    output_args: Any = None  # List[OutputArg] | None
    func_type: Optional[str] = None
    impl_info: Optional[Any] = None

    @property
    def op_name(self):
        return self.triton_kernel_name

    @property
    def framework_name(self) -> str:
        return "torch"

# Cupy baseline 的生成参数
class CupyGenerateArgs(BaseGenerateArgs):
    """Cupy baseline 的生成参数 - 用于从 cupy baseline 生成 Triton kernel"""

    # Cupy 特定字段
    cupy_kernel_name: str
    baseline_func: Any
    baseline_code: str
    func_desc: str
    blas_operation_type: str  # "Level 1" / "Level 2" / "Level 3"
    impl_info: Optional[Any] = None

    @property
    def op_name(self):
        return self.cupy_kernel_name

    @property
    def framework_name(self) -> str:
        return "cupy"
```

#### 5. PromptBuilder 层（新增）

**设计思路**：
- Generator 负责调度和 LLM 调用
- PromptBuilder 负责 prompt 构造
- 使用组合而非继承，支持框架和策略两个维度的独立变化

```python
# src/generator/prompt_builder.py

class PromptBuilder(ABC):
    """Prompt 构造器基类"""

    def __init__(self, mode: str = "basic"):
        """
        Args:
            mode: prompt 模式 - "basic", "reflection", "with_wiki"
        """
        self.mode = mode

    @abstractmethod
    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        """构造生成新 kernel 的 prompt"""
        pass

    @abstractmethod
    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        """构造修复 kernel 的 prompt"""
        pass

    @abstractmethod
    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        """构造优化 kernel 的 prompt"""
        pass

    def build(self, gen_args: BaseGenerateArgs) -> str:
        """
        根据 gen_args 的状态选择合适的 prompt 构造方法
        """
        if gen_args.check_result is not None and not gen_args.check_result.success:
            if gen_args.check_result.code.strip() == gen_args.old_code.strip():
                return self.build_fix(gen_args)
            else:
                return self.build_optimization(gen_args)
        if gen_args.old_code is not None and len(gen_args.old_code.strip()) > 0:
            return self.build_optimization(gen_args)
        return self.build_new(gen_args)

class TorchPromptBuilder(PromptBuilder):
    """Torch 框架的 Prompt 构造器"""

    def build_new(self, gen_args: TritonKernelGenerateArgs) -> str:
        """构造 Torch 特定的新 kernel prompt"""
        prompt = "You are a skilled GPU programmer proficient in Triton.\n"
        prompt += "The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += f"```python\n{gen_args.torch_kernel_code}\n```\n"

        # Torch 特定：ATen operators 信息
        if gen_args.impl_info and isinstance(gen_args.impl_info, list) and len(gen_args.impl_info) > 1:
            prompt += "\nIMPORTANT: This PyTorch API uses multiple ATen operators:\n"
            for op in gen_args.impl_info:
                prompt += f"  - {op}\n"

        # 函数描述
        if gen_args.func_desc:
            prompt += f"\nFunction description: {gen_args.func_desc}\n"

        # 根据 mode 添加额外内容
        if self.mode == "with_wiki" and gen_args.wiki_reference:
            prompt += self._build_wiki_section(gen_args.wiki_reference)

        return prompt

    def build_fix(self, gen_args: TritonKernelGenerateArgs) -> str:
        """构造修复 prompt"""
        # 实现修复逻辑
        pass

    def build_optimization(self, gen_args: TritonKernelGenerateArgs) -> str:
        """构造优化 prompt"""
        # 实现优化逻辑
        pass

class CupyPromptBuilder(PromptBuilder):
    """Cupy 框架的 Prompt 构造器"""

    def build_new(self, gen_args: CupyGenerateArgs) -> str:
        """构造 Cupy 特定的新 kernel prompt"""
        prompt = "You are a skilled GPU programmer proficient in Triton.\n"
        prompt += "The Triton kernel should implement the same functionality as the following cuBLAS baseline:\n"
        prompt += f"```python\n{gen_args.baseline_code}\n```\n"

        # Cupy 特定：BLAS 操作类型
        prompt += f"\nThis is a BLAS {gen_args.blas_operation_type} operation.\n"

        # Cupy 特定：baseline 文档
        if gen_args.func_desc:
            prompt += f"\nBaseline documentation:\n{gen_args.func_desc}\n"

        # 根据 mode 添加额外内容
        if self.mode == "with_wiki" and gen_args.wiki_reference:
            prompt += self._build_wiki_section(gen_args.wiki_reference)

        return prompt

    def build_fix(self, gen_args: CupyGenerateArgs) -> str:
        """构造修复 prompt"""
        # 实现修复逻辑
        pass

    def build_optimization(self, gen_args: CupyGenerateArgs) -> str:
        """构造优化 prompt"""
        # 实现优化逻辑
        pass
```

#### 6. Generator 集成（重构）

**重构思路**：
- Generator 不再直接构造 prompt
- 通过依赖注入接收 PromptBuilder
- Generator 只负责调度和 LLM 调用

```python
# src/generator/triton_kernel_generator.py

class TritonKernelGenerator(BaseGenerator):
    def __init__(self, config: GenerationConfig, prompt_builder: PromptBuilder):
        """
        Args:
            config: 生成配置
            prompt_builder: Prompt 构造器（依赖注入）
        """
        super().__init__(config)
        self.prompt_builder = prompt_builder

    def generate_prompt(self, info: BaseGenerateArgs) -> str:
        """
        生成 prompt（委托给 PromptBuilder）
        """
        return self.prompt_builder.build(info)

    def generate(self, gen_args: BaseGenerateArgs) -> str:
        """
        生成代码
        """
        # 1. 构造 prompt
        prompt = self.generate_prompt(gen_args)

        # 2. 调用 LLM
        code = self.llm.generate(prompt)

        # 3. 提取代码
        extracted_code = extract_first_code(code, ["python"])

        return extracted_code
```

**使用方式**：
```python
# 在 generate_kernel_and_verify.py 中
from generator.prompt_builder import TorchPromptBuilder, CupyPromptBuilder

# 根据框架选择 PromptBuilder
if dataset == "torch":
    prompt_builder = TorchPromptBuilder(mode="basic")
elif dataset == "cupy":
    prompt_builder = CupyPromptBuilder(mode="basic")

# 创建 Generator
generator = TritonKernelGenerator(config, prompt_builder)
```

#### 7. 核心逻辑集成

```python
# scripts/generate_kernel_and_verify.py

class PassAtKTester:
    def __init__(self, ...):
        # 根据 dataset 选择适配器
        self.adapter = self._create_adapter()

    def _create_adapter(self) -> FrameworkAdapter:
        """根据 dataset 创建对应的适配器"""
        if self.dataset in ["pytorch", "gems", "v1", "v2", "qwen_next"]:
            from flagbench.framework.torch_adapter import TorchAdapter
            return TorchAdapter()
        elif self.dataset == "cupy":
            from flagbench.framework.cupy_adapter import CupyAdapter
            return CupyAdapter()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

    def generate_round(self, round_idx: int, remaining_operators: Dict):
        """生成测试（使用适配器）"""
        for op_name, api_info in remaining_operators.items():
            # 使用适配器获取函数
            func = self.adapter.get_operator_function(op_name)

            # 使用适配器创建生成参数
            gen_arg = self.adapter.create_generate_args(
                op_name=op_name,
                func=func,
                impl_info=impl_info_arg
            )

            # 后续逻辑保持不变
            gen_args.append(gen_arg)
```

---

## 实施计划

### 阶段1：接口定义和基础设施（2-3天）

**任务：**
1. 创建 `src/flagbench/framework/` 目录
2. 实现 `FrameworkAdapter` 基类
3. 实现 `BaseGenerateArgs` 基类（重构现有的）
4. 实现 `PromptBuilder` 基类
5. 编写单元测试框架

**产出：**
- `src/flagbench/framework/adapter.py`
- `src/flagbench/framework/generate_args.py`
- `src/generator/prompt_builder.py`
- `tests/framework/test_adapter.py`
- `tests/generator/test_prompt_builder.py`

**详细实施步骤：**

**步骤1：创建目录结构**
```bash
mkdir -p src/flagbench/framework
mkdir -p tests/framework
mkdir -p tests/generator
touch src/flagbench/framework/__init__.py
touch tests/framework/__init__.py
touch tests/generator/__init__.py
```

**步骤2：实现 BaseGenerateArgs（无依赖）**
- 文件：`src/flagbench/framework/generate_args.py`
- 内容：
  - `BaseGenerateArgs` 基类（使用 Pydantic BaseModel）
  - 定义通用字段：`from_mcp`, `user_advice`, `check_result`, `old_code`, `sample_id`, `wiki_reference`
  - 实现 `op_name` 抽象属性
  - 实现 `framework_name` 抽象属性
- 测试：`tests/framework/test_generate_args.py`
  - 测试基类的字段定义
  - 测试抽象属性

**步骤3：实现 FrameworkAdapter（依赖 BaseGenerateArgs）**
- 文件：`src/flagbench/framework/adapter.py`
- 内容：
  - `FrameworkAdapter` 抽象基类
  - 定义抽象方法：`get_operator_function`, `get_signature_info`, `create_generate_args`, `get_reference_code`
  - 定义 `framework_name` 抽象属性
- 测试：`tests/framework/test_adapter.py`
  - 测试抽象基类的接口定义
  - 创建 mock adapter 测试

**步骤4：实现 PromptBuilder（依赖 BaseGenerateArgs）**
- 文件：`src/generator/prompt_builder.py`
- 内容：
  - `PromptBuilder` 抽象基类
  - 定义抽象方法：`build_new`, `build_fix`, `build_optimization`
  - 实现 `build()` 方法（根据 gen_args 状态选择）
  - 定义 `mode` 参数（basic, reflection, with_wiki）
- 测试：`tests/generator/test_prompt_builder.py`
  - 测试抽象基类的接口定义
  - 创建 mock builder 测试
  - 测试 `build()` 方法的逻辑

**步骤5：更新 framework/__init__.py**
- 导出 `BaseGenerateArgs`, `FrameworkAdapter`

**验证标准：**
- ✅ 所有文件创建成功
- ✅ 所有测试通过
- ✅ 代码符合类型检查（mypy）
- ✅ 代码符合代码规范（flake8/black）

### 阶段2：封装现有 Torch 逻辑（3-4天）

**任务：**
1. 实现 `TorchAdapter`
2. 扩展现有的 `BaseGenerateArgs` 和 `TritonKernelGenerateArgs`（如需要）
3. 实现 `TorchPromptBuilder`（迁移现有的 prompt 构造逻辑）
4. 重构 `TritonKernelGenerator` 使用依赖注入的 PromptBuilder
5. 确保不破坏现有功能
6. 添加测试

**产出：**
- `src/flagbench/framework/torch_adapter.py`
- `src/flagbench/framework/generate_args.py` 中的新 `BaseGenerateArgs` 和 `TritonKernelGenerateArgs`
- `src/generator/sampler/generate_samples.py` 中的向后兼容别名
- `src/generator/prompt_builder.py` 中的 `TorchPromptBuilder`
- 重构后的 `src/generator/triton_kernel_generator.py`
- 所有现有测试通过
- 新增 torch adapter 和 prompt builder 测试

**详细实施步骤：**

**步骤1：迁移 TritonKernelGenerateArgs 到新基类**
- 文件：`src/flagbench/framework/generate_args.py` 和 `src/generator/sampler/generate_samples.py`
- 内容：
  - 在 `src/flagbench/framework/generate_args.py` 中创建新的 `BaseGenerateArgs` 和 `TritonKernelGenerateArgs`
  - 在 `src/generator/sampler/generate_samples.py` 中添加导入和别名：
    ```python
    from flagbench.framework.generate_args import (
        BaseGenerateArgs as _NewBaseGenerateArgs,
        TritonKernelGenerateArgs as _NewTritonKernelGenerateArgs
    )
    # 向后兼容别名
    BaseGenerateArgs = _NewBaseGenerateArgs
    TritonKernelGenerateArgs = _NewTritonKernelGenerateArgs
    ```
  - 确保所有现有字段都保留
- 迁移策略：
  - 通过别名保持向后兼容
  - 现有代码无需修改导入
- 测试：确保所有现有测试通过

**步骤2：实现 TorchPromptBuilder（迁移现有逻辑）**
- 文件：`src/generator/prompt_builder.py`
- 内容：
  - 添加 `TorchPromptBuilder` 类
  - 迁移 `TritonKernelGenerator` 中的三个方法：
    - `build_new()` ← `generate_prompt_for_new()`
    - `build_fix()` ← `generate_prompt_for_fix()`
    - `build_optimization()` ← `generate_prompt_for_optimization()`
  - 保留所有现有的 prompt 逻辑（ATen operators、overloads、wiki reference 等）
- 测试：`tests/generator/test_prompt_builder.py`
  - 测试三种 prompt 类型的生成
  - 对比新旧实现的输出一致性

**步骤3：实现 TorchAdapter**
- 文件：`src/flagbench/framework/torch_adapter.py`
- 内容：
  - 实现 `get_operator_function()` - 从 `torch.ops` 获取
  - 实现 `get_signature_info()` - 调用现有的 `get_torch_api_signature()`
  - 实现 `create_generate_args()` - 创建 `TritonKernelGenerateArgs`
  - 实现 `get_reference_code()` - 生成 torch 参考代码
  - 迁移 `scripts/utils.py` 中的 `create_triton_generate_args()` 逻辑
- 测试：`tests/framework/test_adapter.py`
  - 测试每个方法的功能
  - 使用真实的 torch operator 测试

**步骤4：重构 TritonKernelGenerator**
- 文件：`src/generator/triton_kernel_generator.py`
- 修改：
  - 构造函数接收 `prompt_builder: PromptBuilder` 参数
  - 删除三个 prompt 生成方法（已迁移到 PromptBuilder）
  - 修改 `generate_prompt()` 方法委托给 `prompt_builder.build()`
  - 保留其他逻辑不变
- 向后兼容：
  - 如果没有传入 `prompt_builder`，自动创建 `TorchPromptBuilder`（带 deprecation warning）
- 测试：确保所有现有测试通过

**步骤5：更新 generate_kernel_and_verify.py（部分）**
- 修改 `PassAtKTester.__init__()`:
  - 根据 dataset 创建对应的 PromptBuilder
  - 传递给 Generator
- 保持其他逻辑不变
- 测试：运行现有的 v2 dataset 测试

**验证标准：**
- ✅ 所有新增测试通过
- ✅ 所有现有测试通过（向后兼容）
- ✅ 运行 `--dataset v2` 测试成功
- ✅ 代码符合类型检查和规范

**验证：**
```bash
# 确保现有功能正常
python scripts/generate_kernel_and_verify.py --dataset v2 --test-type triton --max-rounds 1 --debug
```

### 阶段3：实现 Cupy 适配器（3-4天）

**任务：**
1. 实现 `CupyAdapter`
2. 实现 `CupyGenerateArgs`
3. 实现 `CupyPromptBuilder`（设计 cupy 特定的 prompt）
4. 在 `generate_kernel_and_verify.py` 中集成 Cupy 支持
5. 添加测试

**产出：**
- `src/flagbench/framework/cupy_adapter.py`
- `src/flagbench/framework/generate_args.py` 中的 `CupyGenerateArgs`
- `src/generator/prompt_builder.py` 中的 `CupyPromptBuilder`
- cupy adapter 和 prompt builder 测试

**详细实施步骤：**

**步骤1：实现 CupyGenerateArgs**
- 文件：`src/flagbench/framework/generate_args.py`
- 内容：
  - 添加 `CupyGenerateArgs` 类
  - Cupy 特定字段：`cupy_kernel_name`, `baseline_func`, `baseline_code`, `func_desc`, `blas_operation_type`, `impl_info`
  - 实现 `framework_name` 属性返回 "cupy"
  - 实现 `op_name` 属性返回 `cupy_kernel_name`
- 测试：扩展 `tests/framework/test_generate_args.py`

**步骤2：实现 CupyPromptBuilder**
- 文件：`src/generator/prompt_builder.py`
- 内容：
  - 添加 `CupyPromptBuilder` 类
  - 实现 `build_new()` - 强调 cuBLAS baseline、BLAS 操作类型
  - 实现 `build_fix()` - cupy 特定的错误修复提示
  - 实现 `build_optimization()` - cupy 特定的优化提示
  - Prompt 重点：
    - 说明这是 cuBLAS baseline 的 Triton 实现
    - 提供 BLAS Level 信息（Level 1/2/3）
    - 包含 baseline 函数的源代码和文档
- 测试：`tests/generator/test_prompt_builder.py`
  - 测试 cupy 特定的 prompt 生成
  - 验证包含 BLAS 类型信息

**步骤3：实现 CupyAdapter**
- 文件：`src/flagbench/framework/cupy_adapter.py`
- 内容：
  - 实现 `get_operator_function()` - 从 `CUPY_OPERATORS` 获取
  - 实现 `get_signature_info()` - 使用 `inspect.signature()`
  - 实现 `create_generate_args()` - 创建 `CupyGenerateArgs`
    - 需要确定 BLAS 操作类型（Level 1/2/3）
    - 可以根据函数名前缀判断（s/d/c/z + asum/axpy/gemm等）
  - 实现 `get_reference_code()` - 使用 `inspect.getsource()`
- 测试：`tests/framework/test_adapter.py`
  - 测试每个方法的功能
  - 使用真实的 cupy baseline 函数测试

**步骤4：更新 generate_kernel_and_verify.py**
- 修改 `PassAtKTester._create_adapter()`:
  - 添加 cupy case，返回 `CupyAdapter()`
- 修改 `PassAtKTester.__init__()`:
  - 根据 adapter 创建对应的 PromptBuilder
- 测试：运行 cupy dataset 测试

**验证标准：**
- ✅ 所有新增测试通过
- ✅ 运行 `--dataset cupy` 测试成功
- ✅ 生成的 prompt 包含 cupy 特定信息
- ✅ 代码符合类型检查和规范

**验证：**
```bash
# 测试 cupy 支持
python scripts/generate_kernel_and_verify.py \
    --dataset cupy \
    --test-type triton \
    --custom-test-modules src/flagbench/accuracy/cupy \
    --max-rounds 1 \
    --debug
```

### 阶段4：重构核心逻辑（1-2天）

**任务：**
1. 修改 `PassAtKTester` 使用适配器
2. 移除硬编码的框架特定逻辑
3. 更新文档

**产出：**
- 重构后的 `generate_kernel_and_verify.py`
- 更新的文档和使用示例

**详细实施步骤：**

**步骤1：完善 PassAtKTester 的 Adapter 集成**
- 修改 `_create_adapter()` 方法：
  - 支持所有 dataset 类型（pytorch, gems, v1, v2, qwen_next, cupy）
  - 返回对应的 Adapter 实例
- 修改 `__init__()` 方法：
  - 根据 adapter 创建对应的 PromptBuilder
  - 传递给 Generator
- 修改 `generate_round()` 方法：
  - 使用 `adapter.create_generate_args()` 创建 GenerateArgs
  - 移除硬编码的框架判断逻辑

**步骤2：清理冗余代码**
- 检查 `scripts/utils.py` 中的 `create_triton_generate_args()`：
  - 如果已迁移到 TorchAdapter，标记为 deprecated
  - 或者删除并更新所有引用
- 检查其他可能的硬编码逻辑

**步骤3：更新文档**
- 更新 `CLAUDE.md` 或 `README.md`：
  - 说明新的架构
  - 说明如何添加新框架支持
  - 提供使用示例

**验证标准：**
- ✅ 所有 dataset 类型都能正常工作
- ✅ 代码中没有硬编码的框架判断
- ✅ 文档清晰完整

### 阶段5：测试和优化（1-2天）

**任务：**
1. 端到端测试
2. 性能优化
3. 代码审查和清理

**产出：**
- 完整的测试覆盖
- 性能优化报告
- 清理后的代码

**详细实施步骤：**

**步骤1：端到端测试**
- 测试所有 dataset 类型：
  ```bash
  # Torch datasets
  python scripts/generate_kernel_and_verify.py --dataset v2 --test-type triton --max-rounds 1 --debug
  python scripts/generate_kernel_and_verify.py --dataset qwen_next --test-type triton --max-rounds 1 --debug

  # Cupy dataset
  python scripts/generate_kernel_and_verify.py --dataset cupy --test-type triton --custom-test-modules src/flagbench/accuracy/cupy --max-rounds 1 --debug
  ```
- 测试不同的 prompt 模式（basic, reflection, with_wiki）
- 测试错误处理和边界情况

**步骤2：性能测试**
- 对比重构前后的性能：
  - Prompt 生成时间
  - 内存使用
  - 整体运行时间
- 如果有性能下降，进行优化

**步骤3：代码审查**
- 检查代码质量：
  - 类型注解完整性
  - 文档字符串完整性
  - 代码规范（flake8/black）
- 检查测试覆盖率
- 清理临时代码和注释

**步骤4：向后兼容性验证**
- 确保现有的脚本和工具仍然可用
- 确保 API 变更有清晰的迁移路径
- 提供 deprecation warnings

**验证标准：**
- ✅ 所有端到端测试通过
- ✅ 性能没有明显下降（<5%）
- ✅ 测试覆盖率 >80%
- ✅ 代码质量检查通过
- ✅ 向后兼容性保持

---

## 需要讨论的问题

### 1. GenerateArgs 的统一性 ✅ 已决策

**问题：** 不同框架的 GenerateArgs 可能有完全不同的字段

**决策：采用方案A（基类 + 子类）**

**设计原则：**
1. **职责单一**：GenerateArgs 专注于存储算子信息，PromptBuilder 负责 prompt 构造
2. **直接访问**：PromptBuilder 直接访问 GenerateArgs 的字段，简单清晰
3. **类型安全**：使用基类定义通用字段，子类添加框架特定字段
4. **结构化信息**：重载、schema 等信息应该是独立的结构化字段，而不是放在描述文本中

**优势：**
- 类型安全，IDE 支持好
- 清晰的扩展路径
- 易于测试和维护
- 简单直接，无需额外的数据转换层

**关键设计：**
- `op_name` 属性：返回算子名称（抽象属性）
- `framework_name` 属性：标识框架类型（抽象属性）
- Prompt 构造由独立的 PromptBuilder 负责（见下文）

### 2. Prompt Template 管理 ✅ 已决策

**问题：** 不同框架可能需要不同的 prompt

**现状分析：**
- Prompt 管理在 `src/generator/` 目录中（不在 flagbench 中）
- 当前通过字符串拼接构造 prompt，没有使用模板文件
- `TritonKernelGenerator` 有三个 prompt 生成方法：
  - `generate_prompt_for_new()` - 生成新 kernel
  - `generate_prompt_for_optimization()` - 优化现有 kernel
  - `generate_prompt_for_fix()` - 修复错误 kernel

**决策：采用 PromptBuilder 模式（方案C）**

**设计原则：**
1. **职责分离**：Generator 负责调度和 LLM 调用，PromptBuilder 负责 prompt 构造
2. **组合优于继承**：使用组合而非继承，提高灵活性
3. **两个维度独立变化**：
   - 框架维度：torch, cupy, numpy, jax
   - 策略维度：basic, reflection, with_wiki

**优势：**
- 类数量 = 框架数 + 1（而非框架数 × 策略数）
- 可以在运行时切换 PromptBuilder
- 易于单独测试 prompt 构造逻辑
- 符合单一职责原则

### 3. 向后兼容性

**问题：** 如何确保不破坏现有功能？

**策略：**
1. 保留旧的函数作为 deprecated
2. 添加兼容层
3. 逐步迁移
4. 充分的测试覆盖

### 4. 其他框架支持

**未来可能支持的框架：**
- NumPy
- JAX
- TensorFlow
- Custom CUDA kernels

**需要考虑：**
- 这些框架的特殊需求
- 接口是否足够通用
- 是否需要更多的抽象层

---

## 风险和挑战

### 技术风险

1. **重构范围大**
   - 影响核心逻辑
   - 需要充分测试
   - 可能引入新 bug

2. **Prompt 适配复杂**
   - 不同框架的 API 风格差异大
   - LLM 可能需要不同的提示策略
   - 需要实验验证效果

3. **性能影响**
   - 适配器层可能增加开销
   - 需要性能测试和优化

### 实施风险

1. **时间投入**
   - 预计需要 1-2 周完整实施
   - 可能影响其他开发计划

2. **学习曲线**
   - 新的架构需要团队理解
   - 需要文档和培训

---

## 下一步行动

**已解决的问题：**
1. ✅ GenerateArgs 的设计方案：采用方案A（基类 + 子类）
2. ✅ GenerateArgs 职责：专注于数据存储，PromptBuilder 直接访问字段
3. ✅ Prompt 构造职责：由独立的 PromptBuilder 负责
4. ✅ Prompt 管理架构：采用方案C（PromptBuilder 模式，组合优于继承）

**待讨论的问题：**
1. ⚠️ 向后兼容性策略的具体实施方案
2. ⚠️ 实施的优先级和时间安排

**可以开始的工作：**
- 阶段1：接口定义和基础设施（BaseGenerateArgs、PromptBuilder 基类、FrameworkAdapter）
- 阶段2：封装现有 Torch 逻辑（TorchAdapter、TritonKernelGenerateArgs、TorchPromptBuilder）
- 阶段3：实现 Cupy 支持（CupyAdapter、CupyGenerateArgs、CupyPromptBuilder）

**架构优势总结：**
- 职责清晰：Adapter 负责创建 GenerateArgs，GenerateArgs 负责数据提供，PromptBuilder 负责 prompt 构造，Generator 负责调度和 LLM 调用
- 易于扩展：添加新框架只需实现对应的 Adapter、GenerateArgs 和 PromptBuilder
- 两个维度独立变化：框架类型和 prompt 策略可以独立扩展
- 类数量可控：框架数 + 策略数，而非框架数 × 策略数
