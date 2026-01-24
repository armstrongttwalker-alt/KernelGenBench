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
        return CupyGenerateArgs(
            op_name=op_name,
            baseline_func=func,
            signature=self.get_signature_info(func, op_name),
            reference_code=self.get_reference_code(func, op_name),
            impl_info=impl_info
        )

    def get_reference_code(self, func, op_name: str) -> str:
        """读取 cupy baseline 源代码作为参考"""
        import inspect
        return inspect.getsource(func)
```

#### 4. GenerateArgs 层次结构

```python
# src/flagbench/framework/generate_args.py

@dataclass
class BaseGenerateArgs(ABC):
    """生成参数基类 - 只负责存储和提供算子信息"""

    # 通用字段
    op_name: str
    signature_info: Dict
    reference_code: str
    impl_info: Any

    # 可选字段
    sample_id: int = 0

    def get_prompt_data(self) -> Dict:
        """
        返回用于构造 prompt 的结构化数据
        这个方法只是数据的 getter，不负责构造 prompt
        """
        return {
            "op_name": self.op_name,
            "signature": self.signature_info,
            "reference_code": self.reference_code,
            "sample_id": self.sample_id,
        }

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """框架名称"""
        pass

@dataclass
class TorchGenerateArgs(BaseGenerateArgs):
    """Torch API 的生成参数"""
    torch_op_func: Any
    torch_op_func_name: str
    overloads: List[str]        # 重载列表
    schemas: Dict[str, str]     # 每个重载的 schema

    @property
    def framework_name(self) -> str:
        return "torch"

    def get_prompt_data(self) -> Dict:
        """返回 Torch 特定的数据"""
        base_data = super().get_prompt_data()
        base_data.update({
            "framework": "torch",
            "func_name": self.torch_op_func_name,
            "overloads": self.overloads,
            "schemas": self.schemas,
        })
        return base_data

@dataclass
class CupyGenerateArgs(BaseGenerateArgs):
    """Cupy baseline 的生成参数"""
    baseline_func: Any
    blas_operation_type: str  # "Level 1" / "Level 2" / "Level 3"

    @property
    def framework_name(self) -> str:
        return "cupy"

    def get_prompt_data(self) -> Dict:
        """返回 Cupy 特定的数据"""
        base_data = super().get_prompt_data()
        base_data.update({
            "framework": "cupy",
            "blas_type": self.blas_operation_type,
            "baseline_doc": self.baseline_func.__doc__ if self.baseline_func else "",
        })
        return base_data
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

    def build_new(self, gen_args: TorchGenerateArgs) -> str:
        """构造 Torch 特定的新 kernel prompt"""
        prompt_data = gen_args.get_prompt_data()

        prompt = "You are a skilled GPU programmer proficient in Triton.\n"
        prompt += "The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += f"```python\n{prompt_data['reference_code']}\n```\n"

        # Torch 特定：ATen operators 信息
        if 'impl_info' in prompt_data and len(prompt_data['impl_info']) > 1:
            prompt += "\nIMPORTANT: This PyTorch API uses multiple ATen operators:\n"
            for op in prompt_data['impl_info']:
                prompt += f"  - {op}\n"

        # 添加 overloads 信息
        if 'overloads' in prompt_data:
            prompt += "\nSupported overloads:\n"
            for overload in prompt_data['overloads']:
                prompt += f"  - {overload}\n"

        # 根据 mode 添加额外内容
        if self.mode == "with_wiki" and gen_args.wiki_reference:
            prompt += self._build_wiki_section(gen_args.wiki_reference)

        return prompt

    def build_fix(self, gen_args: TorchGenerateArgs) -> str:
        """构造修复 prompt"""
        # 实现修复逻辑
        pass

    def build_optimization(self, gen_args: TorchGenerateArgs) -> str:
        """构造优化 prompt"""
        # 实现优化逻辑
        pass

class CupyPromptBuilder(PromptBuilder):
    """Cupy 框架的 Prompt 构造器"""

    def build_new(self, gen_args: CupyGenerateArgs) -> str:
        """构造 Cupy 特定的新 kernel prompt"""
        prompt_data = gen_args.get_prompt_data()

        prompt = "You are a skilled GPU programmer proficient in Triton.\n"
        prompt += "The Triton kernel should implement the same functionality as the following cuBLAS baseline:\n"
        prompt += f"```python\n{prompt_data['reference_code']}\n```\n"

        # Cupy 特定：BLAS 操作类型
        if 'blas_type' in prompt_data:
            prompt += f"\nThis is a BLAS {prompt_data['blas_type']} operation.\n"

        # Cupy 特定：baseline 文档
        if 'baseline_doc' in prompt_data:
            prompt += f"\nBaseline documentation:\n{prompt_data['baseline_doc']}\n"

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

### 阶段2：封装现有 Torch 逻辑（3-4天）

**任务：**
1. 实现 `TorchAdapter`
2. 重构现有的 `TritonKernelGenerateArgs` 为 `TorchGenerateArgs`
3. 实现 `TorchPromptBuilder`（迁移现有的 prompt 构造逻辑）
4. 重构 `TritonKernelGenerator` 使用依赖注入的 PromptBuilder
5. 确保不破坏现有功能
6. 添加测试

**产出：**
- `src/flagbench/framework/torch_adapter.py`
- `src/flagbench/framework/generate_args.py` 中的 `TorchGenerateArgs`
- `src/generator/prompt_builder.py` 中的 `TorchPromptBuilder`
- 重构后的 `src/generator/triton_kernel_generator.py`
- 所有现有测试通过
- 新增 torch adapter 和 prompt builder 测试

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

### 阶段5：测试和优化（1-2天）

**任务：**
1. 端到端测试
2. 性能优化
3. 代码审查和清理

---

## 需要讨论的问题

### 1. GenerateArgs 的统一性 ✅ 已决策

**问题：** 不同框架的 GenerateArgs 可能有完全不同的字段

**决策：采用方案A（基类 + 子类）**

**设计原则：**
1. **职责单一**：GenerateArgs 专注于存储和提供算子信息，不负责 prompt 构造
2. **统一接口**：通过 `get_prompt_data()` 方法返回结构化数据
3. **类型安全**：使用基类定义通用字段，子类添加框架特定字段
4. **结构化信息**：重载、schema 等信息应该是独立的结构化字段，而不是放在描述文本中

**优势：**
- 类型安全，IDE 支持好
- 清晰的扩展路径
- 易于测试和维护
- 强制实现统一接口

**关键设计：**
- `get_prompt_data()` 方法：返回用于构造 prompt 的结构化数据
- `framework_name` 属性：标识框架类型
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
2. ✅ GenerateArgs 职责：专注于数据提供，通过 `get_prompt_data()` 方法
3. ✅ Prompt 构造职责：由独立的 PromptBuilder 负责
4. ✅ Prompt 管理架构：采用方案C（PromptBuilder 模式，组合优于继承）

**待讨论的问题：**
1. ⚠️ 向后兼容性策略的具体实施方案
2. ⚠️ 实施的优先级和时间安排

**可以开始的工作：**
- 阶段1：接口定义和基础设施（BaseGenerateArgs、PromptBuilder 基类、FrameworkAdapter）
- 阶段2：封装现有 Torch 逻辑（TorchAdapter、TorchGenerateArgs、TorchPromptBuilder）
- 阶段3：实现 Cupy 支持（CupyAdapter、CupyGenerateArgs、CupyPromptBuilder）

**架构优势总结：**
- 职责清晰：Adapter 负责创建 GenerateArgs，GenerateArgs 负责数据提供，PromptBuilder 负责 prompt 构造，Generator 负责调度和 LLM 调用
- 易于扩展：添加新框架只需实现对应的 Adapter、GenerateArgs 和 PromptBuilder
- 两个维度独立变化：框架类型和 prompt 策略可以独立扩展
- 类数量可控：框架数 + 策略数，而非框架数 × 策略数
