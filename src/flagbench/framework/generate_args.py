"""
GenerateArgs 层次结构

定义了用于 Triton kernel 生成的参数类。
"""

from pydantic import BaseModel
from typing import Optional, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InputArg:
    """输入参数定义"""
    arg_name: str
    arg_type: str
    arg_value: Any = None
    arg_default: Any = None
    arg_desc: str = ""


@dataclass
class OutputArg:
    """输出参数定义"""
    arg_type: str
    arg_value: Any = None
    arg_desc: str = ""


class BaseGenerateArgs(BaseModel, ABC):
    """生成参数基类 - 只负责存储算子信息"""

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
        """子类必须实现此属性，返回算子名称"""
        pass

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """子类必须实现此属性，返回框架名称"""
        pass

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型（如函数对象）


class TritonKernelGenerateArgs(BaseGenerateArgs):
    """Torch API 的生成参数 - 用于从 torch API 生成 Triton kernel"""

    # Torch 特定字段（保持与现有 TritonKernelGenerateArgs 一致）
    triton_kernel_name: str
    func_desc: str
    torch_kernel_code: str
    input_args: Optional[Any] = None  # List[InputArg] | None | dict
    output_args: Optional[Any] = None  # List[OutputArg] | None
    func_type: Optional[str] = None  # "unary", "binary", "reduction", "other"
    impl_info: Optional[Any] = None  # dict | list

    @property
    def op_name(self):
        return self.triton_kernel_name

    @property
    def framework_name(self) -> str:
        return "torch"


class CupyGenerateArgs(BaseGenerateArgs):
    """Cupy baseline 的生成参数 - 用于从 cupy baseline 生成 Triton kernel"""

    # Cupy 特定字段
    cupy_kernel_name: str  # 算子名称，如 "cublas::sgemm"
    baseline_func: Any  # baseline 函数对象
    baseline_code: str  # baseline 函数源代码（通过 inspect.getsource() 获取）
    func_desc: str  # 函数描述
    blas_operation_type: str  # BLAS 操作类型："Level 1" / "Level 2" / "Level 3" / "Extension"
    impl_info: Optional[Any] = None  # 实现信息

    @property
    def op_name(self):
        return self.cupy_kernel_name

    @property
    def framework_name(self) -> str:
        return "cupy"
