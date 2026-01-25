"""
FrameworkAdapter 基类

定义了框架适配器的统一接口，用于封装不同框架的特定逻辑。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from .generate_args import BaseGenerateArgs


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
    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> BaseGenerateArgs:
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
