"""
TorchAdapter - PyTorch 框架适配器

从 scripts/utils.py 迁移 torch 相关逻辑
"""

import torch
from typing import Any, Dict
from .adapter import FrameworkAdapter
from .generate_args import TritonKernelGenerateArgs


class TorchAdapter(FrameworkAdapter):
    """PyTorch 框架适配器"""

    @property
    def framework_name(self) -> str:
        """返回框架名称"""
        return "torch"

    def get_operator_function(self, op_name: str) -> Any:
        """
        从 torch.ops 获取算子函数

        Args:
            op_name: 算子名称，格式如 "aten::add"

        Returns:
            torch.ops 中的函数对象
        """
        # 解析 namespace 和 kernel_name
        # 格式: "aten::add" -> namespace="aten", kernel_name="add"
        if "::" not in op_name:
            raise ValueError(f"Invalid op_name format: {op_name}. Expected format: 'namespace::kernel_name'")

        namespace, kernel_name = op_name.split("::", 1)

        # 从 torch.ops 获取 namespace
        if not hasattr(torch.ops, namespace):
            raise AttributeError(f"torch.ops does not have namespace: {namespace}")

        torch_op_namespace = getattr(torch.ops, namespace)

        # 从 namespace 获取 kernel
        if not hasattr(torch_op_namespace, kernel_name):
            raise AttributeError(f"torch.ops.{namespace} does not have kernel: {kernel_name}")

        return getattr(torch_op_namespace, kernel_name)

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """
        获取 torch API 的签名信息

        Args:
            func: torch.ops 中的函数对象
            op_name: 算子名称

        Returns:
            签名信息字典，包含 input_args, output_args, func_desc
        """
        # 获取函数描述
        func_desc = f"PyTorch operator: {op_name}"
        if hasattr(func, '__doc__') and func.__doc__:
            doc_lines = func.__doc__.strip()
            if doc_lines:
                func_desc = doc_lines

        # 获取函数签名
        input_args = self._extract_function_signature(func)

        return {
            "input_args": input_args,
            "output_args": None,
            "func_desc": func_desc,
        }

    def _extract_function_signature(self, func: Any) -> Dict[str, str]:
        """
        提取函数签名信息

        对于 torch.ops 算子，使用 _schemas 属性
        """
        if hasattr(func, '_schemas'):
            schema = func._schemas
            # schema 是一个字典，键是 overload 名称，值是签名对象
            input_output_info = {k: str(v) for k, v in schema.items()}
            return input_output_info

        # 如果没有 _schemas 属性，返回空字典
        return {}

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> TritonKernelGenerateArgs:
        """
        创建 TritonKernelGenerateArgs

        Args:
            op_name: 算子名称，格式如 "aten::add"
            func: torch.ops 中的函数对象
            impl_info: 实现信息

        Returns:
            TritonKernelGenerateArgs 实例
        """
        # 获取签名信息
        sig_info = self.get_signature_info(func, op_name)

        # 生成参考代码
        torch_kernel_code = self.get_reference_code(func, op_name)

        # 创建 TritonKernelGenerateArgs
        return TritonKernelGenerateArgs(
            triton_kernel_name=op_name,
            func_desc=sig_info["func_desc"],
            torch_kernel_code=torch_kernel_code,
            input_args=sig_info["input_args"],
            output_args=sig_info["output_args"],
            impl_info=impl_info,
            from_mcp=False,
        )

    def get_reference_code(self, func: Any, op_name: str) -> str:
        """
        生成 torch 参考代码

        Args:
            func: torch.ops 中的函数对象
            op_name: 算子名称，格式如 "aten::add"

        Returns:
            参考代码字符串
        """
        # 提取 kernel 名称
        # "aten::add" -> "add"
        kernel_name = op_name.split("::")[-1]

        # 构造函数名称
        # 例如: "torch.ops.aten.add"
        namespace = op_name.split("::")[0]
        torch_op_func_name = f"torch.ops.{namespace}.{kernel_name}"

        # 生成参考代码
        torch_kernel_code = f"""
# Reference PyTorch implementation for {op_name}
import torch

{kernel_name} = {torch_op_func_name}
""".strip()

        return torch_kernel_code

    def get_impl_info(self, kernel_name: str) -> Any:
        """
        获取 torch 算子的实现信息

        Args:
            kernel_name: 算子名称（不含namespace，如 "add"）

        Returns:
            从 IMPL_INFO 获取的实现信息
        """
        from flagbench.dataset import IMPL_INFO
        return IMPL_INFO.get(kernel_name)

