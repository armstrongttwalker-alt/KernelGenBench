"""
Framework 模块

提供框架适配器和生成参数的基础接口。
"""

from .adapter import FrameworkAdapter
from .generate_args import BaseGenerateArgs, TritonKernelGenerateArgs, InputArg, OutputArg
from .torch_adapter import TorchAdapter

__all__ = [
    'FrameworkAdapter',
    'BaseGenerateArgs',
    'TritonKernelGenerateArgs',
    'InputArg',
    'OutputArg',
    'TorchAdapter',
]
