from .test_func_generator import TestFuncGenerator, TestFuncGenerateArgs
from .benchmark_func_generator import BenchmarkFuncGenerator, BenchmarkFuncGenerateArgs
from .triton_kernel_generator import TritonKernelGenerator, TritonKernelAdviceGenerator, TritonKernelGenerateArgs
from .torch_kernel_generator import TorchKernelGenerator, TorchKernelAdviceGenerator, TorchKernelGenerateArgs
from .generator import print_prompt
from .generator import BaseGenerator


GENERATOR = {
    "triton": TritonKernelGenerator,
    "torch": TorchKernelGenerator,
    "accuracy": TestFuncGenerator,
    "performance": BenchmarkFuncGenerator
}

GENERATOR_ARGS = {
    "triton": TritonKernelGenerateArgs,
    "torch": TorchKernelGenerateArgs,
    "accuracy": TestFuncGenerateArgs,
    "performance": BenchmarkFuncGenerateArgs
}