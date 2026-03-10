"""
KernelGenBench prompt builder

根据 generate_args 类型路由到对应的 prompt builder。
"""

from typing import Any


class KernelGenBenchPromptBuilder:
    """KernelGenBench prompt builder，根据 args 类型分发"""

    def __init__(self, mode: str = "basic"):
        from .vllm_prompt_builder import VllmPromptBuilder
        from .cublas_prompt_builder import CublasPromptBuilder
        from .torch_prompt_builder import TorchPromptBuilder
        self.vllm_builder = VllmPromptBuilder(mode=mode)
        self.cublas_builder = CublasPromptBuilder(mode=mode)
        self.torch_builder = TorchPromptBuilder(mode=mode)

    def _get_builder(self, gen_args: Any):
        from flagbench.framework.generate_args import VllmGenerateArgs, CublasGenerateArgs, TritonKernelGenerateArgs
        if isinstance(gen_args, VllmGenerateArgs):
            return self.vllm_builder
        elif isinstance(gen_args, CublasGenerateArgs):
            return self.cublas_builder
        elif isinstance(gen_args, TritonKernelGenerateArgs):
            return self.torch_builder
        else:
            raise TypeError(f"Unknown args type: {type(gen_args)}")

    def build(self, gen_args: Any):
        return self._get_builder(gen_args).build(gen_args)

    def build_new(self, gen_args: Any):
        return self._get_builder(gen_args).build_new(gen_args)

    def build_fix(self, gen_args: Any):
        return self._get_builder(gen_args).build_fix(gen_args)

    def build_optimization(self, gen_args: Any):
        return self._get_builder(gen_args).build_optimization(gen_args)
