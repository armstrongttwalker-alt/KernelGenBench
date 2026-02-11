"""
PromptBuilder 层

负责构造用于 LLM 生成 Triton kernel 的 prompt。
"""

from abc import ABC, abstractmethod
from flagbench.framework.generate_args import BaseGenerateArgs
from runtime import get_device_constraints


class PromptBuilder(ABC):
    """Prompt 构造器基类"""

    def __init__(self, mode: str = "basic"):
        """
        Args:
            mode: prompt 模式 - "basic", "reflection", "with_wiki"
        """
        self.mode = mode

    def _get_device_constraints(self) -> str:
        """获取当前设备的 Prompt 约束"""
        return get_device_constraints()

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
            if gen_args.old_code and gen_args.check_result.code and \
               gen_args.check_result.code.strip() == gen_args.old_code.strip():
                return self.build_fix(gen_args)
            else:
                return self.build_optimization(gen_args)
        if gen_args.old_code is not None and len(gen_args.old_code.strip()) > 0:
            return self.build_optimization(gen_args)
        return self.build_new(gen_args)
