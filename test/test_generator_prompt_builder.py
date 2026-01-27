"""
测试 PromptBuilder 基类
"""

import pytest
from generator.prompt_builder import PromptBuilder
from flagbench.framework.generate_args import BaseGenerateArgs


class MockGenerateArgs(BaseGenerateArgs):
    """用于测试的 Mock GenerateArgs"""

    test_op_name: str = "test_op"

    @property
    def op_name(self):
        return self.test_op_name

    @property
    def framework_name(self) -> str:
        return "mock"


class MockVerifyResult:
    """用于测试的 Mock VerifyResult"""

    def __init__(self, success: bool, code: str = ""):
        self.success = success
        self.code = code


class MockPromptBuilder(PromptBuilder):
    """用于测试的 Mock PromptBuilder"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        return f"NEW_PROMPT for {gen_args.op_name}"

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        return f"FIX_PROMPT for {gen_args.op_name}"

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        return f"OPTIMIZATION_PROMPT for {gen_args.op_name}"


class TestPromptBuilder:
    """测试 PromptBuilder 基类"""

    def test_mode_initialization(self):
        """测试 mode 参数初始化"""
        builder_basic = MockPromptBuilder(mode="basic")
        assert builder_basic.mode == "basic"

        builder_reflection = MockPromptBuilder(mode="reflection")
        assert builder_reflection.mode == "reflection"

        builder_wiki = MockPromptBuilder(mode="with_wiki")
        assert builder_wiki.mode == "with_wiki"

    def test_default_mode(self):
        """测试默认 mode"""
        builder = MockPromptBuilder()
        assert builder.mode == "basic"

    def test_build_new_for_fresh_generation(self):
        """测试新生成场景：没有 old_code 和 check_result"""
        builder = MockPromptBuilder()
        gen_args = MockGenerateArgs(test_op_name="test_op")

        prompt = builder.build(gen_args)
        assert prompt == "NEW_PROMPT for test_op"

    def test_build_fix_for_failed_verification(self):
        """测试修复场景：check_result 失败且代码未变"""
        builder = MockPromptBuilder()
        old_code = "old code content"
        check_result = MockVerifyResult(success=False, code=old_code)

        gen_args = MockGenerateArgs(
            test_op_name="test_op",
            old_code=old_code,
            check_result=check_result
        )

        prompt = builder.build(gen_args)
        assert prompt == "FIX_PROMPT for test_op"

    def test_build_optimization_for_changed_code(self):
        """测试优化场景：check_result 失败但代码已变"""
        builder = MockPromptBuilder()
        old_code = "old code content"
        new_code = "new code content"
        check_result = MockVerifyResult(success=False, code=new_code)

        gen_args = MockGenerateArgs(
            test_op_name="test_op",
            old_code=old_code,
            check_result=check_result
        )

        prompt = builder.build(gen_args)
        assert prompt == "OPTIMIZATION_PROMPT for test_op"

    def test_build_optimization_for_existing_code(self):
        """测试优化场景：有 old_code 但没有 check_result"""
        builder = MockPromptBuilder()
        gen_args = MockGenerateArgs(
            test_op_name="test_op",
            old_code="existing code"
        )

        prompt = builder.build(gen_args)
        assert prompt == "OPTIMIZATION_PROMPT for test_op"

    def test_build_new_for_successful_verification(self):
        """测试成功验证场景：check_result 成功"""
        builder = MockPromptBuilder()
        check_result = MockVerifyResult(success=True, code="code")

        gen_args = MockGenerateArgs(
            test_op_name="test_op",
            check_result=check_result
        )

        prompt = builder.build(gen_args)
        assert prompt == "NEW_PROMPT for test_op"

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            PromptBuilder()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
