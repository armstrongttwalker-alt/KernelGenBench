"""
测试重构后的 TritonKernelGenerator

验证：
1. 向后兼容性（不传入 prompt_builder）
2. 新功能（传入 prompt_builder）
3. Deprecation warning
4. Prompt 生成委托给 PromptBuilder
"""

import pytest
import warnings
from unittest.mock import Mock, MagicMock, patch

from generator.triton_kernel_generator import TritonKernelGenerator
from generator.torch_prompt_builder import TorchPromptBuilder
from flagbench.framework.generate_args import TritonKernelGenerateArgs


class TestTritonKernelGeneratorRefactor:
    """测试重构后的 TritonKernelGenerator"""

    def test_backward_compatibility_without_prompt_builder(self):
        """测试向后兼容性 - 不传入 prompt_builder 参数"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 捕获 deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generator = TritonKernelGenerator(config)

            # 验证 deprecation warning 被触发
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "prompt_builder parameter is not provided" in str(w[0].message)
            assert "backward compatibility" in str(w[0].message)

        # 验证自动创建了 TorchPromptBuilder
        assert generator.prompt_builder is not None
        assert isinstance(generator.prompt_builder, TorchPromptBuilder)

    def test_with_explicit_prompt_builder(self):
        """测试显式传入 prompt_builder 参数"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 创建 TorchPromptBuilder
        prompt_builder = TorchPromptBuilder()

        # 不应该有 deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generator = TritonKernelGenerator(config, prompt_builder=prompt_builder)

            # 验证没有 deprecation warning
            assert len(w) == 0

        # 验证使用了传入的 prompt_builder
        assert generator.prompt_builder is prompt_builder

    def test_generate_prompt_delegates_to_builder(self):
        """测试 generate_prompt 委托给 PromptBuilder"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 创建 mock prompt_builder
        mock_builder = Mock()
        mock_builder.build = Mock(return_value="test prompt")

        # 创建 generator
        generator = TritonKernelGenerator(config, prompt_builder=mock_builder)

        # 创建测试数据
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="add = torch.ops.aten.add",
            input_args={"default": "Tensor self, Tensor other"},
            output_args=None,
            impl_info=None,
        )

        # 调用 generate_prompt
        result = generator.generate_prompt(gen_args)

        # 验证调用了 builder.build()
        mock_builder.build.assert_called_once_with(gen_args)
        assert result == "test prompt"

    def test_generate_prompt_for_new_kernel(self):
        """测试生成新 kernel 的 prompt"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 创建 TorchPromptBuilder
        prompt_builder = TorchPromptBuilder()

        # 创建 generator（不应该有 warning）
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generator = TritonKernelGenerator(config, prompt_builder=prompt_builder)
            assert len(w) == 0

        # 创建测试数据（新 kernel）
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="add = torch.ops.aten.add",
            input_args={"default": "Tensor self, Tensor other"},
            output_args=None,
            impl_info=None,
        )

        # 调用 generate_prompt
        prompt = generator.generate_prompt(gen_args)

        # 验证 prompt 包含关键内容
        assert "You are a skilled GPU programmer proficient in Triton" in prompt
        assert "add = torch.ops.aten.add" in prompt
        assert "generate a Triton kernel function" in prompt

    def test_generate_prompt_for_fix(self):
        """测试生成修复 kernel 的 prompt"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 创建 TorchPromptBuilder
        prompt_builder = TorchPromptBuilder()

        # 创建 generator
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            generator = TritonKernelGenerator(config, prompt_builder=prompt_builder)

        # 创建 mock check_result
        check_result = Mock()
        check_result.success = False
        check_result.code = "old code"
        check_result.traceback = "Error: something went wrong"
        check_result.params = "test params"
        check_result.op_name = "aten::add"

        # 创建测试数据（修复 kernel）
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="add = torch.ops.aten.add",
            input_args={"default": "Tensor self, Tensor other"},
            output_args=None,
            impl_info=None,
            check_result=check_result,
            old_code="old code",
        )

        # 调用 generate_prompt
        prompt = generator.generate_prompt(gen_args)

        # 验证 prompt 包含关键内容
        assert "fix the following Triton kernel function" in prompt
        assert "Error: something went wrong" in prompt
        assert "old code" in prompt

    def test_generate_prompt_for_optimization(self):
        """测试生成优化 kernel 的 prompt"""
        # 创建 mock config
        config = Mock()
        config.use_ai_advice = False

        # 创建 TorchPromptBuilder
        prompt_builder = TorchPromptBuilder()

        # 创建 generator
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            generator = TritonKernelGenerator(config, prompt_builder=prompt_builder)

        # 创建测试数据（优化 kernel）
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="add = torch.ops.aten.add",
            input_args={"default": "Tensor self, Tensor other"},
            output_args=None,
            impl_info=None,
            old_code="def add(x, y): return x + y",
        )

        # 调用 generate_prompt
        prompt = generator.generate_prompt(gen_args)

        # 验证 prompt 包含关键内容
        assert "optimize the following Triton kernel function" in prompt
        assert "def add(x, y): return x + y" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
