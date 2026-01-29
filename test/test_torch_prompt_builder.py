"""
测试 TorchPromptBuilder
"""

import pytest
from generator.torch_prompt_builder import TorchPromptBuilder
from flagbench.framework.generate_args import TritonKernelGenerateArgs


class MockVerifyResult:
    """用于测试的 Mock VerifyResult"""

    def __init__(self, success: bool, code: str = "", traceback: str = "", params: str = "", op_name: str = ""):
        self.success = success
        self.code = code
        self.traceback = traceback
        self.params = params
        self.op_name = op_name


class TestTorchPromptBuilder:
    """测试 TorchPromptBuilder"""

    def test_initialization(self):
        """测试初始化"""
        builder = TorchPromptBuilder()
        assert builder.mode == "basic"

        builder_with_mode = TorchPromptBuilder(mode="reflection")
        assert builder_with_mode.mode == "reflection"

    def test_build_new_basic(self):
        """测试 build_new 方法 - 基本场景"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"": "x: Tensor, y: Tensor"}
        )

        prompt = builder.build_new(gen_args)

        # 验证 prompt 包含关键内容
        assert "You are a skilled GPU programmer proficient in Triton" in prompt
        assert "def add(x, y):" in prompt
        assert "return x + y" in prompt
        assert "test_add" in prompt
        assert "```python" in prompt

    def test_build_new_with_impl_info(self):
        """测试 build_new 方法 - 包含 impl_info"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"": "x: Tensor, y: Tensor"},
            impl_info=[("aten.add.Tensor", None), ("aten.add.Scalar", None)]
        )

        prompt = builder.build_new(gen_args)

        # 验证包含多算子相关内容
        assert "multiple ATen C++ operators" in prompt
        assert "aten.add.Tensor" in prompt
        assert "aten.add.Scalar" in prompt
        assert "aten_add_Tensor" in prompt
        assert "aten_add_Scalar" in prompt

    def test_build_new_with_wiki_reference(self):
        """测试 build_new 方法 - 包含 wiki_reference"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"": "x: Tensor, y: Tensor"},
            wiki_reference=[
                {"link": "https://example.com/ref1", "code": "def ref1(): pass"},
                {"link": "https://example.com/ref2", "code": "def ref2(): pass"}
            ]
        )

        prompt = builder.build_new(gen_args)

        # 验证包含 wiki reference
        assert "Reference Implementations" in prompt
        assert "https://example.com/ref1" in prompt
        assert "def ref1(): pass" in prompt

    def test_build_new_with_user_advice(self):
        """测试 build_new 方法 - 包含 user_advice"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"": "x: Tensor, y: Tensor"},
            user_advice="Use vectorized operations for better performance"
        )

        prompt = builder.build_new(gen_args)

        # 验证包含 user advice
        assert "Use vectorized operations for better performance" in prompt

    def test_build_fix_basic(self):
        """测试 build_fix 方法 - 基本场景"""
        builder = TorchPromptBuilder()
        check_result = MockVerifyResult(
            success=False,
            code="def add(x, y):\n    return x + y  # buggy code",
            traceback="RuntimeError: shape mismatch",
            params="x: shape=(2, 3), y: shape=(3,)",
            op_name="test_add"
        )
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor", "y": "Tensor"},
            output_args={"output": "Tensor"},
            check_result=check_result
        )

        prompt = builder.build_fix(gen_args)

        # 验证 prompt 包含关键内容
        assert "fix the following Triton kernel" in prompt
        assert "buggy code" in prompt
        assert "RuntimeError: shape mismatch" in prompt
        assert "x: shape=(2, 3), y: shape=(3,)" in prompt
        assert "test_add" in prompt

    def test_build_fix_with_impl_info(self):
        """测试 build_fix 方法 - 包含 impl_info"""
        builder = TorchPromptBuilder()
        check_result = MockVerifyResult(
            success=False,
            code="def add(x, y):\n    return x + y",
            traceback="RuntimeError: error",
            op_name="aten::add"
        )
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="aten::add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor", "y": "Tensor"},
            output_args={"output": "Tensor"},
            impl_info=[("aten.add.Tensor", None), ("aten.add.Scalar", None)],
            check_result=check_result
        )

        prompt = builder.build_fix(gen_args)

        # 验证包含多算子相关内容
        assert "multiple ATen C++ operators" in prompt
        assert "aten.add.Tensor" in prompt
        assert "aten.add.Scalar" in prompt

    def test_build_optimization_basic(self):
        """测试 build_optimization 方法 - 基本场景"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor", "y": "Tensor"},
            output_args={"output": "Tensor"},
            old_code="def add(x, y):\n    # old implementation\n    return x + y"
        )

        prompt = builder.build_optimization(gen_args)

        # 验证 prompt 包含关键内容
        assert "optimize the following Triton kernel" in prompt
        assert "old implementation" in prompt
        assert "Add two tensors" in prompt
        assert "test_add" in prompt

    def test_build_optimization_with_wiki_reference(self):
        """测试 build_optimization 方法 - 包含 wiki_reference"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor", "y": "Tensor"},
            output_args={"output": "Tensor"},
            old_code="def add(x, y): return x + y",
            wiki_reference=[{"link": "https://example.com/ref", "code": "def ref(): pass"}]
        )

        prompt = builder.build_optimization(gen_args)

        # 验证包含 wiki reference
        assert "Reference Implementations" in prompt
        assert "https://example.com/ref" in prompt

    def test_build_calls_build_new_for_fresh_generation(self):
        """测试 build 方法 - 新生成场景调用 build_new"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"": "x: Tensor, y: Tensor"}
        )

        prompt = builder.build(gen_args)

        # 验证调用了 build_new
        assert "You are a skilled GPU programmer proficient in Triton" in prompt
        assert "generate a Triton kernel function" in prompt

    def test_build_calls_build_fix_for_failed_verification(self):
        """测试 build 方法 - 失败验证场景调用 build_fix"""
        builder = TorchPromptBuilder()
        old_code = "def add(x, y):\n    return x + y"
        check_result = MockVerifyResult(
            success=False,
            code=old_code,
            traceback="RuntimeError: error",
            op_name="test_add"
        )
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor"},
            output_args={"output": "Tensor"},
            old_code=old_code,
            check_result=check_result
        )

        prompt = builder.build(gen_args)

        # 验证调用了 build_fix
        assert "fix the following Triton kernel" in prompt
        assert "RuntimeError: error" in prompt

    def test_build_calls_build_optimization_for_changed_code(self):
        """测试 build 方法 - 代码已变场景调用 build_optimization"""
        builder = TorchPromptBuilder()
        old_code = "def add(x, y):\n    return x + y"
        new_code = "def add(x, y):\n    return x + y  # modified"
        check_result = MockVerifyResult(
            success=False,
            code=new_code,
            traceback="RuntimeError: error",
            op_name="test_add"
        )
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor"},
            output_args={"output": "Tensor"},
            old_code=old_code,
            check_result=check_result
        )

        prompt = builder.build(gen_args)

        # 验证调用了 build_optimization
        assert "optimize the following Triton kernel" in prompt

    def test_build_calls_build_optimization_for_existing_code(self):
        """测试 build 方法 - 有旧代码场景调用 build_optimization"""
        builder = TorchPromptBuilder()
        gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test_add",
            func_desc="Add two tensors",
            torch_kernel_code="def add(x, y):\n    return x + y",
            input_args={"x": "Tensor"},
            output_args={"output": "Tensor"},
            old_code="def add(x, y):\n    return x + y"
        )

        prompt = builder.build(gen_args)

        # 验证调用了 build_optimization
        assert "optimize the following Triton kernel" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
