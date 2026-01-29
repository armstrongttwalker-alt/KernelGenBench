"""
测试 CupyPromptBuilder
"""

import pytest
from unittest.mock import Mock

from generator.cupy_prompt_builder import CupyPromptBuilder
from flagbench.framework.generate_args import CupyGenerateArgs


class TestCupyPromptBuilder:
    """测试 CupyPromptBuilder"""

    def test_basic_instantiation(self):
        """测试基本实例化"""
        builder = CupyPromptBuilder(mode="basic")
        assert builder is not None
        assert isinstance(builder, CupyPromptBuilder)
        assert builder.mode == "basic"

    def test_with_wiki_mode(self):
        """测试 with_wiki 模式"""
        builder = CupyPromptBuilder(mode="with_wiki")
        assert builder.mode == "with_wiki"

    def test_build_new_basic(self):
        """测试 build_new() - 基本模式"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_baseline,
            baseline_code="def saxpy(alpha, x, y):\n    return alpha * x + y",
            func_desc="Single precision a*x + y",
            blas_operation_type="Level 1"
        )

        prompt = builder.build_new(gen_args)

        # 验证 prompt 包含关键内容
        assert "skilled GPU programmer proficient in Triton" in prompt
        assert "cuBLAS baseline" in prompt
        assert "Level 1" in prompt
        assert "cublas::saxpy" in prompt
        assert "def saxpy(alpha, x, y)" in prompt
        assert "Single precision a*x + y" in prompt

    def test_build_new_level_2_blas(self):
        """测试 build_new() - Level 2 BLAS"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemv",
            baseline_func=mock_baseline,
            baseline_code="def sgemv(A, x):\n    return A @ x",
            func_desc="Matrix-vector multiplication",
            blas_operation_type="Level 2"
        )

        prompt = builder.build_new(gen_args)

        assert "Level 2" in prompt
        assert "matrix-vector" in prompt
        assert "sgemv" in prompt

    def test_build_new_level_3_blas(self):
        """测试 build_new() - Level 3 BLAS"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_baseline,
            baseline_code="def sgemm(A, B):\n    return A @ B",
            func_desc="Matrix-matrix multiplication",
            blas_operation_type="Level 3"
        )

        prompt = builder.build_new(gen_args)

        assert "Level 3" in prompt
        assert "matrix-matrix" in prompt
        assert "sgemm" in prompt
        assert "tiling" in prompt or "blocking" in prompt

    def test_build_new_extension(self):
        """测试 build_new() - Extension operations"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgeam",
            baseline_func=mock_baseline,
            baseline_code="def sgeam(A, B):\n    return A + B",
            func_desc="Matrix addition",
            blas_operation_type="Extension"
        )

        prompt = builder.build_new(gen_args)

        assert "Extension" in prompt
        assert "sgeam" in prompt

    def test_build_new_with_wiki(self):
        """测试 build_new() - with_wiki 模式"""
        builder = CupyPromptBuilder(mode="with_wiki")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_baseline,
            baseline_code="def saxpy(alpha, x, y):\n    return alpha * x + y",
            func_desc="Single precision a*x + y",
            blas_operation_type="Level 1",
            wiki_reference="SAXPY is a Level 1 BLAS operation..."
        )

        prompt = builder.build_new(gen_args)

        assert "Reference Information" in prompt
        assert "SAXPY is a Level 1 BLAS operation" in prompt

    def test_build_fix(self):
        """测试 build_fix()"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        # 创建 mock check_result
        check_result = Mock()
        check_result.success = False
        check_result.code = "def saxpy_old(): pass"
        check_result.traceback = "Error: dimension mismatch"
        check_result.params = "alpha=2.0, x=tensor([1,2,3]), y=tensor([4,5,6])"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_baseline,
            baseline_code="def saxpy(alpha, x, y):\n    return alpha * x + y",
            func_desc="Single precision a*x + y",
            blas_operation_type="Level 1",
            check_result=check_result,
            old_code="def saxpy_old(): pass"
        )

        prompt = builder.build_fix(gen_args)

        assert "fix the following Triton kernel" in prompt
        assert "Error: dimension mismatch" in prompt
        assert "def saxpy_old(): pass" in prompt
        assert "cuBLAS Baseline Function" in prompt

    def test_build_optimization(self):
        """测试 build_optimization()"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_baseline,
            baseline_code="def sgemm(A, B):\n    return A @ B",
            func_desc="Matrix-matrix multiplication",
            blas_operation_type="Level 3",
            old_code="def sgemm_old(A, B):\n    # old implementation\n    pass"
        )

        prompt = builder.build_optimization(gen_args)

        assert "optimize the following Triton kernel" in prompt
        assert "def sgemm_old(A, B)" in prompt
        assert "Optimization Goals" in prompt
        assert "cuBLAS" in prompt
        assert "tiling" in prompt or "blocking" in prompt

    def test_type_checking(self):
        """测试类型检查"""
        builder = CupyPromptBuilder(mode="basic")

        # 使用错误的类型应该抛出 TypeError
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        wrong_gen_args = TritonKernelGenerateArgs(
            triton_kernel_name="test",
            func_desc="test",
            torch_kernel_code="test"
        )

        with pytest.raises(TypeError, match="CupyPromptBuilder requires CupyGenerateArgs"):
            builder.build_new(wrong_gen_args)

    def test_blas_level_guidance(self):
        """测试不同 BLAS Level 的指导信息"""
        builder = CupyPromptBuilder(mode="basic")

        def mock_baseline():
            return "mock"

        # Level 1
        gen_args_l1 = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_baseline,
            baseline_code="def saxpy(): pass",
            func_desc="Level 1",
            blas_operation_type="Level 1"
        )
        prompt_l1 = builder.build_new(gen_args_l1)
        assert "vector-vector" in prompt_l1

        # Level 2
        gen_args_l2 = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemv",
            baseline_func=mock_baseline,
            baseline_code="def sgemv(): pass",
            func_desc="Level 2",
            blas_operation_type="Level 2"
        )
        prompt_l2 = builder.build_new(gen_args_l2)
        assert "matrix-vector" in prompt_l2

        # Level 3
        gen_args_l3 = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_baseline,
            baseline_code="def sgemm(): pass",
            func_desc="Level 3",
            blas_operation_type="Level 3"
        )
        prompt_l3 = builder.build_new(gen_args_l3)
        assert "matrix-matrix" in prompt_l3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
