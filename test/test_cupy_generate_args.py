"""
测试 CupyGenerateArgs
"""

import pytest
from flagbench.framework.generate_args import CupyGenerateArgs


class TestCupyGenerateArgs:
    """测试 CupyGenerateArgs"""

    def test_basic_instantiation(self):
        """测试基本实例化"""
        def mock_baseline_func():
            pass

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_baseline_func,
            baseline_code="def sgemm(): pass",
            func_desc="Single precision matrix multiplication",
            blas_operation_type="Level 3"
        )

        assert gen_args is not None
        assert isinstance(gen_args, CupyGenerateArgs)

    def test_op_name_property(self):
        """测试 op_name 属性"""
        def mock_baseline_func():
            pass

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_baseline_func,
            baseline_code="def saxpy(): pass",
            func_desc="Single precision a*x + y",
            blas_operation_type="Level 1"
        )

        assert gen_args.op_name == "cublas::saxpy"

    def test_framework_name_property(self):
        """测试 framework_name 属性"""
        def mock_baseline_func():
            pass

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_baseline_func,
            baseline_code="def sgemm(): pass",
            func_desc="Single precision matrix multiplication",
            blas_operation_type="Level 3"
        )

        assert gen_args.framework_name == "cupy"

    def test_all_fields(self):
        """测试所有字段"""
        def mock_baseline_func(a, b):
            return a + b

        baseline_code = """def mock_baseline_func(a, b):
    return a + b"""

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::daxpy",
            baseline_func=mock_baseline_func,
            baseline_code=baseline_code,
            func_desc="Double precision a*x + y",
            blas_operation_type="Level 1",
            impl_info={"precision": "double"}
        )

        assert gen_args.cupy_kernel_name == "cublas::daxpy"
        assert gen_args.baseline_func == mock_baseline_func
        assert gen_args.baseline_code == baseline_code
        assert gen_args.func_desc == "Double precision a*x + y"
        assert gen_args.blas_operation_type == "Level 1"
        assert gen_args.impl_info == {"precision": "double"}

    def test_blas_operation_types(self):
        """测试不同的 BLAS 操作类型"""
        def mock_func():
            pass

        # Level 1
        gen_args_l1 = CupyGenerateArgs(
            cupy_kernel_name="cublas::saxpy",
            baseline_func=mock_func,
            baseline_code="def saxpy(): pass",
            func_desc="Level 1 operation",
            blas_operation_type="Level 1"
        )
        assert gen_args_l1.blas_operation_type == "Level 1"

        # Level 2
        gen_args_l2 = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemv",
            baseline_func=mock_func,
            baseline_code="def sgemv(): pass",
            func_desc="Level 2 operation",
            blas_operation_type="Level 2"
        )
        assert gen_args_l2.blas_operation_type == "Level 2"

        # Level 3
        gen_args_l3 = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_func,
            baseline_code="def sgemm(): pass",
            func_desc="Level 3 operation",
            blas_operation_type="Level 3"
        )
        assert gen_args_l3.blas_operation_type == "Level 3"

        # Extension
        gen_args_ext = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgeam",
            baseline_func=mock_func,
            baseline_code="def sgeam(): pass",
            func_desc="Extension operation",
            blas_operation_type="Extension"
        )
        assert gen_args_ext.blas_operation_type == "Extension"

    def test_common_fields_from_base(self):
        """测试从 BaseGenerateArgs 继承的通用字段"""
        def mock_func():
            pass

        gen_args = CupyGenerateArgs(
            cupy_kernel_name="cublas::sgemm",
            baseline_func=mock_func,
            baseline_code="def sgemm(): pass",
            func_desc="Matrix multiplication",
            blas_operation_type="Level 3",
            from_mcp=True,
            user_advice="Optimize for large matrices",
            old_code="old implementation",
            sample_id=5
        )

        assert gen_args.from_mcp is True
        assert gen_args.user_advice == "Optimize for large matrices"
        assert gen_args.old_code == "old implementation"
        assert gen_args.sample_id == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
