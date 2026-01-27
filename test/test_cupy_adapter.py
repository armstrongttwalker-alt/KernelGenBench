"""
测试 CupyAdapter
"""

import pytest
from flagbench.framework.cupy_adapter import CupyAdapter
from flagbench.framework.generate_args import CupyGenerateArgs


class TestCupyAdapter:
    """测试 CupyAdapter"""

    def test_framework_name(self):
        """测试 framework_name 属性"""
        adapter = CupyAdapter()
        assert adapter.framework_name == "cupy"

    def test_get_operator_function_basic(self):
        """测试 get_operator_function - 基本场景"""
        adapter = CupyAdapter()

        # 测试获取一个存在的算子
        func = adapter.get_operator_function("cupy::caxpy")

        # 验证返回的是函数对象
        assert callable(func)

    def test_get_operator_function_invalid_format(self):
        """测试 get_operator_function - 无效格式"""
        adapter = CupyAdapter()

        # 测试无效格式（缺少 ::）
        with pytest.raises(ValueError, match="Invalid op_name format"):
            adapter.get_operator_function("saxpy")

    def test_get_operator_function_not_found(self):
        """测试 get_operator_function - 算子不存在"""
        adapter = CupyAdapter()

        # 测试不存在的算子
        with pytest.raises(KeyError, match="not found in CUPY_OPERATORS"):
            adapter.get_operator_function("cupy::nonexistent_op")

    def test_get_signature_info(self):
        """测试 get_signature_info"""
        adapter = CupyAdapter()

        # 获取一个算子
        func = adapter.get_operator_function("cupy::caxpy")

        # 获取签名信息
        sig_info = adapter.get_signature_info(func, "cupy::caxpy")

        # 验证返回的字典结构
        assert "signature" in sig_info
        assert "parameters" in sig_info
        assert "return_annotation" in sig_info
        assert "func_desc" in sig_info

        # 验证 func_desc 包含算子名称
        assert "cupy::caxpy" in sig_info["func_desc"] or "caxpy" in sig_info["func_desc"].lower()

    def test_get_reference_code(self):
        """测试 get_reference_code"""
        adapter = CupyAdapter()

        # 获取一个算子
        func = adapter.get_operator_function("cupy::caxpy")

        # 生成参考代码
        ref_code = adapter.get_reference_code(func, "cupy::caxpy")

        # 验证参考代码包含函数定义
        assert "def" in ref_code
        assert callable(func)

    def test_create_generate_args(self):
        """测试 create_generate_args"""
        adapter = CupyAdapter()

        # 获取一个算子
        func = adapter.get_operator_function("cupy::caxpy")

        # 创建 GenerateArgs
        impl_info = None
        gen_args = adapter.create_generate_args("cupy::caxpy", func, impl_info)

        # 验证返回的是 CupyGenerateArgs 实例
        assert isinstance(gen_args, CupyGenerateArgs)

        # 验证字段
        assert gen_args.cupy_kernel_name == "cupy::caxpy"
        assert gen_args.baseline_func == func
        assert gen_args.baseline_code is not None
        assert gen_args.func_desc is not None
        assert gen_args.blas_operation_type in ["Level 1", "Level 2", "Level 3", "Extension"]
        assert gen_args.from_mcp is False

        # 验证 framework_name 属性
        assert gen_args.framework_name == "cupy"

        # 验证 op_name 属性
        assert gen_args.op_name == "cupy::caxpy"

    def test_determine_blas_type_level1(self):
        """测试 _determine_blas_type - Level 1 BLAS"""
        adapter = CupyAdapter()

        # Level 1 operations
        assert adapter._determine_blas_type("cupy::saxpy") == "Level 1"
        assert adapter._determine_blas_type("cupy::daxpy") == "Level 1"
        assert adapter._determine_blas_type("cupy::caxpy") == "Level 1"
        assert adapter._determine_blas_type("cupy::sasum") == "Level 1"
        assert adapter._determine_blas_type("cupy::sdot") == "Level 1"
        assert adapter._determine_blas_type("cupy::snrm2") == "Level 1"
        assert adapter._determine_blas_type("cupy::sscal") == "Level 1"

    def test_determine_blas_type_level2(self):
        """测试 _determine_blas_type - Level 2 BLAS"""
        adapter = CupyAdapter()

        # Level 2 operations
        assert adapter._determine_blas_type("cupy::sgemv") == "Level 2"
        assert adapter._determine_blas_type("cupy::dgemv") == "Level 2"
        assert adapter._determine_blas_type("cupy::sger") == "Level 2"
        assert adapter._determine_blas_type("cupy::ssymv") == "Level 2"

    def test_determine_blas_type_level3(self):
        """测试 _determine_blas_type - Level 3 BLAS"""
        adapter = CupyAdapter()

        # Level 3 operations
        assert adapter._determine_blas_type("cupy::sgemm") == "Level 3"
        assert adapter._determine_blas_type("cupy::dgemm") == "Level 3"
        assert adapter._determine_blas_type("cupy::cgemm") == "Level 3"
        assert adapter._determine_blas_type("cupy::ssyrk") == "Level 3"
        assert adapter._determine_blas_type("cupy::strmm") == "Level 3"

    def test_determine_blas_type_extension(self):
        """测试 _determine_blas_type - Extension operations"""
        adapter = CupyAdapter()

        # Extension operations
        assert adapter._determine_blas_type("cupy::sgeam") == "Extension"
        assert adapter._determine_blas_type("cupy::sdgmm") == "Extension"

    def test_get_impl_info(self):
        """测试 get_impl_info"""
        adapter = CupyAdapter()

        # 测试 cupy 的 get_impl_info 应该返回 None
        impl_info = adapter.get_impl_info("caxpy")

        # 验证返回 None（cupy 不需要 torch 的 impl_info）
        assert impl_info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
