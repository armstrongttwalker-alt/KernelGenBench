"""
测试 TorchAdapter
"""

import pytest
import torch
from flagbench.framework.torch_adapter import TorchAdapter
from flagbench.framework.generate_args import TritonKernelGenerateArgs


class TestTorchAdapter:
    """测试 TorchAdapter"""

    def test_framework_name(self):
        """测试 framework_name 属性"""
        adapter = TorchAdapter()
        assert adapter.framework_name == "torch"

    def test_get_operator_function_basic(self):
        """测试 get_operator_function - 基本场景"""
        adapter = TorchAdapter()

        # 测试获取 aten::add 算子
        func = adapter.get_operator_function("aten::add")

        # 验证返回的是函数对象
        assert callable(func)
        assert hasattr(func, '_schemas')

    def test_get_operator_function_invalid_format(self):
        """测试 get_operator_function - 无效格式"""
        adapter = TorchAdapter()

        # 测试无效格式（缺少 ::）
        with pytest.raises(ValueError, match="Invalid op_name format"):
            adapter.get_operator_function("add")

    def test_get_operator_function_invalid_namespace(self):
        """测试 get_operator_function - 无效 namespace"""
        adapter = TorchAdapter()

        # 测试不存在的 namespace
        # 注意：torch.ops 会动态创建 namespace，所以错误会在获取 kernel 时抛出
        with pytest.raises(AttributeError, match="does not have kernel"):
            adapter.get_operator_function("invalid_namespace::add")

    def test_get_signature_info(self):
        """测试 get_signature_info"""
        adapter = TorchAdapter()

        # 获取 aten::add 算子
        func = adapter.get_operator_function("aten::add")

        # 获取签名信息
        sig_info = adapter.get_signature_info(func, "aten::add")

        # 验证返回的字典结构
        assert "input_args" in sig_info
        assert "output_args" in sig_info
        assert "func_desc" in sig_info

        # 验证 input_args 是字典（包含 overload 信息）
        assert isinstance(sig_info["input_args"], dict)

        # 验证 func_desc 包含算子名称
        assert "aten::add" in sig_info["func_desc"] or "add" in sig_info["func_desc"].lower()

    def test_get_reference_code(self):
        """测试 get_reference_code"""
        adapter = TorchAdapter()

        # 获取 aten::add 算子
        func = adapter.get_operator_function("aten::add")

        # 生成参考代码
        ref_code = adapter.get_reference_code(func, "aten::add")

        # 验证参考代码包含关键内容
        assert "import torch" in ref_code
        assert "torch.ops.aten.add" in ref_code
        assert "add =" in ref_code
        assert "Reference PyTorch implementation" in ref_code

    def test_create_generate_args(self):
        """测试 create_generate_args"""
        adapter = TorchAdapter()

        # 获取 aten::add 算子
        func = adapter.get_operator_function("aten::add")

        # 创建 GenerateArgs
        impl_info = [("aten.add.Tensor", None), ("aten.add.Scalar", None)]
        gen_args = adapter.create_generate_args("aten::add", func, impl_info)

        # 验证返回的是 TritonKernelGenerateArgs 实例
        assert isinstance(gen_args, TritonKernelGenerateArgs)

        # 验证字段
        assert gen_args.triton_kernel_name == "aten::add"
        assert gen_args.func_desc is not None
        assert gen_args.torch_kernel_code is not None
        assert gen_args.input_args is not None
        assert gen_args.impl_info == impl_info
        assert gen_args.from_mcp is False

        # 验证 framework_name 属性
        assert gen_args.framework_name == "torch"

        # 验证 op_name 属性
        assert gen_args.op_name == "aten::add"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


