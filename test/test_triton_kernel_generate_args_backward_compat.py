"""
测试 TritonKernelGenerateArgs 的向后兼容性
"""

import pytest


class TestTritonKernelGenerateArgsBackwardCompatibility:
    """测试 TritonKernelGenerateArgs 向后兼容性"""

    def test_old_import_path_works(self):
        """测试旧的导入路径仍然有效"""
        from generator.sampler.generate_samples import TritonKernelGenerateArgs

        # 应该能够成功导入
        assert TritonKernelGenerateArgs is not None

    def test_new_import_path_works(self):
        """测试新的导入路径有效"""
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        # 应该能够成功导入
        assert TritonKernelGenerateArgs is not None

    def test_both_paths_point_to_same_class(self):
        """测试两个导入路径指向同一个类"""
        from generator.sampler.generate_samples import TritonKernelGenerateArgs as OldTritonKernelGenerateArgs
        from flagbench.framework.generate_args import TritonKernelGenerateArgs as NewTritonKernelGenerateArgs

        # 应该是同一个类
        assert OldTritonKernelGenerateArgs is NewTritonKernelGenerateArgs

    def test_can_create_instance_from_old_path(self):
        """测试可以从旧路径创建实例"""
        from generator.sampler.generate_samples import TritonKernelGenerateArgs

        instance = TritonKernelGenerateArgs(
            triton_kernel_name="test_kernel",
            func_desc="Test function",
            torch_kernel_code="def test(): pass"
        )

        assert instance.triton_kernel_name == "test_kernel"
        assert instance.func_desc == "Test function"
        assert instance.torch_kernel_code == "def test(): pass"

    def test_can_create_instance_from_new_path(self):
        """测试可以从新路径创建实例"""
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        instance = TritonKernelGenerateArgs(
            triton_kernel_name="test_kernel",
            func_desc="Test function",
            torch_kernel_code="def test(): pass"
        )

        assert instance.triton_kernel_name == "test_kernel"
        assert instance.func_desc == "Test function"
        assert instance.torch_kernel_code == "def test(): pass"

    def test_all_fields_preserved(self):
        """测试所有字段都被保留"""
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        instance = TritonKernelGenerateArgs(
            triton_kernel_name="test_kernel",
            func_desc="Test function",
            torch_kernel_code="def test(): pass",
            input_args={"arg1": "type1"},
            output_args={"out1": "type1"},
            func_type="unary",
            impl_info=["op1", "op2"],
            from_mcp=True,
            user_advice="Some advice",
            old_code="old code",
            sample_id=5,
            wiki_reference={"ref": "data"}
        )

        # 验证所有字段
        assert instance.triton_kernel_name == "test_kernel"
        assert instance.func_desc == "Test function"
        assert instance.torch_kernel_code == "def test(): pass"
        assert instance.input_args == {"arg1": "type1"}
        assert instance.output_args == {"out1": "type1"}
        assert instance.func_type == "unary"
        assert instance.impl_info == ["op1", "op2"]
        assert instance.from_mcp is True
        assert instance.user_advice == "Some advice"
        assert instance.old_code == "old code"
        assert instance.sample_id == 5
        assert instance.wiki_reference == {"ref": "data"}

    def test_op_name_property(self):
        """测试 op_name 属性"""
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        instance = TritonKernelGenerateArgs(
            triton_kernel_name="test_kernel",
            func_desc="Test function",
            torch_kernel_code="def test(): pass"
        )

        assert instance.op_name == "test_kernel"

    def test_framework_name_property(self):
        """测试 framework_name 属性"""
        from flagbench.framework.generate_args import TritonKernelGenerateArgs

        instance = TritonKernelGenerateArgs(
            triton_kernel_name="test_kernel",
            func_desc="Test function",
            torch_kernel_code="def test(): pass"
        )

        assert instance.framework_name == "torch"

    def test_input_arg_and_output_arg_classes(self):
        """测试 InputArg 和 OutputArg 类"""
        from generator.sampler.generate_samples import InputArg, OutputArg
        from flagbench.framework.generate_args import InputArg as NewInputArg, OutputArg as NewOutputArg

        # 验证两个导入路径指向同一个类
        assert InputArg is NewInputArg
        assert OutputArg is NewOutputArg

        # 验证可以创建实例
        input_arg = InputArg(
            arg_name="x",
            arg_type="Tensor",
            arg_value=None,
            arg_default=None,
            arg_desc="Input tensor"
        )
        assert input_arg.arg_name == "x"
        assert input_arg.arg_type == "Tensor"

        output_arg = OutputArg(
            arg_type="Tensor",
            arg_value=None,
            arg_desc="Output tensor"
        )
        assert output_arg.arg_type == "Tensor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
