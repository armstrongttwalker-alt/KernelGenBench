"""
测试 BaseGenerateArgs 基类
"""

import pytest
from flagbench.framework.generate_args import BaseGenerateArgs


class MockGenerateArgs(BaseGenerateArgs):
    """用于测试的 Mock GenerateArgs"""

    mock_op_name: str = "test_op"
    mock_framework: str = "test_framework"

    @property
    def op_name(self):
        return self.mock_op_name

    @property
    def framework_name(self) -> str:
        return self.mock_framework


class TestBaseGenerateArgs:
    """测试 BaseGenerateArgs 基类"""

    def test_basic_fields(self):
        """测试基本字段的定义和默认值"""
        args = MockGenerateArgs()

        # 测试默认值
        assert args.from_mcp is False
        assert args.user_advice is None
        assert args.check_result is None
        assert args.old_code is None
        assert args.sample_id == 0
        assert args.wiki_reference is None

    def test_field_assignment(self):
        """测试字段赋值"""
        args = MockGenerateArgs(
            from_mcp=True,
            user_advice="test advice",
            sample_id=5
        )

        assert args.from_mcp is True
        assert args.user_advice == "test advice"
        assert args.sample_id == 5

    def test_op_name_property(self):
        """测试 op_name 抽象属性"""
        args = MockGenerateArgs(mock_op_name="custom_op")
        assert args.op_name == "custom_op"

    def test_framework_name_property(self):
        """测试 framework_name 抽象属性"""
        args = MockGenerateArgs(mock_framework="custom_framework")
        assert args.framework_name == "custom_framework"

    def test_arbitrary_types_allowed(self):
        """测试 Config.arbitrary_types_allowed"""
        # 测试可以存储任意类型（如函数对象）
        def mock_func():
            return "test"

        args = MockGenerateArgs(check_result=mock_func)
        assert callable(args.check_result)
        assert args.check_result() == "test"

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            BaseGenerateArgs()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
