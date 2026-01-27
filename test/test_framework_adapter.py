"""
测试 FrameworkAdapter 基类
"""

import pytest
from typing import Any, Dict
from flagbench.framework.adapter import FrameworkAdapter
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


class MockAdapter(FrameworkAdapter):
    """用于测试的 Mock Adapter"""

    def get_operator_function(self, op_name: str) -> Any:
        """返回 mock 函数"""
        def mock_func():
            return f"mock_func_{op_name}"
        return mock_func

    def get_signature_info(self, func: Any, op_name: str) -> Dict:
        """返回 mock 签名信息"""
        return {
            "op_name": op_name,
            "func": str(func),
            "parameters": ["param1", "param2"]
        }

    def create_generate_args(self, op_name: str, func: Any, impl_info: Any) -> BaseGenerateArgs:
        """创建 mock GenerateArgs"""
        return MockGenerateArgs(test_op_name=op_name)

    def get_reference_code(self, func: Any, op_name: str) -> str:
        """返回 mock 参考代码"""
        return f"# Reference code for {op_name}\ndef {op_name}(): pass"

    @property
    def framework_name(self) -> str:
        return "mock_framework"


class TestFrameworkAdapter:
    """测试 FrameworkAdapter 基类"""

    def test_get_operator_function(self):
        """测试 get_operator_function 方法"""
        adapter = MockAdapter()
        func = adapter.get_operator_function("test_op")

        assert callable(func)
        assert func() == "mock_func_test_op"

    def test_get_signature_info(self):
        """测试 get_signature_info 方法"""
        adapter = MockAdapter()
        func = lambda: None
        sig_info = adapter.get_signature_info(func, "test_op")

        assert isinstance(sig_info, dict)
        assert sig_info["op_name"] == "test_op"
        assert "parameters" in sig_info

    def test_create_generate_args(self):
        """测试 create_generate_args 方法"""
        adapter = MockAdapter()
        func = lambda: None
        gen_args = adapter.create_generate_args("test_op", func, None)

        assert isinstance(gen_args, BaseGenerateArgs)
        assert gen_args.op_name == "test_op"

    def test_get_reference_code(self):
        """测试 get_reference_code 方法"""
        adapter = MockAdapter()
        func = lambda: None
        ref_code = adapter.get_reference_code(func, "test_op")

        assert isinstance(ref_code, str)
        assert "test_op" in ref_code
        assert "def" in ref_code

    def test_framework_name_property(self):
        """测试 framework_name 属性"""
        adapter = MockAdapter()
        assert adapter.framework_name == "mock_framework"

    def test_cannot_instantiate_abstract_class(self):
        """测试不能直接实例化抽象基类"""
        with pytest.raises(TypeError):
            FrameworkAdapter()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
