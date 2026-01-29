"""
测试 generate_kernel_and_verify.py 的 PromptBuilder 创建逻辑

验证步骤5的修改：
1. TorchPromptBuilder 能正确创建
"""

import pytest
import sys
from pathlib import Path

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from generator.torch_prompt_builder import TorchPromptBuilder


class TestPromptBuilderCreation:
    """测试 PromptBuilder 创建逻辑"""

    def test_torch_prompt_builder_basic_mode(self):
        """测试 TorchPromptBuilder 基本模式创建"""
        prompt_builder = TorchPromptBuilder(mode="basic")

        assert prompt_builder is not None
        assert isinstance(prompt_builder, TorchPromptBuilder)
        assert prompt_builder.mode == "basic"

    def test_torch_prompt_builder_with_wiki_mode(self):
        """测试 TorchPromptBuilder with_wiki 模式创建"""
        prompt_builder = TorchPromptBuilder(mode="with_wiki")

        assert prompt_builder is not None
        assert isinstance(prompt_builder, TorchPromptBuilder)
        assert prompt_builder.mode == "with_wiki"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
