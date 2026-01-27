"""
测试 generate_kernel_and_verify.py 的 Cupy 集成
"""

import pytest
import sys
from pathlib import Path

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from generator.cupy_prompt_builder import CupyPromptBuilder


class TestCupyIntegration:
    """测试 Cupy 集成"""

    def test_cupy_prompt_builder_basic_mode(self):
        """测试 CupyPromptBuilder 基本模式创建"""
        prompt_builder = CupyPromptBuilder(mode="basic")

        assert prompt_builder is not None
        assert isinstance(prompt_builder, CupyPromptBuilder)
        assert prompt_builder.mode == "basic"

    def test_cupy_prompt_builder_with_wiki_mode(self):
        """测试 CupyPromptBuilder with_wiki 模式创建"""
        prompt_builder = CupyPromptBuilder(mode="with_wiki")

        assert prompt_builder is not None
        assert isinstance(prompt_builder, CupyPromptBuilder)
        assert prompt_builder.mode == "with_wiki"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
