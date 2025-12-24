#!/usr/bin/env python3
"""Test script for accuracy to performance test conversion."""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import convert_accuracy_to_performance_test

# Sample accuracy test function (from soft_margin_loss)
SAMPLE_TEST_FUNC = """import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("soft_margin_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_tensor(shape, dtype, reduction):
    self = torch.randn(shape, dtype=dtype, device="cuda")
    target = (torch.randint(0, 2, shape, device="cuda").to(dtype) * 2) - 1

    ref_self = self.clone()
    ref_target = target.clone()
    ref_out = torch.ops.aten.soft_margin_loss(ref_self, ref_target, reduction)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.soft_margin_loss(self, target, reduction)

    assert_close(act_out, ref_out, dtype=dtype)
"""


def main():
    print("=" * 80)
    print("Testing Accuracy to Performance Test Conversion")
    print("=" * 80)
    print()

    # Test the conversion
    test_funcs = {
        "aten::soft_margin_loss": SAMPLE_TEST_FUNC
    }

    print("Original Test Function:")
    print("-" * 80)
    print(SAMPLE_TEST_FUNC)
    print()

    print("Converting...")
    print()

    try:
        converted = convert_accuracy_to_performance_test(test_funcs)

        print("Converted Performance Test Function:")
        print("-" * 80)
        print(converted["aten::soft_margin_loss"])
        print()

        print("✓ Conversion successful!")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
