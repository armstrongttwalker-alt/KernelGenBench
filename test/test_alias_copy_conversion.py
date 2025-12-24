#!/usr/bin/env python3
"""Test conversion for alias_copy operator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import convert_accuracy_to_performance_test

ALIAS_COPY_TEST = """import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("alias_copy")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_alias_copy_out(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device="cuda")

    ref_input = input_tensor.clone()
    ref_out = torch.empty_like(ref_input)
    torch.ops.aten.alias_copy.out(ref_input, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.empty_like(input_tensor)
        torch.ops.aten.alias_copy.out(input_tensor, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)
"""

def main():
    print("Testing alias_copy_out conversion")
    print("=" * 80)

    test_funcs = {"aten::alias_copy": ALIAS_COPY_TEST}

    try:
        converted = convert_accuracy_to_performance_test(test_funcs)
        print("Converted:")
        print("-" * 80)
        print(converted["aten::alias_copy"])

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
