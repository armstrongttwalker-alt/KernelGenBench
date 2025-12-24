#!/usr/bin/env python3
"""Test conversion for mv operator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import convert_accuracy_to_performance_test

# mv operator test functions
MV_TEST_FUNC = """import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

import torch

@label("mv")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_tensor(shape, dtype):
    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device='cuda')
    vec = torch.randn((n,), dtype=dtype, device='cuda')

    ref_mat = mat.clone()
    ref_vec = vec.clone()
    ref_out = torch.ops.aten.mv(ref_mat, ref_vec)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mv(mat, vec)

    assert_close(act_out, ref_out, dtype=dtype)


@label("mv")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mv_out_tensor(shape, dtype):
    m, n = shape
    mat = torch.randn((m, n), dtype=dtype, device='cuda')
    vec = torch.randn((n,), dtype=dtype, device='cuda')

    ref_mat = mat.clone()
    ref_vec = vec.clone()
    ref_out_buf = torch.empty((m,), dtype=dtype, device='cuda')
    ref_out = torch.ops.aten.mv.out(ref_mat, ref_vec, out=ref_out_buf)

    act_out_buf = torch.empty((m,), dtype=dtype, device='cuda')
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mv.out(mat, vec, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)
"""

def main():
    print("=" * 80)
    print("Testing MV Operator Conversion")
    print("=" * 80)
    print()

    test_funcs = {"aten::mv": MV_TEST_FUNC}

    print("Original Test Functions:")
    print("-" * 80)
    print(MV_TEST_FUNC)
    print()

    print("Converting...")
    print()

    try:
        converted = convert_accuracy_to_performance_test(test_funcs)

        print("Converted Performance Test Functions:")
        print("-" * 80)
        print(converted["aten::mv"])
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
