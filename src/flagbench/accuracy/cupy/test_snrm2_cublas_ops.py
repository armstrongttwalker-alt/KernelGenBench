import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("snrm2")
@parametrize("n", [
    1,      # Edge case
    32,     # Edge case (minimum valid size)
    15,     # Non-symmetric
    160,    # Non-symmetric
    1024,   # Non-symmetric / power-of-2 aligned
    495,    # Non-symmetric
    5333,   # Non-symmetric (large)
    71,     # Non-symmetric
    33,     # Non-aligned
    273,    # Non-aligned
    4113,   # Non-aligned
    16,     # Power-of-2 aligned
    256,    # Power-of-2 aligned
    4096,   # Power-of-2 aligned
    497,    # 2D ops list coverage
])
@parametrize("incx", [1])
@parametrize("dtype", [torch.float32])
def test_snrm2_cublas_baseline(n, incx, dtype):
    x = torch.randn((n,), dtype=dtype, device='cuda')
    result = torch.empty(1, dtype=dtype, device='cuda')

    ref_out = flagbench.baseline.snrm2(n, x, incx, result)
    act_out = flagbench.triton.snrm2(n, x, incx, result)

    assert_close(act_out, ref_out, dtype, reduce_dim=n)