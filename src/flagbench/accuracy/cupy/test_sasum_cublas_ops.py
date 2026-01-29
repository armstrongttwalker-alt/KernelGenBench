import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sasum")
@parametrize("N", [
    32,          # Edge case (minimum valid size)
    15, 160, 1024,  # Non-symmetric categories
    495, 5333, 71,  # Non-symmetric categories
    33, 273, 4113,  # Non-aligned sizes
    16, 256, 4096   # Power-of-2 aligned
])
@parametrize("make_strided", [False, True])
@parametrize("dtype", [torch.float32])
def test_sasum_cublas_baseline(N, make_strided, dtype):
    if make_strided:
        base = torch.randn(N * 2, dtype=dtype, device='cuda')
        x = base[::2]
    else:
        x = torch.randn(N, dtype=dtype, device='cuda')

    n = N
    incx = 1
    result = torch.empty(1, dtype=dtype, device='cuda')

    ref_out = flagbench.baseline.sasum(n, x, incx, result)
    act_out = flagbench.triton.sasum(n, x, incx, result)

    assert_close(act_out, ref_out, dtype, reduce_dim=n)