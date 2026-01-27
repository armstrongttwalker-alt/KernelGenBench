import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sdot")
@parametrize("n", [
    1,
    32,          # Edge case
    15,          # Non-symmetric small
    160,         # Non-symmetric
    1024,        # Non-symmetric large
    495,         # Non-symmetric
    5333,        # Non-symmetric large
    71,          # Non-symmetric
    33,          # Non-aligned small
    273,         # Non-aligned
    4113,        # Non-aligned large
    16,          # Power-of-2 small
    256,         # Power-of-2
    4096,        # Power-of-2 large
    497          # From GEMV shapes
])
@parametrize("incx", [1, 2, 4])
@parametrize("incy", [1, 3, 5])
@parametrize("dtype", [torch.float32])
def test_sdot_cublas_baseline(n, incx, incy, dtype):
    Lx = 1 + (n - 1) * incx
    Ly = 1 + (n - 1) * incy
    base_x = torch.randn(Lx, dtype=dtype, device='cuda')
    base_y = torch.randn(Ly, dtype=dtype, device='cuda')
    x = base_x[0:Lx:incx]
    y = base_y[0:Ly:incy]
    result = torch.empty(1, dtype=dtype, device='cuda')

    ref_out = flagbench.baseline.sdot(n, x, incx, y, incy, result)
    act_out = flagbench.triton.sdot(n, x, incx, y, incy, result)

    assert_close(act_out, ref_out, dtype)