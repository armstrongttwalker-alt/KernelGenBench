import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dscal")
@parametrize("n", [
    32,       # Edge case
    1024,     # Non-symmetric
    71,       # Non-symmetric
    4113,     # Non-aligned
    4096,     # Power-of-2 aligned
    497,      # 2D GEMV-like
])
@parametrize("alpha", [
    1.0,
    0.0,
    0.001,
    -0.999,
    100.001,
    -111.999,
    0.5,
    -0.5,
])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.float64])
def test_dscal_cublas_baseline(n, alpha, incx, dtype):
    base_len = (n - 1) * incx + 1

    base_b = torch.randn(base_len, dtype=dtype, device='cuda')
    base_t = base_b.clone()

    x_b = base_b[::incx]
    x_t = base_t[::incx]

    ref_out = flagbench.baseline.dscal(n, alpha, x_b, incx)
    act_out = flagbench.triton.dscal(n, alpha, x_t, incx)

    assert_close(act_out, ref_out, dtype)