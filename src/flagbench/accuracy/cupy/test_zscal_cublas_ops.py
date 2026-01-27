import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zscal")
@parametrize("n", [
    1, 32, 15, 160, 1024,
    495, 5333, 71, 33, 273,
    4113, 16, 256, 4096, 497,
])
@parametrize("alpha", [
    1.0 + 0.0j,
    0.0 + 0.0j,
    0.001 - 0.999j,
    100.001 - 111.999j,
    0.5 + 0.5j,
    -0.5 - 0.5j,
])
@parametrize("incx", [1, 2, 4])
@parametrize("dtype", [torch.complex128])
def test_zscal_cublas_baseline(n, alpha, incx, dtype):
    base_len = 1 + (n - 1) * incx
    base0 = torch.randn(base_len, dtype=dtype, device="cuda")
    base1 = base0.clone()
    x0 = base0[::incx]
    x1 = base1[::incx]

    ref_out = flagbench.baseline.zscal(n, alpha, x0, incx)
    act_out = flagbench.triton.zscal(n, alpha, x1, incx)

    assert_close(act_out, ref_out, dtype)