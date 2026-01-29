import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zdotc")
@parametrize("n", [
    1,
    32,
    160,
    1024,
    5333,
    71,
    273,
    4113,
    4096,
    497,
])
@parametrize("incx, incy", [
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 2),
])
@parametrize("dtype", [torch.complex128])
def test_zdotc_cublas_baseline(n, incx, incy, dtype):
    x_base = torch.randn(n * incx, dtype=torch.float64, device="cuda") + 1j * torch.randn(n * incx, dtype=torch.float64, device="cuda")
    y_base = torch.randn(n * incy, dtype=torch.float64, device="cuda") + 1j * torch.randn(n * incy, dtype=torch.float64, device="cuda")
    x = x_base[::incx][:n].to(dtype)
    y = y_base[::incy][:n].to(dtype)
    result = torch.empty(1, dtype=dtype, device="cuda")
    ref_out = flagbench.baseline.zdotc(n, x, incx, y, incy, result)
    act_out = flagbench.triton.zdotc(n, x, incx, y, incy, result)
    assert_close(act_out, ref_out, dtype, reduce_dim=n)