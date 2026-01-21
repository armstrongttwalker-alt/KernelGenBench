import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cdotc")
@parametrize("incx, incy, n", [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (33, 273, 4113),
    (16, 256, 4096),
    (160, 160, 1024),
    (5333, 5333, 497),
])
@parametrize("dtype", [torch.complex64])
def test_cdotc_cublas_baseline(incx, incy, n, dtype):
    length_x = (n - 1) * incx + 1
    length_y = (n - 1) * incy + 1

    x_base = torch.randn(length_x, dtype=dtype, device="cuda")
    y_base = torch.randn(length_y, dtype=dtype, device="cuda")

    x = x_base[::incx]
    y = y_base[::incy]

    result = torch.empty(1, dtype=dtype, device="cuda")

    ref_out = flagbench.baseline.cdotc(n, x, incx, y, incy, result)
    act_out = flagbench.triton.cdotc(n, x, incx, y, incy, result)

    assert_close(act_out, ref_out, dtype)