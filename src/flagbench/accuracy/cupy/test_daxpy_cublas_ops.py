import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("daxpy")
@parametrize("n", [
    1,
    32,     # from (1,1,32)
    15, 160, 1024,   # from (15,160,1024)
    495, 5333, 71,   # from (495,5333,71)
    33, 273, 4113,   # non-aligned
    16, 256, 4096,   # power-of-2 aligned
    497,             # from (5333,497)
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
@parametrize("incx, incy", [
    (1, 1),
    (2, 1),
    (1, 2),
    (4, 2),
    (3, 3),
    (8, 1),
])
@parametrize("dtype", [torch.float64])
def test_daxpy_cublas_baseline(n, alpha, incx, incy, dtype):
    base_len_x = n * incx
    base_len_y = n * incy

    x_buf = torch.randn(base_len_x, dtype=dtype, device="cuda")
    x = x_buf[::incx]

    y_buf = torch.randn(base_len_y, dtype=dtype, device="cuda")
    y_baseline = y_buf[::incy]
    y_triton = y_buf.clone()[::incy]

    ref_out = flagbench.baseline.daxpy(n, alpha, x, incx, y_baseline, incy)
    act_out = flagbench.triton.daxpy(n, alpha, x, incx, y_triton, incy)

    assert_close(act_out, ref_out, dtype)