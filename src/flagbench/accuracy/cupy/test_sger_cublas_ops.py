import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sger")
@parametrize("M, N", [
    (1, 1),
    (1, 32),
    (15, 160),
    (160, 1024),
    (495, 5333),
    (5333, 497),
    (33, 273),
    (33, 4113),
    (16, 256),
    (256, 4096),
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
    (3, 4),
])
@parametrize("dtype", [torch.float32])
def test_sger_cublas_baseline(M, N, alpha, incx, incy, dtype):
    x_store_len = (M - 1) * incx + 1
    y_store_len = (N - 1) * incy + 1
    x_store = torch.randn(x_store_len, dtype=dtype, device='cuda')
    y_store = torch.randn(y_store_len, dtype=dtype, device='cuda')
    x = x_store[::incx]
    y = y_store[::incy]
    A_init = torch.randn((M, N), dtype=dtype, device='cuda')
    A_baseline = A_init.clone()
    A_triton = A_init.clone()
    m = M
    n = N
    lda = M
    ref_out = flagbench.baseline.sger(m, n, alpha, x, incx, y, incy, A_baseline, lda)
    act_out = flagbench.triton.sger(m, n, alpha, x, incx, y, incy, A_triton, lda)
    assert_close(act_out, ref_out, dtype)