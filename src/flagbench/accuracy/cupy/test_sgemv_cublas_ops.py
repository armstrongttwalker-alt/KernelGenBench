import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgemv")
@parametrize("M, N", [
    (1, 32),
    (160, 1024),
    (5333, 497),
    (33, 273),
    (16, 256),
    (495, 5333),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("trans", ["N", "T"])
@parametrize("incx", [1, 2])
@parametrize("incy", [1, 3])
@parametrize("dtype", [torch.float32])
def test_sgemv_cublas_baseline(M, N, alpha, beta, trans, incx, incy, dtype):
    A = torch.randn((M, N), dtype=dtype, device="cuda")
    lda = max(1, M)
    kdim = N if trans in ("N", "n") else M
    ydim = M if trans in ("N", "n") else N

    x_store = torch.randn(1 + (kdim - 1) * incx, dtype=dtype, device="cuda")
    x = x_store[::incx]

    y_store = torch.randn(1 + (ydim - 1) * incy, dtype=dtype, device="cuda")
    y_baseline = y_store[::incy]
    y_triton = y_store.clone()[::incy]

    ref_out = flagbench.baseline.sgemv(trans, M, N, alpha, A, lda, x, incx, beta, y_baseline, incy)
    act_out = flagbench.triton.sgemv(trans, M, N, alpha, A, lda, x, incx, beta, y_triton, incy)

    assert_close(act_out, ref_out, dtype, reduce_dim=kdim)