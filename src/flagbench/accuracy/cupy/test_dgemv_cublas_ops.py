import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dgemv")
@parametrize("m, n", [
    (1, 32),
    (160, 1024),
    (5333, 497),
    (33, 273),
    (16, 256),
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
@parametrize("dtype", [torch.float64])
def test_dgemv_cublas_baseline(m, n, alpha, beta, trans, dtype):
    A = torch.randn((m, n), dtype=dtype, device='cuda')
    lda = A.shape[0]
    if trans == "N":
        x = torch.randn((n,), dtype=dtype, device='cuda')
        y_init = torch.randn((m,), dtype=dtype, device='cuda')
    else:
        x = torch.randn((m,), dtype=dtype, device='cuda')
        y_init = torch.randn((n,), dtype=dtype, device='cuda')
    incx = 1
    incy = 1

    y_baseline = y_init.clone()
    y_triton = y_init.clone()

    ref_out = flagbench.baseline.dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y_baseline, incy)
    act_out = flagbench.triton.dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y_triton, incy)

    assert_close(act_out, ref_out, dtype)