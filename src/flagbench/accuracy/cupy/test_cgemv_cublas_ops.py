import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgemv")
@parametrize("M, N", [
    (1, 32),
    (160, 1024),
    (5333, 497),
    (256, 4096),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("trans", ["N", "T", "H"])
@parametrize("dtype", [torch.complex64])
def test_cgemv_cublas_baseline(M, N, alpha, beta, trans, dtype):
    if trans == 'N':
        x_len = N
        y_len = M
        K = N
    else:
        x_len = M
        y_len = N
        K = M

    A = torch.randn((M, N), dtype=dtype, device='cuda')
    x = torch.randn((x_len,), dtype=dtype, device='cuda')
    y0 = torch.randn((y_len,), dtype=dtype, device='cuda')

    lda = M
    incx = 1
    incy = 1

    y_baseline = y0.clone()
    y_triton = y0.clone()

    ref_out = flagbench.baseline.cgemv(trans, M, N, alpha, A, lda, x, incx, beta, y_baseline, incy)
    act_out = flagbench.triton.cgemv(trans, M, N, alpha, A, lda, x, incx, beta, y_triton, incy)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)