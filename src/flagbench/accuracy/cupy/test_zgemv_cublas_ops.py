import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgemv")
@parametrize("M, N", [
    (1, 32),
    (160, 1024),
    (5333, 497),
    (33, 273),
    (16, 256),
    (495, 5333),
    (273, 4113),
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
@parametrize("dtype", [torch.complex128])
def test_zgemv_cublas_baseline(M, N, alpha, beta, trans, dtype):
    m, n = M, N
    A = torch.randn((m, n), dtype=dtype, device="cuda")
    if trans == "N":
        x = torch.randn((n,), dtype=dtype, device="cuda")
        y = torch.randn((m,), dtype=dtype, device="cuda")
    else:
        x = torch.randn((m,), dtype=dtype, device="cuda")
        y = torch.randn((n,), dtype=dtype, device="cuda")

    lda = max(1, m)
    incx = 1
    incy = 1

    ref_out = flagbench.baseline.zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y.clone(), incy)
    act_out = flagbench.triton.zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y.clone(), incy)

    assert_close(act_out, ref_out, dtype)