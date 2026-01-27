import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ssbmv")
@parametrize("n, k", [
    (160, 15),
    (5333, 71),
    (273, 33),
    (256, 16),
    (32, 1),
    (1024, 160),
    (5333, 497),
    (4113, 273),
])
@parametrize("uplo", ["U", "L"])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("dtype", [torch.float32])
def test_ssbmv_cublas_baseline(n, k, uplo, alpha, beta, dtype):
    lda = k + 1
    incx = 1
    incy = 1

    A = torch.randn((lda, n), dtype=dtype, device="cuda")
    x = torch.randn((n,), dtype=dtype, device="cuda")
    y0 = torch.randn((n,), dtype=dtype, device="cuda")

    y_ref = y0.clone()
    y_act = y0.clone()

    ref_out = flagbench.baseline.ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y_ref, incy)
    act_out = flagbench.triton.ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y_act, incy)

    assert_close(act_out, ref_out, dtype)