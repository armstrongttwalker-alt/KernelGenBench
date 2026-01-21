import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgerc")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (495, 5333),
    (33, 273),
    (16, 256),
    (160, 1024),
    (5333, 497),
])
@parametrize("alpha", [
    1.0,
    0.0,
    0.001,
    100.001,
    -0.5,
    -111.999,
    0.5,
    -0.999,
    1j,
    -1j,
    0.5 + 0.5j,
    -0.5 - 0.5j,
])
@parametrize("dtype", [torch.complex128])
def test_zgerc_cublas_baseline(m, n, alpha, dtype):
    incx = 1
    incy = 1
    lda = max(1, m)

    xr = torch.randn(m, dtype=torch.float64, device="cuda")
    xi = torch.randn(m, dtype=torch.float64, device="cuda")
    x = xr + 1j * xi

    yr = torch.randn(n, dtype=torch.float64, device="cuda")
    yi = torch.randn(n, dtype=torch.float64, device="cuda")
    y = yr + 1j * yi

    Ar = torch.randn((m, n), dtype=torch.float64, device="cuda")
    Ai = torch.randn((m, n), dtype=torch.float64, device="cuda")
    A = Ar + 1j * Ai

    A_ref = A.clone()
    A_act = A.clone()

    ref_out = flagbench.baseline.zgerc(m, n, alpha, x, incx, y, incy, A_ref, lda)
    act_out = flagbench.triton.zgerc(m, n, alpha, x, incx, y, incy, A_act, lda)

    assert_close(act_out, ref_out, dtype)