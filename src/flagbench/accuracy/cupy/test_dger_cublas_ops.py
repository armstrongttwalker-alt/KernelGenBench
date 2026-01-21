import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dger")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (160, 1024),
    (495, 5333),
    (33, 273),
    (16, 256),
    (5333, 497),
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
@parametrize("dtype", [torch.float64])
def test_dger_cublas_baseline(m, n, alpha, dtype):
    incx = 1
    incy = 1
    lda = m

    x = torch.randn((m,), dtype=dtype, device='cuda')
    y = torch.randn((n,), dtype=dtype, device='cuda')
    A = torch.randn((m, n), dtype=dtype, device='cuda')

    A_baseline = A.clone()
    A_triton = A.clone()

    ref_out = flagbench.baseline.dger(m, n, alpha, x, incx, y, incy, A_baseline, lda)
    act_out = flagbench.triton.dger(m, n, alpha, x, incx, y, incy, A_triton, lda)

    assert_close(act_out, ref_out, dtype)