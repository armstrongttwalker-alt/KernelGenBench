import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgeru")
@parametrize("m, n", [
    (1, 1),
    (15, 160),
    (495, 5333),
    (33, 273),
    (16, 256),
    (1, 32),
    (160, 1024),
    (5333, 497),
])
@parametrize("alpha", [
    (1 + 0j),
    (0 + 0j),
    (0.001 - 0.999j),
    (100.001 - 111.999j),
    (0.5 + 0.5j),
    (-0.5 - 0.5j),
])
@parametrize("dtype", [torch.complex128])
def test_zgeru_cublas_baseline(m, n, alpha, dtype):
    incx = 1
    incy = 1
    lda = m

    x = torch.randn(m, dtype=dtype, device="cuda")
    y = torch.randn(n, dtype=dtype, device="cuda")
    A_init = torch.randn((m, n), dtype=dtype, device="cuda")

    A_baseline = A_init.clone()
    A_triton = A_init.clone()

    ref_out = flagbench.baseline.zgeru(m, n, alpha, x, incx, y, incy, A_baseline, lda)
    act_out = flagbench.triton.zgeru(m, n, alpha, x, incx, y, incy, A_triton, lda)

    assert_close(act_out, ref_out, dtype)