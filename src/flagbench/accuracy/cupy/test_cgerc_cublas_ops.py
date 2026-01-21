import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgerc")
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
    (2, 3),
    (4, 5),
])
@parametrize("column_major", [True, False])
@parametrize("dtype", [torch.complex64])
def test_cgerc_cublas_baseline(m, n, alpha, incx, incy, column_major, dtype):
    # Prepare strided input vectors x and y
    Lx = 1 + (m - 1) * incx
    Ly = 1 + (n - 1) * incy
    x_base = torch.randn(Lx, dtype=dtype, device='cuda')
    y_base = torch.randn(Ly, dtype=dtype, device='cuda')
    x = x_base[::incx]
    y = y_base[::incy]

    # Prepare matrix A with chosen memory layout
    if column_major:
        A = torch.randn((n, m), dtype=dtype, device='cuda').t()
    else:
        A = torch.randn((m, n), dtype=dtype, device='cuda')

    # Compute leading dimension consistent with layout
    lda = A.stride(1) if column_major else A.stride(0)

    alpha_c = complex(alpha, 0.0)

    A_baseline = A.clone()
    A_triton = A.clone()

    ref_out = flagbench.baseline.cgerc(m, n, alpha_c, x, incx, y, incy, A_baseline, lda)
    act_out = flagbench.triton.cgerc(m, n, alpha_c, x, incx, y, incy, A_triton, lda)

    assert_close(act_out, ref_out, dtype)