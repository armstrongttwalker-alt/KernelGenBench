import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dsbmv")
@parametrize("n, k", [
    (1, 0),            # Edge case
    (160, 15),         # Non-symmetric
    (5333, 71),        # Non-symmetric (large n)
    (273, 33),         # Non-aligned
    (256, 16),         # Power-of-2 aligned
    (32, 1),           # Small
    (1024, 160),       # Wider band
    (5333, 497),       # Large with significant bandwidth
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("uplo", ["U", "L"])
@parametrize("incx, incy", [
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 3),
])
@parametrize("dtype", [torch.float64])
def test_dsbmv_cublas_baseline(n, k, alpha, beta, uplo, incx, incy, dtype):
    k_eff = min(k, n - 1)
    pad = 0 if (n % 2 == 0) else 2
    lda = k_eff + 1 + pad

    A = torch.randn((lda, n), dtype=dtype, device='cuda')

    x_full = torch.randn(1 + (n - 1) * incx, dtype=dtype, device='cuda')
    x = x_full[::incx]

    y_full_ref = torch.randn(1 + (n - 1) * incy, dtype=dtype, device='cuda')
    y_ref = y_full_ref[::incy]
    y_full_act = y_full_ref.clone()
    y_act = y_full_act[::incy]

    ref_out = flagbench.baseline.dsbmv(uplo, n, k_eff, alpha, A, lda, x, incx, beta, y_ref, incy)
    act_out = flagbench.triton.dsbmv(uplo, n, k_eff, alpha, A, lda, x, incx, beta, y_act, incy)

    assert_close(act_out, ref_out, dtype)