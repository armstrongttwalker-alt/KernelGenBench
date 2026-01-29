import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cscal")
@parametrize("n", [
    32,          # Edge case
    71,          # Non-symmetric
    497,         # GEMV-like
    1024,        # Non-symmetric / GEMV-like
    4096,        # Power-of-2 aligned
    4113,        # Non-aligned
])
@parametrize("alpha", [
    complex(1.0, 0.0),           # Standard
    complex(0.0, 1.0),           # Standard imaginary
    complex(0.001, -0.999),      # Fractional/small
    complex(100.001, -111.999),  # Large/negative imaginary
    complex(0.5, 0.5),           # Symmetric
    complex(-0.5, -0.5),         # Negative symmetric
])
@parametrize("incx", [1, 2, 3, 5])
@parametrize("dtype", [torch.complex64])
def test_cscal_cublas_baseline(n, alpha, incx, dtype):
    length = (n - 1) * incx + 1
    base = torch.randn(length, dtype=dtype, device="cuda")
    x_baseline = base[::incx]
    x_triton = base.clone()[::incx]

    ref_out = flagbench.baseline.cscal(n, alpha, x_baseline, incx)
    act_out = flagbench.triton.cscal(n, alpha, x_triton, incx)

    assert_close(act_out, ref_out, dtype)