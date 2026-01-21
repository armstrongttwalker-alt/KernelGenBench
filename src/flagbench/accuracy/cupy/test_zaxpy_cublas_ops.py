import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zaxpy")
@parametrize("n", [
    1,          # Edge case
    32,         # Small aligned
    71,         # Small non-aligned
    160,        # Typical mid-size
    273,        # Non-aligned
    1024,       # Power-of-2
    4096,       # Power-of-2 large
    4113,       # Non-aligned large
    5333,       # Large non-symmetric
])
@parametrize("alpha", [
    complex(1.0, 0.0),
    complex(0.0, 0.0),
    complex(0.001, -0.999),
    complex(100.001, -111.999),
    complex(0.5, 0.5),
    complex(-0.5, -0.5),
])
@parametrize("incx, incy", [
    (1, 1),     # Contiguous
    (2, 1),     # Strided x
    (3, 2),     # Strided x and y
])
@parametrize("dtype", [torch.complex128])
def test_zaxpy_cublas_baseline(n, alpha, incx, incy, dtype):
    len_x = 1 + (n - 1) * abs(incx)
    len_y = 1 + (n - 1) * abs(incy)

    base_x1 = torch.randn(len_x, dtype=dtype, device='cuda')
    base_x2 = base_x1.clone()
    x1 = base_x1[0:len_x:incx]
    x2 = base_x2[0:len_x:incx]

    base_y1 = torch.randn(len_y, dtype=dtype, device='cuda')
    base_y2 = base_y1.clone()
    y1 = base_y1[0:len_y:incy]
    y2 = base_y2[0:len_y:incy]

    ref_out = flagbench.baseline.zaxpy(n, alpha, x1, incx, y1, incy)
    act_out = flagbench.triton.zaxpy(n, alpha, x2, incx, y2, incy)

    assert_close(act_out, ref_out, dtype)