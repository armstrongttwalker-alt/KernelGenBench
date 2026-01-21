import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ddot")
@parametrize("n", [
    1, 32,                 # Edge/minimum and small
    15, 160, 1024,         # Non-symmetric inspired sizes
    495, 5333, 71,         # More non-symmetric
    33, 273, 4113,         # Non-aligned
    16, 256, 4096,         # Power-of-2 aligned
    497                    # Additional real-world size
])
@parametrize("incx, incy", [
    (1, 1),                # Contiguous
    (2, 1),                # Strided x
    (1, 3),                # Strided y
    (3, 2),                # Both strided
    (4, 5),                # Larger strides
])
@parametrize("dtype", [torch.float64])
def test_ddot_cublas_baseline(n, incx, incy, dtype):
    base_len_x = (n - 1) * incx + 1
    base_len_y = (n - 1) * incy + 1

    base_x = torch.randn(base_len_x, dtype=dtype, device='cuda')
    base_y = torch.randn(base_len_y, dtype=dtype, device='cuda')

    x = base_x[::incx]
    y = base_y[::incy]

    result = torch.empty(1, dtype=dtype, device='cuda')

    ref_out = flagbench.baseline.ddot(n, x, incx, y, incy, result)
    act_out = flagbench.triton.ddot(n, x, incx, y, incy, result)

    assert_close(act_out, ref_out, dtype)