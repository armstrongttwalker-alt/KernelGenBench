import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("saxpy")
@parametrize("n", [
    1,          # Edge case
    32,         # Edge case
    71,         # Non-symmetric component
    160,        # Non-symmetric component
    497,        # 2D op dimension
    1024,       # Non-symmetric component
    4113,       # Non-aligned size
    4096,       # Power-of-2 aligned
    5333,       # Non-symmetric (large N)
])
@parametrize("alpha", [
    1.0,        # Standard
    0.0,        # Standard
    0.001,      # Fractional/small
    -0.999,     # Fractional/negative
    100.001,    # Large
    -111.999,   # Large negative
    0.5,        # Symmetric
    -0.5,       # Symmetric negative
])
@parametrize("incx", [1, 2, 3])
@parametrize("incy", [1, 2, 3])
@parametrize("dtype", [torch.float32])
def test_saxpy_cublas_baseline(n, alpha, incx, incy, dtype):
    # Initialize strided input tensors
    x_storage = torch.randn(n * incx, dtype=dtype, device='cuda')
    y_storage0 = torch.randn(n * incy, dtype=dtype, device='cuda')
    y_storage1 = y_storage0.clone()
    x = x_storage[::incx][:n]
    y0 = y_storage0[::incy][:n]
    y1 = y_storage1[::incy][:n]

    # Call baseline
    ref_out = flagbench.baseline.saxpy(n, alpha, x, incx, y0, incy)

    # Call Triton implementation
    act_out = flagbench.triton.saxpy(n, alpha, x, incx, y1, incy)

    # Compare results
    assert_close(act_out, ref_out, dtype)