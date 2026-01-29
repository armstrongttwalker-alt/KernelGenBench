import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zdotu")
@parametrize("n", [
    1,
    32,
    71,
    160,
    273,
    497,
    1024,
    4096,
    4113,
    5333,
])
@parametrize("stride_x, stride_y", [
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 2),
    (2, 3),
])
@parametrize("offset_x, offset_y", [
    (0, 0),
    (1, 0),
])
@parametrize("dtype", [torch.complex128])
def test_zdotu_cublas_baseline(n, stride_x, stride_y, offset_x, offset_y, dtype):
    # Allocate base buffers to accommodate strides and offsets
    base_len_x = offset_x + (n - 1) * stride_x + 1
    base_len_y = offset_y + (n - 1) * stride_y + 1

    rx = torch.randn(base_len_x, dtype=torch.float64, device="cuda")
    ix = torch.randn(base_len_x, dtype=torch.float64, device="cuda")
    x_base = rx + 1j * ix

    ry = torch.randn(base_len_y, dtype=torch.float64, device="cuda")
    iy = torch.randn(base_len_y, dtype=torch.float64, device="cuda")
    y_base = ry + 1j * iy

    # Create possibly strided 1D views of length n
    x = x_base[offset_x::stride_x][:n]
    y = y_base[offset_y::stride_y][:n]

    incx = stride_x
    incy = stride_y

    result = torch.empty(1, dtype=dtype, device="cuda")

    ref_out = flagbench.baseline.zdotu(n, x, incx, y, incy, result)
    act_out = flagbench.triton.zdotu(n, x, incx, y, incy, result)

    assert_close(act_out, ref_out, dtype)