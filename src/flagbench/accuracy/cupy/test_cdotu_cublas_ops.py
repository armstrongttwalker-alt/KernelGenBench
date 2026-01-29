import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cdotu")
@parametrize("n", [
    32,          # Edge case
    1024,        # Non-symmetric (real-world pattern)
    71,          # Non-symmetric
    4113,        # Non-aligned size
    4096,        # Power-of-2 aligned
    497,         # 2D operation-derived
])
@parametrize("dtype", [torch.complex64])
def test_cdotu_cublas_baseline(n, dtype):
    x = torch.randn(n, dtype=dtype, device="cuda")
    y = torch.randn(n, dtype=dtype, device="cuda")
    incx = 1
    incy = 1
    result = torch.empty(1, dtype=dtype, device="cuda")

    ref_out = flagbench.baseline.cdotu(n, x, incx, y, incy, result)
    act_out = flagbench.triton.cdotu(n, x, incx, y, incy, result)

    assert_close(act_out, ref_out, dtype)