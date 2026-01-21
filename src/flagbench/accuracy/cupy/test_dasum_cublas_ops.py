import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dasum")
@parametrize("n", [
    32,
    71,
    1024,
    4096,
    4113,
    497,
])
@parametrize("incx", [
    1,
    2,
    3,
    8,
])
@parametrize("dtype", [torch.float64])
def test_dasum_cublas_baseline(n, incx, dtype):
    base_len = n * incx
    base = torch.randn(base_len, dtype=dtype, device="cuda")
    x = base[::incx]
    result = torch.empty(1, dtype=dtype, device="cuda")

    ref_out = flagbench.baseline.dasum(n, x, incx, result)
    act_out = flagbench.triton.dasum(n, x, incx, result)

    assert_close(act_out, ref_out, dtype)