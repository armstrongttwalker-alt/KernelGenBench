import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dnrm2")
@parametrize("n", [
    1,
    32,
    15, 160, 1024,
    495, 5333, 71,
    33, 273, 4113,
    16, 256, 4096,
    497,
])
@parametrize("incx", [1, 2])
@parametrize("dtype", [torch.float64])
def test_dnrm2_cublas_baseline(n, incx, dtype):
    base_len = (n - 1) * incx + 1
    base = torch.randn(base_len, dtype=dtype, device='cuda')
    x = base[0:incx * n:incx]
    result = torch.empty(1, dtype=dtype, device='cuda')

    ref_out = flagbench.baseline.dnrm2(n, x, incx, result)
    act_out = flagbench.triton.dnrm2(n, x, incx, result)

    assert_close(act_out, ref_out, dtype)