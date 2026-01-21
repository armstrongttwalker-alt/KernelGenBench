import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sscal")
@parametrize("n", [
    1,
    32,
    160,
    1024,
    5333,
    497,
    4096,
    273,
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
@parametrize("dtype", [torch.float32])
def test_sscal_cublas_baseline(n, alpha, dtype):
    x_base = torch.randn((n,), dtype=dtype, device='cuda')
    x_triton = x_base.clone()
    incx = 1

    ref_out = flagbench.baseline.sscal(n, alpha, x_base, incx)
    act_out = flagbench.triton.sscal(n, alpha, x_triton, incx)

    assert_close(act_out, ref_out, dtype)