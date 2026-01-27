import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("caxpy")
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
@parametrize("alpha", [
    1.0 + 0.0j,
    0.0 + 1.0j,
    0.001 - 0.999j,
    100.001 - 111.999j,
    0.5 + 0.5j,
    -0.5 - 0.5j,
])
@parametrize("dtype", [torch.complex64])
def test_caxpy_cublas_baseline(n, alpha, dtype):
    x = (torch.randn(n, dtype=torch.float32, device='cuda') + 1j * torch.randn(n, dtype=torch.float32, device='cuda')).to(dtype)
    y = (torch.randn(n, dtype=torch.float32, device='cuda') + 1j * torch.randn(n, dtype=torch.float32, device='cuda')).to(dtype)

    y_ref = y.clone()
    y_act = y.clone()
    incx = 1
    incy = 1

    ref_out = flagbench.baseline.caxpy(n, alpha, x, incx, y_ref, incy)
    act_out = flagbench.triton.caxpy(n, alpha, x, incx, y_act, incy)

    assert_close(act_out, ref_out, dtype)