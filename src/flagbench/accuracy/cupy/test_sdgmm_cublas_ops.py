import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sdgmm")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (16, 256),
    (33, 273),
    (15, 160),
    (160, 1024),
    (5333, 497),
    (495, 5333),
    (16, 4096),
    (273, 4113),
])
@parametrize("mode", [0, 1])
@parametrize("dtype", [torch.float32])
def test_sdgmm_cublas_baseline(m, n, mode, dtype):
    A = torch.randn((m, n), dtype=dtype, device="cuda")
    x_len = m if mode == 0 else n
    x = torch.randn((x_len,), dtype=dtype, device="cuda")
    C = torch.empty((m, n), dtype=dtype, device="cuda")
    lda = m
    ldc = m
    incx = 1

    ref_out = flagbench.baseline.sdgmm(mode, m, n, A, lda, x, incx, C, ldc)
    act_out = flagbench.triton.sdgmm(mode, m, n, A, lda, x, incx, C, ldc)

    assert_close(act_out, ref_out, dtype)