import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ddgmm")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (495, 5333),
    (33, 273),
    (16, 256),
    (160, 1024),
    (5333, 497),
])
@parametrize("mode", [0, 1])
@parametrize("contig", [True, False])
@parametrize("dtype", [torch.float64])
def test_ddgmm_cublas_baseline(m, n, mode, contig, dtype):
    device = "cuda"
    if contig:
        A = torch.randn((m, n), dtype=dtype, device=device).contiguous()
    else:
        A = torch.randn((n, m), dtype=dtype, device=device).t()
    x_len = m if mode == 0 else n
    if contig:
        x = torch.randn((x_len,), dtype=dtype, device=device).contiguous()
    else:
        x_big = torch.randn((x_len * 2,), dtype=dtype, device=device).contiguous()
        x = x_big[::2]
    incx = x.stride(0)
    if contig:
        C = torch.empty((m, n), dtype=dtype, device=device).contiguous()
    else:
        C = torch.empty((n, m), dtype=dtype, device=device).t()
    lda = max(1, A.stride(0))
    ldc = max(1, C.stride(0))
    ref_out = flagbench.baseline.ddgmm(mode, m, n, A, lda, x, incx, C, ldc)
    act_out = flagbench.triton.ddgmm(mode, m, n, A, lda, x, incx, C, ldc)
    assert_close(act_out, ref_out, dtype)