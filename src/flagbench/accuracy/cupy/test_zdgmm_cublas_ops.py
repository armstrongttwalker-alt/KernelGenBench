import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zdgmm")
@parametrize("m, n", [
    (1, 1),
    (15, 160),
    (495, 5333),
    (33, 273),
    (16, 256),
    (33, 4113),
    (1, 32),
    (160, 1024),
    (5333, 497),
])
@parametrize("mode", [0, 1])
@parametrize("strided", [False, True])
@parametrize("dtype", [torch.complex128])
def test_zdgmm_cublas_baseline(m, n, mode, strided, dtype):
    if strided:
        A_base_real = torch.randn((m * 2, n * 3), dtype=torch.float64, device='cuda')
        A_base_imag = torch.randn((m * 2, n * 3), dtype=torch.float64, device='cuda')
        A = (A_base_real + 1j * A_base_imag)[::2, ::3]
        C_base_real = torch.randn((m * 2, n * 3), dtype=torch.float64, device='cuda')
        C_base_imag = torch.randn((m * 2, n * 3), dtype=torch.float64, device='cuda')
        C = (C_base_real + 1j * C_base_imag)[::2, ::3]
    else:
        A_real = torch.randn((m, n), dtype=torch.float64, device='cuda')
        A_imag = torch.randn((m, n), dtype=torch.float64, device='cuda')
        A = A_real + 1j * A_imag
        C_real = torch.randn((m, n), dtype=torch.float64, device='cuda')
        C_imag = torch.randn((m, n), dtype=torch.float64, device='cuda')
        C = C_real + 1j * C_imag

    x_len = m if mode == 0 else n
    x_real = torch.randn((x_len,), dtype=torch.float64, device='cuda')
    x_imag = torch.randn((x_len,), dtype=torch.float64, device='cuda')
    x = x_real + 1j * x_imag

    lda = A.stride(0)
    incx = 1
    ldc = C.stride(0)

    ref_out = flagbench.baseline.zdgmm(mode, m, n, A, lda, x, incx, C, ldc)
    act_out = flagbench.triton.zdgmm(mode, m, n, A, lda, x, incx, C, ldc)

    assert_close(act_out, ref_out, dtype)