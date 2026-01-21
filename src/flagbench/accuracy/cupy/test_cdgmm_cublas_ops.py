import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cdgmm")
@parametrize("M, N, K", [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (33, 273, 4113),
    (16, 256, 4096),
    (1, 32, 64),
    (160, 1024, 2048),
    (5333, 497, 6000),
])
@parametrize("mode", [0, 1])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.complex64])
def test_cdgmm_cublas_baseline(M, N, K, mode, incx, dtype):
    def rand_complex(shape, dtype, device):
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return real + 1j * imag

    A = rand_complex((M, N), dtype, 'cuda')
    C = rand_complex((M, N), dtype, 'cuda')

    L = M if mode == 0 else N
    S = 1 + (L - 1) * incx
    x = rand_complex((S,), dtype, 'cuda')

    lda = max(1, M)
    ldc = max(1, M)

    ref_out = flagbench.baseline.cdgmm(mode, M, N, A, lda, x, incx, C, ldc)
    act_out = flagbench.triton.cdgmm(mode, M, N, A, lda, x, incx, C, ldc)

    assert_close(act_out, ref_out, dtype)