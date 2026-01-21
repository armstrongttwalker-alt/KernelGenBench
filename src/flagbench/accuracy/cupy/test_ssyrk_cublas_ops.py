import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("ssyrk")
@parametrize("n, k", [
    (1, 32),
    (15, 1024),
    (160, 1024),
    (495, 71),
    (33, 4113),
    (273, 4113),
    (16, 4096),
    (497, 5333),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("trans", [False, True])
@parametrize("uplo", [0, 1])
@parametrize("dtype", [torch.float32])
def test_ssyrk_cublas_baseline(n, k, alpha, beta, trans, uplo, dtype):
    A_shape = (k, n) if trans else (n, k)
    A = torch.randn(A_shape, dtype=dtype, device='cuda')
    misl = torch.randn((n, n), dtype=dtype, device='cuda')
    C_baseline = misl.clone()
    C_triton = misl.clone()

    lda = A.shape[0]
    ldc = n

    ref_out = flagbench.baseline.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C_baseline, ldc)
    act_out = flagbench.triton.ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C_triton, ldc)

    assert_close(act_out, ref_out, dtype)