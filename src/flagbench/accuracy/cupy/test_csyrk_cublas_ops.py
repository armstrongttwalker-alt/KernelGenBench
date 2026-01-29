import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("csyrk")
@parametrize("n, k", [
    (1, 32),        # Edge case
    (15, 1024),     # Non-symmetric derived from (15, 160, 1024)
    (160, 1024),    # 2D pattern
    (495, 71),      # Non-symmetric derived from (495, 5333, 71)
    (273, 4113),    # Non-aligned
    (256, 4096),    # Power-of-2 aligned
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (-0.5, -0.5),
])
@parametrize("uplo", [0, 1])  # 0: upper, 1: lower
@parametrize("trans", [False, True])  # False: 'N', True: 'T'
@parametrize("dtype", [torch.complex64])
def test_csyrk_cublas_baseline(n, k, alpha, beta, uplo, trans, dtype):
    A_shape = (k, n) if trans else (n, k)

    real_A = torch.randn(A_shape, dtype=torch.float32, device="cuda")
    imag_A = torch.randn(A_shape, dtype=torch.float32, device="cuda")
    A = torch.complex(real_A, imag_A).to(dtype)

    real_C = torch.randn((n, n), dtype=torch.float32, device="cuda")
    imag_C = torch.randn((n, n), dtype=torch.float32, device="cuda")
    C_init = torch.complex(real_C, imag_C).to(dtype)

    C_ref = C_init.clone()
    C_act = C_init.clone()

    lda = A.shape[0]
    ldc = n

    alpha_c = complex(alpha)
    beta_c = complex(beta)

    ref_out = flagbench.baseline.csyrk(uplo, trans, n, k, alpha_c, A, lda, beta_c, C_ref, ldc)
    act_out = flagbench.triton.csyrk(uplo, trans, n, k, alpha_c, A, lda, beta_c, C_act, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=k)