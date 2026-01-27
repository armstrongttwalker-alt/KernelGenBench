import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zsyrk")
@parametrize("n, k", [
    (1, 32),
    (15, 1024),
    (495, 71),
    (33, 4113),
    (256, 4096),
    (160, 1024),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("uplo", [0, 1])
@parametrize("trans", [False, True])
@parametrize("strided", [False, True])
@parametrize("dtype", [torch.complex128])
def test_zsyrk_cublas_baseline(n, k, alpha, beta, uplo, trans, strided, dtype):
    A_shape = (k, n) if trans else (n, k)
    C_shape = (n, n)

    def make_complex(shape):
        return torch.randn(shape, dtype=torch.float64, device='cuda') + 1j * torch.randn(shape, dtype=torch.float64, device='cuda')

    A = make_complex(A_shape).to(dtype)
    C_init = make_complex(C_shape).to(dtype)

    if strided:
        A_pad = make_complex((A_shape[0] + 2, A_shape[1] + 3)).to(dtype)
        A = A_pad[1:1 + A_shape[0], 1:1 + A_shape[1]]
        C_pad = make_complex((C_shape[0] + 2, C_shape[1] + 2)).to(dtype)
        C_init = C_pad[1:1 + C_shape[0], 1:1 + C_shape[1]]

    C_baseline = C_init.clone()
    C_triton = C_init.clone()

    lda = A.shape[0]
    ldc = C_init.shape[0]

    alpha_c = complex(alpha, 0.0)
    beta_c = complex(beta, 0.0)

    ref_out = flagbench.baseline.zsyrk(uplo, trans, n, k, alpha_c, A, lda, beta_c, C_baseline, ldc)
    act_out = flagbench.triton.zsyrk(uplo, trans, n, k, alpha_c, A, lda, beta_c, C_triton, ldc)

    assert_close(act_out, ref_out, dtype)