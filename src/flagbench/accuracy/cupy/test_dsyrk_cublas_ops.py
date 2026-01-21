import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dsyrk")
@parametrize("n, k", [
    (1, 32),
    (160, 1024),
    (5333, 71),
    (273, 4113),
    (256, 4096),
    (5333, 497),
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
@parametrize("column_major", [True, False])
@parametrize("dtype", [torch.float64])
def test_dsyrk_cublas_baseline(n, k, alpha, beta, trans, uplo, column_major, dtype):
    A_shape = (k, n) if trans else (n, k)
    if column_major:
        if trans:
            A_tmp = torch.randn((n, k), dtype=dtype, device="cuda").contiguous()
            A = A_tmp.t()
        else:
            A_tmp = torch.randn((k, n), dtype=dtype, device="cuda").contiguous()
            A = A_tmp.t()
    else:
        A = torch.randn(A_shape, dtype=dtype, device="cuda").contiguous()
    C0 = torch.randn((n, n), dtype=dtype, device="cuda").contiguous()
    C_ref = C0.clone()
    C_act = C0.clone()
    if trans:
        lda = k if column_major else n
    else:
        lda = n if column_major else k
    ldc = n
    ref_out = flagbench.baseline.dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C_ref, ldc)
    act_out = flagbench.triton.dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C_act, ldc)
    assert_close(act_out, ref_out, dtype, reduce_dim=k)