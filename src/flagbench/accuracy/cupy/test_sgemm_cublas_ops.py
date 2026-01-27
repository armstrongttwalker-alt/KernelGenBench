import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgemm")
@parametrize("M, N, K", [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (33, 273, 4113),
    (16, 256, 4096),
    (1, 32, 1),
    (160, 1024, 1),
    (5333, 497, 1),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.001, -0.999),
    (100.001, -111.999),
    (0.5, 0.5),
    (-0.5, -0.5),
])
@parametrize("transa, transb", [
    ("N", "N"),
    ("N", "T"),
    ("T", "N"),
    ("T", "T"),
])
@parametrize("column_major", [True, False])
@parametrize("make_strided", [False, True])
@parametrize("dtype", [torch.float32])
def test_sgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, column_major, make_strided, dtype):
    A_shape = (M, K) if transa == "N" else (K, M)
    B_shape = (K, N) if transb == "N" else (N, K)

    def make_tensor(shape):
        t = torch.randn(shape, dtype=dtype, device="cuda")
        if make_strided:
            rows, cols = shape
            storage = torch.randn(rows, cols * 2, dtype=dtype, device="cuda")
            t = torch.as_strided(storage, size=(rows, cols), stride=(cols * 2, 2))
        return t

    A = make_tensor(A_shape)
    B = make_tensor(B_shape)
    C_init = make_tensor((M, N))

    lda = A_shape[0] if column_major else A_shape[1]
    ldb = B_shape[0] if column_major else B_shape[1]
    ldc = M if column_major else N

    C_ref = C_init.clone()
    ref_out = flagbench.baseline.sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc)

    C_act = C_init.clone()
    act_out = flagbench.triton.sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_act, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)