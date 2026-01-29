import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("sgeam")
@parametrize("M, N", [
    (1, 1),
    (1, 32),
    (15, 160),
    (495, 5333),
    (33, 273),
    (16, 256),
    (256, 4096),
    (273, 4113),
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
@parametrize("transa, transb", [
    ("N", "N"),
    ("N", "T"),
    ("T", "N"),
    ("T", "T"),
])
@parametrize("column_major", [True, False])
@parametrize("dtype", [torch.float32])
def test_sgeam_cublas_baseline(M, N, alpha, beta, transa, transb, column_major, dtype):
    A_shape = (M, N) if transa == "N" else (N, M)
    B_shape = (M, N) if transb == "N" else (N, M)

    A = torch.randn(A_shape, dtype=dtype, device="cuda")
    B = torch.randn(B_shape, dtype=dtype, device="cuda")
    C = torch.randn((M, N), dtype=dtype, device="cuda")

    lda = A.shape[0] if column_major else A.shape[1]
    ldb = B.shape[0] if column_major else B.shape[1]
    ldc = C.shape[0] if column_major else C.shape[1]

    ref_out = flagbench.baseline.sgeam(transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc)
    act_out = flagbench.triton.sgeam(transa, transb, M, N, alpha, A, lda, beta, B, ldb, C, ldc)

    assert_close(act_out, ref_out, dtype)