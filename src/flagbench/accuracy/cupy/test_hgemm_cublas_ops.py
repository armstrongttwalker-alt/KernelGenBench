import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("hgemm")
@parametrize("M, N, K", [
    (1, 1, 32),
    (15, 160, 1024),
    (495, 5333, 71),
    (33, 273, 4113),
    (16, 256, 4096),
    (160, 1, 1024),
    (5333, 1, 497),
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
    ("C", "N"),
    ("N", "C"),
    ("C", "C"),
])
@parametrize("column_major", [False, True])
@parametrize("dtype", [torch.float16])
def test_hgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, column_major, dtype):
    ta = transa.upper()
    tb = transb.upper()
    A_shape = (K, M) if ta in ("T", "C") else (M, K)
    B_shape = (N, K) if tb in ("T", "C") else (K, N)

    def make_tensor(shape, dtype, device, column_major):
        if column_major:
            base = torch.randn((shape[1], shape[0]), dtype=dtype, device=device).contiguous()
            return base.t()
        else:
            return torch.randn(shape, dtype=dtype, device=device).contiguous()

    A = make_tensor(A_shape, dtype, "cuda", column_major)
    B = make_tensor(B_shape, dtype, "cuda", column_major)
    C = make_tensor((M, N), dtype, "cuda", column_major)

    lda = A.shape[0]
    ldb = B.shape[0]
    ldc = C.shape[0]

    ref_out = flagbench.baseline.hgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    act_out = flagbench.triton.hgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)