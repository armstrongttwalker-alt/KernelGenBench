import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dgemm")
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
])
@parametrize("layout_kind", ["contig", "col_major", "row_padded"])
@parametrize("dtype", [torch.float64])
def test_dgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, layout_kind, dtype):
    def make_tensor(shape):
        if layout_kind == "contig":
            return torch.randn(shape, dtype=dtype, device="cuda")
        elif layout_kind == "col_major":
            base = torch.randn((shape[1], shape[0]), dtype=dtype, device="cuda")
            return base.t()
        else:
            m, n = shape
            pad = max(1, n // 10 + 1)
            storage = torch.randn((m, n + pad), dtype=dtype, device="cuda")
            return torch.as_strided(storage, size=(m, n), stride=(storage.stride(0), 1), storage_offset=0)

    A_shape = (M, K) if transa == "N" else (K, M)
    B_shape = (K, N) if transb == "N" else (N, K)
    C_shape = (M, N)

    A = make_tensor(A_shape)
    B = make_tensor(B_shape)
    C = make_tensor(C_shape)

    lda = A.stride(0)
    ldb = B.stride(0)
    ldc = C.stride(0)

    C_ref = C.clone()
    C_act = C.clone()

    ref_out = flagbench.baseline.dgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc)
    act_out = flagbench.triton.dgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_act, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)