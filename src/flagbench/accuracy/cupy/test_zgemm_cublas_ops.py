import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgemm")
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
    ("N", "H"),
    ("T", "N"),
    ("T", "T"),
    ("T", "H"),
    ("H", "N"),
    ("H", "T"),
    ("H", "H"),
])
@parametrize("column_major", [True, False])
@parametrize("strided", [True, False])
@parametrize("dtype", [torch.complex128])
def test_zgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, column_major, strided, dtype):
    def make_matrix(rows, cols, column_major, strided, dtype):
        pad = 8 if strided else 0
        if column_major:
            base = torch.randn((cols, rows + pad), dtype=dtype, device='cuda')
            view = base.transpose(0, 1)[:rows, :cols]
            ld = view.stride(1)
        else:
            base = torch.randn((rows, cols + pad), dtype=dtype, device='cuda')
            view = base[:, :cols]
            ld = view.stride(0)
        return view, int(ld)

    a_rows, a_cols = (M, K) if transa == 'N' else (K, M)
    b_rows, b_cols = (K, N) if transb == 'N' else (N, K)

    A, lda = make_matrix(a_rows, a_cols, column_major, strided, dtype)
    B, ldb = make_matrix(b_rows, b_cols, column_major, strided, dtype)
    C, ldc = make_matrix(M, N, column_major, strided, dtype)

    alpha_t = torch.tensor(complex(alpha, 0.0), dtype=torch.complex128, device='cuda')
    beta_t = torch.tensor(complex(beta, 0.0), dtype=torch.complex128, device='cuda')

    ref_out = flagbench.baseline.zgemm(transa, transb, M, N, K, alpha_t, A, lda, B, ldb, beta_t, C, ldc)
    act_out = flagbench.triton.zgemm(transa, transb, M, N, K, alpha_t, A, lda, B, ldb, beta_t, C, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)