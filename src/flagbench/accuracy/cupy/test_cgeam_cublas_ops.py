import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgeam")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (160, 1024),
    (33, 273),
    (5333, 497),
    (256, 4096),
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
@parametrize("dtype", [torch.complex64])
def test_cgeam_cublas_baseline(m, n, alpha, beta, transa, transb, column_major, strided, dtype):
    def make_tensor(rows, cols, dtype, column_major, strided):
        if strided:
            big_r = rows * 2 if rows > 1 else 2
            big_c = cols * 2 if cols > 1 else 2
            if column_major:
                base = torch.randn((big_c, big_r), dtype=dtype, device='cuda')
                mat = base.t()
            else:
                mat = torch.randn((big_r, big_c), dtype=dtype, device='cuda')
            return mat[0:big_r:2, 0:big_c:2]
        else:
            if column_major:
                base = torch.randn((cols, rows), dtype=dtype, device='cuda')
                return base.t()
            else:
                return torch.randn((rows, cols), dtype=dtype, device='cuda')

    def shape_for(trans, m, n):
        return (m, n) if trans in ("N", "n") else (n, m)

    def leading_dim(rows, cols, column_major):
        return max(1, rows) if column_major else max(1, cols)

    A_rows, A_cols = shape_for(transa, m, n)
    B_rows, B_cols = shape_for(transb, m, n)

    A = make_tensor(A_rows, A_cols, dtype, column_major, strided)
    B = make_tensor(B_rows, B_cols, dtype, column_major, strided)
    C = make_tensor(m, n, dtype, column_major, strided)

    lda = leading_dim(A.shape[0], A.shape[1], column_major)
    ldb = leading_dim(B.shape[0], B.shape[1], column_major)
    ldc = leading_dim(C.shape[0], C.shape[1], column_major)

    ref_out = flagbench.baseline.cgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    act_out = flagbench.triton.cgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)

    assert_close(act_out, ref_out, dtype)