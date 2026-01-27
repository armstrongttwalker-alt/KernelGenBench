import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("dgeam")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (495, 5333),
    (33, 273),
    (33, 4113),
    (16, 256),
    (256, 4096),
    (160, 1024),
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
@parametrize("dtype", [torch.float64])
def test_dgeam_cublas_baseline(m, n, alpha, beta, transa, transb, column_major, dtype):
    def make_tensor(shape, column_major):
        if column_major:
            return torch.randn((shape[1], shape[0]), dtype=dtype, device="cuda").transpose(0, 1)
        else:
            return torch.randn(shape, dtype=dtype, device="cuda")

    def orig_shape(m, n, trans):
        return (m, n) if trans == "N" else (n, m)

    A_shape = orig_shape(m, n, transa)
    B_shape = orig_shape(m, n, transb)
    C_shape = (m, n)

    A = make_tensor(A_shape, column_major)
    B = make_tensor(B_shape, column_major)
    C = make_tensor(C_shape, column_major)

    if column_major:
        lda = A.shape[0]
        ldb = B.shape[0]
        ldc = C.shape[0]
    else:
        lda = A.shape[1]
        ldb = B.shape[1]
        ldc = C.shape[1]

    ref_out = flagbench.baseline.dgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    act_out = flagbench.triton.dgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)

    assert_close(act_out, ref_out, dtype)