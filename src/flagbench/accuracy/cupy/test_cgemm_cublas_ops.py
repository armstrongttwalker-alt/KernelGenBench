import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgemm")
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
    ("N", "H"),
    ("T", "N"),
    ("T", "T"),
    ("T", "H"),
    ("H", "N"),
    ("H", "T"),
    ("H", "H"),
])
@parametrize("column_major", [True, False])
@parametrize("dtype", [torch.complex64])
def test_cgemm_cublas_baseline(M, N, K, alpha, beta, transa, transb, column_major, dtype):
    A_shape = (K, M) if transa in ("T", "H") else (M, K)
    B_shape = (N, K) if transb in ("T", "H") else (K, N)
    C_shape = (M, N)

    def rand_complex(shape):
        real = torch.randn(shape, dtype=torch.float32, device='cuda')
        imag = torch.randn(shape, dtype=torch.float32, device='cuda')
        return torch.complex(real, imag).to(dtype)

    def make_tensor(shape, is_col_major):
        pad = 3
        if is_col_major:
            base = rand_complex((shape[1], shape[0] + pad))
            view = base.transpose(0, 1)[:shape[0], :shape[1]]
            return view
        else:
            base = rand_complex((shape[0], shape[1] + pad))
            view = base[:, :shape[1]]
            return view

    A = make_tensor(A_shape, column_major)
    B = make_tensor(B_shape, column_major)
    C = make_tensor(C_shape, column_major)

    if column_major:
        lda = A.stride(1)
        ldb = B.stride(1)
        ldc = C.stride(1)
    else:
        lda = A.stride(0)
        ldb = B.stride(0)
        ldc = C.stride(0)

    alpha_c = complex(alpha, 0.0)
    beta_c = complex(beta, 0.0)

    ref_out = flagbench.baseline.cgemm(transa, transb, M, N, K, alpha_c, A, lda, B, ldb, beta_c, C, ldc)
    act_out = flagbench.triton.cgemm(transa, transb, M, N, K, alpha_c, A, lda, B, ldb, beta_c, C, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)