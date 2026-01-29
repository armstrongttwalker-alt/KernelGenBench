import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("zgeam")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (15, 160),
    (160, 1024),
    (5333, 497),
    (33, 273),
    (16, 256),
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
@parametrize("dtype", [torch.complex128])
def test_zgeam_cublas_baseline(m, n, alpha, beta, transa, transb, column_major, dtype):
    def rand_complex(shape):
        real = torch.randn(shape, dtype=torch.float64, device='cuda')
        imag = torch.randn(shape, dtype=torch.float64, device='cuda')
        return (real + 1j * imag).to(dtype)

    # A
    if transa == 'N':
        if column_major:
            A = rand_complex((n, m)).t()
            lda = m
        else:
            A = rand_complex((m, n)).contiguous()
            lda = n
    else:
        if column_major:
            A = rand_complex((m, n)).t()
            lda = n
        else:
            A = rand_complex((n, m)).contiguous()
            lda = m

    # B
    if transb == 'N':
        if column_major:
            B = rand_complex((n, m)).t()
            ldb = m
        else:
            B = rand_complex((m, n)).contiguous()
            ldb = n
    else:
        if column_major:
            B = rand_complex((m, n)).t()
            ldb = n
        else:
            B = rand_complex((n, m)).contiguous()
            ldb = m

    # C
    if column_major:
        C = rand_complex((n, m)).t()
        ldc = m
    else:
        C = rand_complex((m, n)).contiguous()
        ldc = n

    alpha = complex(alpha)
    beta = complex(beta)

    ref_out = flagbench.baseline.zgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    act_out = flagbench.triton.zgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)

    assert_close(act_out, ref_out, dtype)