import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasZgemm_v2")
@parametrize("M_N_K", [(16, 16, 16), (64, 32, 48), (128, 256, 64), (17, 33, 65)])
@parametrize("alpha_beta", [(1.0+0j, 0.0+0j), (0.5+0.5j, 0.5+0j), (-1.0+0j, 1.0+0j)])
@parametrize("transa_transb", [("N", "N"), ("N", "T"), ("T", "N"), ("C", "C")])
def test_accuracy_cublasZgemm_v2(M_N_K, alpha_beta, transa_transb):
    dtype = torch.complex128
    M, N, K = M_N_K
    alpha, beta = alpha_beta
    transa, transb = transa_transb

    A_shape = (K, M) if transa in ['T', 'C'] else (M, K)
    B_shape = (N, K) if transb in ['T', 'C'] else (K, N)
    lda = A_shape[1]
    ldb = B_shape[1]
    ldc = M

    A = torch.randn(A_shape, dtype=dtype, device=device)
    B = torch.randn(B_shape, dtype=dtype, device=device)
    C0 = torch.randn(M, N, dtype=dtype, device=device)
    C1 = C0.clone()

    ref_out = flagbench.baseline.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C0, ldc)
    act_out = flagbench.triton.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C1, ldc)

    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 64 or N < 64 or K < 64:
        return None

    for _ in range(10):
        _ = flagbench.baseline.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C0.clone(), ldc)
        _ = flagbench.triton.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C1.clone(), ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C0.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasZgemm_v2(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C1.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    return CustomBenchmarkResult(
        ref_time=ms_baseline,
        res_time=ms_triton,
        speedup=ms_baseline / ms_triton if ms_triton > 0 else 0,
        params={'dtype': str(dtype), 'transa': transa, 'transb': transb, 'alpha': alpha, 'beta': beta, 'M': M, 'N': N, 'K': K}
    )
