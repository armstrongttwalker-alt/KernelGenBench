import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasDgemm_v2")
@parametrize("M, N, K", [
    (1, 1, 1),
    (16, 16, 16),
    (32, 64, 16),
    (64, 32, 48),
    (128, 256, 64),
    (17, 33, 65),
    (71, 71, 71),
    (1024, 1024, 256),
])
@parametrize("alpha, beta", [
    (1.0, 0.0),
    (0.5, 0.5),
    (-1.0, 1.0),
])
@parametrize("transa, transb", [("N", "N"), ("N", "T"), ("T", "N"), ("T", "T")])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDgemm_v2(M, N, K, alpha, beta, transa, transb, dtype):
    A_shape = (K, M) if transa == 'T' else (M, K)
    B_shape = (N, K) if transb == 'T' else (K, N)
    A = torch.randn(A_shape, dtype=dtype, device=device)
    B = torch.randn(B_shape, dtype=dtype, device=device)
    C = torch.randn(M, N, dtype=dtype, device=device)

    lda = A.shape[0]
    ldb = B.shape[0]
    ldc = M

    C_ref = C.clone()
    C_act = C.clone()

    ref_out = flagbench.baseline.cublasDgemm_v2(
        transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_ref, ldc
    )
    act_out = flagbench.triton.cublasDgemm_v2(
        transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C_act, ldc
    )
    assert_close(act_out, ref_out, dtype, reduce_dim=K)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if M < 128 or N < 128 or K < 64:
        return None

    A_b = torch.randn(A_shape, dtype=dtype, device=device)
    B_b = torch.randn(B_shape, dtype=dtype, device=device)

    for _ in range(10):
        C_w = torch.randn(M, N, dtype=dtype, device=device)
        _ = flagbench.baseline.cublasDgemm_v2(transa, transb, M, N, K, alpha, A_b, lda, B_b, ldb, beta, C_w.clone(), ldc)
        _ = flagbench.triton.cublasDgemm_v2(transa, transb, M, N, K, alpha, A_b, lda, B_b, ldb, beta, C_w.clone(), ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    C_b = torch.randn(M, N, dtype=dtype, device=device)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasDgemm_v2(transa, transb, M, N, K, alpha, A_b, lda, B_b, ldb, beta, C_b.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasDgemm_v2(transa, transb, M, N, K, alpha, A_b, lda, B_b, ldb, beta, C_b.clone(), ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
