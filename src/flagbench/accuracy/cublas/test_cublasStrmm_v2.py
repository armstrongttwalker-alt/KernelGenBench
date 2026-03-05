import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasStrmm_v2")
@parametrize("m", [4, 16, 48])
@parametrize("n", [4, 16, 48])
@parametrize("alpha", [1.0])
@parametrize("side", [0, 1])
@parametrize("uplo", [0, 1])
@parametrize("trans", [0, 1])
@parametrize("diag", [0, 1])
@parametrize("dtype", [torch.float32])
def test_accuracy_cublasStrmm_v2(m, n, alpha, side, uplo, trans, diag, dtype):
    # side=0: LEFT (A is m×m), side=1: RIGHT (A is n×n)
    k = m if side == 0 else n
    lda = k
    ldb = m
    ldc = m

    A = torch.randn(k, k, dtype=dtype, device=device)
    B = torch.randn(m, n, dtype=dtype, device=device)
    C_ref = torch.empty(m, n, dtype=dtype, device=device)
    C_act = torch.empty(m, n, dtype=dtype, device=device)

    ref_out = flagbench.baseline.cublasStrmm_v2(
        side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_ref, ldc
    )
    act_out = flagbench.triton.cublasStrmm_v2(
        side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_act, ldc
    )
    assert_close(act_out, ref_out, dtype, reduce_dim=k)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if m < 64 or n < 64:
        return None

    for _ in range(10):
        _ = flagbench.baseline.cublasStrmm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_ref, ldc)
        _ = flagbench.triton.cublasStrmm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_act, ldc)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasStrmm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_ref, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasStrmm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B.clone(), ldb, C_act, ldc)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
