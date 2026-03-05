import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasDdgmm")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDdgmm(n, incx, dtype):
    m = n
    A = torch.randn(m, n, dtype=dtype, device=device).contiguous()

    x_right = torch.zeros(n * incx, dtype=dtype, device=device)
    x_right_values = torch.randn(n, dtype=dtype, device=device)
    x_right[::incx] = x_right_values

    x_left = torch.zeros(m * incx, dtype=dtype, device=device)
    x_left_values = torch.randn(m, dtype=dtype, device=device)
    x_left[::incx] = x_left_values

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    C_right_ref = torch.empty_like(A)
    C_right_act = torch.empty_like(A)

    ref_right = flagbench.baseline.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A.clone(), n, x_right, incx, C_right_ref, n)
    act_right = flagbench.triton.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A.clone(), n, x_right, incx, C_right_act, n)
    assert_close(act_right, ref_right, dtype)

    C_left_ref = torch.empty_like(A)
    C_left_act = torch.empty_like(A)

    ref_left = flagbench.baseline.cublasDdgmm(CUBLAS_SIDE_RIGHT, n, m, A.clone(), n, x_left, incx, C_left_ref, n)
    act_left = flagbench.triton.cublasDdgmm(CUBLAS_SIDE_RIGHT, n, m, A.clone(), n, x_left, incx, C_left_act, n)
    assert_close(act_left, ref_left, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    A_b = torch.randn(m, n, dtype=dtype, device=device).contiguous()
    x_b = torch.zeros(n * incx, dtype=dtype, device=device)
    x_b[::incx] = torch.randn(n, dtype=dtype, device=device)
    C_b = torch.empty_like(A_b)

    for _ in range(10):
        _ = flagbench.baseline.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A_b.clone(), n, x_b.clone(), incx, C_b, n)
        _ = flagbench.triton.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A_b.clone(), n, x_b.clone(), incx, C_b, n)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A_b.clone(), n, x_b.clone(), incx, C_b, n)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasDdgmm(CUBLAS_SIDE_LEFT, n, m, A_b.clone(), n, x_b.clone(), incx, C_b, n)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
