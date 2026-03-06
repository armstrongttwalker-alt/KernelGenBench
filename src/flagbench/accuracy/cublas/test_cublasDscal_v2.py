import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
import torch

@label("cublasDscal_v2")
@parametrize("n", [1, 32, 71, 1024, 4096, 5333])
@parametrize("alpha", [1.0, 0.0, -0.5, 100.001])
@parametrize("incx", [1, 2, 3])
@parametrize("dtype", [torch.float64])
def test_accuracy_cublasDscal_v2(n, alpha, incx, dtype):
    x0 = torch.randn(n * incx, dtype=dtype, device=device)
    x1 = x0.clone()

    ref_out = flagbench.baseline.cublasDscal_v2(n, alpha, x0, incx)
    act_out = flagbench.triton.cublasDscal_v2(n, alpha, x1, incx)

    assert_close(act_out, ref_out, dtype)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    if n < 1024:
        return None

    x_b = torch.randn(n * incx, dtype=dtype, device=device)

    for _ in range(10):
        _ = flagbench.baseline.cublasDscal_v2(n, alpha, x_b.clone(), incx)
        _ = flagbench.triton.cublasDscal_v2(n, alpha, x_b.clone(), incx)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.baseline.cublasDscal_v2(n, alpha, x_b.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_baseline = start_event.elapsed_time(end_event) / 100

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(100):
        _ = flagbench.triton.cublasDscal_v2(n, alpha, x_b.clone(), incx)
    end_event.record()
    torch.cuda.synchronize()
    ms_triton = start_event.elapsed_time(end_event) / 100

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
