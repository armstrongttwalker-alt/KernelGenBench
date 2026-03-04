import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("allspark_repack_weight")
@parametrize("config", [
    (128, 64),
    (256, 128),
    (512, 256),
    (256, 256),
    (1024, 512),
    (1024, 1024),
    (1024, 2048),
    (2048, 512),
    (2048, 1024),
    (2048, 2048),
    (4096, 1024),
    (4096, 2048),
    (4096, 4096),
])
def test_accuracy_allspark_repack_weight(config):
    K = config[0]
    N = config[1]

    qweight = torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
    scale = torch.randn(1, N, device=device, dtype=torch.float16)

    ref_out = flagbench.baseline.allspark_repack_weight(qweight, scale)
    act_out = flagbench.triton.allspark_repack_weight(qweight, scale)

    assert torch.equal(ref_out[0], act_out[0]), f"weight mismatch"
    assert torch.equal(ref_out[1], act_out[1]), f"scale mismatch"

    # ===== Performance Test =====
    if K < 1024 or N < 512:
        return None

    qweight_bench = torch.randint(0, 255, (K, N), device=device, dtype=torch.uint8)
    scale_bench = torch.randn(1, N, device=device, dtype=torch.float16)

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.allspark_repack_weight(qweight_bench, scale_bench),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.allspark_repack_weight(qweight_bench, scale_bench),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
