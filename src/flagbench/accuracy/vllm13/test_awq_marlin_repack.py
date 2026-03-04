import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("awq_marlin_repack")
@parametrize("config", [
    (256, 256, 4),
    (512, 256, 4),
    (256, 512, 4),
    (512, 512, 4),
    (512, 1024, 4),
    (1024, 512, 4),
    (1024, 1024, 4),
    (2048, 1024, 4),
    (256, 256, 8),
    (512, 512, 8),
    (512, 1024, 8),
    (1024, 512, 8),
    (1024, 1024, 8),
])
def test_accuracy_awq_marlin_repack(config):
    K = config[0]
    N = config[1]
    num_bits = config[2]
    pack_factor = 32 // num_bits

    b_q_weight = torch.randint(0, 2**31, (K, N // pack_factor), device=device, dtype=torch.int32)

    ref_out = flagbench.baseline.awq_marlin_repack(b_q_weight, K, N, num_bits)
    act_out = flagbench.triton.awq_marlin_repack(b_q_weight, K, N, num_bits)

    assert torch.equal(act_out, ref_out), f"Mismatch: max diff={(act_out - ref_out).abs().max()}"

    # ===== Performance Test =====
    if K < 512 or N < 512:
        return None

    b_q_weight_bench = torch.randint(0, 2**31, (K, N // pack_factor), device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.awq_marlin_repack(b_q_weight_bench, K, N, num_bits),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.awq_marlin_repack(b_q_weight_bench, K, N, num_bits),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
