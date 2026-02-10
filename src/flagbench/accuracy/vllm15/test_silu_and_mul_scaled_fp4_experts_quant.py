import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("silu_and_mul_scaled_fp4_experts_quant")
@parametrize("m_size", [1, 71, 128, 1024, 5333])
@parametrize("k2_size", [32, 497, 512, 4096, 8192])
@parametrize("n_experts", [4, 16, 64])
@parametrize("topk", [1, 4])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_silu_and_mul_scaled_fp4_experts_quant(m_size, k2_size, n_experts, topk, dtype):
    # ===== Accuracy Test =====
    # Create inputs
    # input_tensor: [m_topk, k*2]
    input_tensor = torch.randn((m_size, k2_size), device='cuda', dtype=dtype)

    # input_global_scale: per-expert scaling [n_experts]
    # Use positive scales to mimic typical quantization scales
    input_global_scale = (torch.rand((n_experts,), device='cuda', dtype=torch.float32) + 0.5)

    # expert_offsets: cumulative row counts per expert [n_experts+1], sum to m_size
    base = m_size // n_experts
    rem = m_size % n_experts
    per_expert_rows = [base + (1 if i < rem else 0) for i in range(n_experts)]
    expert_offsets_list = [0]
    for cnt in per_expert_rows:
        expert_offsets_list.append(expert_offsets_list[-1] + cnt)
    expert_offsets = torch.tensor(expert_offsets_list, device='cuda', dtype=torch.int32)

    # blockscale_offsets: cumulative block counts per expert [n_experts+1]
    # Assume blocks per row computed along k dimension with 64-size blocks
    k = k2_size // 2
    blocks_per_row = max(1, (k + 63) // 64)
    per_expert_blocks = [r * blocks_per_row for r in per_expert_rows]
    blockscale_offsets_list = [0]
    for b in per_expert_blocks:
        blockscale_offsets_list.append(blockscale_offsets_list[-1] + b)
    blockscale_offsets = torch.tensor(blockscale_offsets_list, device='cuda', dtype=torch.int32)

    # Call baseline: flagbench.baseline.silu_and_mul_scaled_fp4_experts_quant(...)
    ref_out = flagbench.baseline.silu_and_mul_scaled_fp4_experts_quant(
        input_tensor, input_global_scale, expert_offsets, blockscale_offsets, topk)

    # Call triton: flagbench.triton.silu_and_mul_scaled_fp4_experts_quant(...)
    act_out = flagbench.triton.silu_and_mul_scaled_fp4_experts_quant(
        input_tensor, input_global_scale, expert_offsets, blockscale_offsets, topk)

    # Compare: assert_close(act_out, ref_out, dtype)
    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if (m_size < 1024) or (k2_size < 4096):
        return None

    # Prepare fresh data for benchmarking
    x_bench = torch.randn((m_size, k2_size), device='cuda', dtype=dtype)
    scales_bench = (torch.rand((n_experts,), device='cuda', dtype=torch.float32) + 0.5)
    expert_offsets_bench = expert_offsets.clone()
    blockscale_offsets_bench = blockscale_offsets.clone()

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.silu_and_mul_scaled_fp4_experts_quant(
            x_bench.clone(), scales_bench.clone(), expert_offsets_bench, blockscale_offsets_bench, topk
        ),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.silu_and_mul_scaled_fp4_experts_quant(
            x_bench.clone(), scales_bench.clone(), expert_offsets_bench, blockscale_offsets_bench, topk
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )