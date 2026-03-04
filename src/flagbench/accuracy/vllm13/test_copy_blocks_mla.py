import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("copy_blocks_mla")
@parametrize("config", [
    (8, 16, 576, 2),
    (32, 16, 576, 4),
    (64, 32, 576, 8),
    (128, 16, 576, 16),
    (16, 32, 576, 4),
    (256, 64, 576, 32),
])
def test_accuracy_copy_blocks_mla(config):
    num_blocks = config[0]
    block_size = config[1]
    kv_dim = config[2]
    num_mappings = config[3]

    kvc_ref = [torch.randn(num_blocks, block_size, kv_dim, device=device, dtype=torch.float16)]

    # Use non-overlapping src/dst to avoid CUDA race conditions
    perm = torch.randperm(num_blocks)
    src_idx = perm[:num_mappings].to(torch.int64)
    dst_idx = perm[num_mappings:2*num_mappings].to(torch.int64)
    block_mapping = torch.stack([src_idx, dst_idx], dim=1).to(device)

    kvc_act = [k.clone() for k in kvc_ref]
    kvc_ref2 = [k.clone() for k in kvc_ref]

    flagbench.baseline.copy_blocks_mla(kvc_ref2, block_mapping)
    flagbench.triton.copy_blocks_mla(kvc_act, block_mapping)

    assert torch.equal(kvc_act[0], kvc_ref2[0]), f"mismatch: max diff={(kvc_act[0]-kvc_ref2[0]).abs().max()}"

    # ===== Performance Test =====
    if num_blocks < 64 or num_mappings < 8:
        return None

    kvc_bench = [torch.randn(num_blocks, block_size, kv_dim, device=device, dtype=torch.float16)]

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.copy_blocks_mla([k.clone() for k in kvc_bench], block_mapping),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.copy_blocks_mla([k.clone() for k in kvc_bench], block_mapping),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
