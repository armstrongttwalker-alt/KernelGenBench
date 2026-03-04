import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("topk_softmax")
@parametrize("num_tokens", [1, 7, 32, 128, 1024, 4096])
@parametrize("num_experts", [8, 16, 64])
@parametrize("top_k", [1, 2, 4])
@parametrize("renormalize", [False, True])
def test_accuracy_topk_softmax(num_tokens, num_experts, top_k, renormalize):
    # ===== Accuracy Test =====
    # top_k must not exceed num_experts
    if top_k > num_experts:
        return None

    gating_output = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)

    # Output tensors for baseline
    ref_weights = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    ref_ids = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    ref_indices = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)

    # Output tensors for triton
    act_weights = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
    act_ids = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
    act_indices = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)

    flagbench.baseline.topk_softmax(
        ref_weights, ref_ids, ref_indices, gating_output, renormalize)

    flagbench.triton.topk_softmax(
        act_weights, act_ids, act_indices, gating_output, renormalize)

    # Compare weights and ids
    assert_close(act_weights, ref_weights, torch.float32)
    assert (act_ids == ref_ids).all(), "topk_ids mismatch"

    # ===== Performance Test =====
    if num_tokens < 1024:
        return None

    def bench_fn(impl):
        def fn():
            w = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.float32)
            i = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
            t = torch.empty(num_tokens, top_k, device='cuda', dtype=torch.int32)
            impl(w, i, t, gating_output, renormalize)
        return fn

    ms_baseline = triton.testing.do_bench(bench_fn(flagbench.baseline.topk_softmax), warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(bench_fn(flagbench.triton.topk_softmax), warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
