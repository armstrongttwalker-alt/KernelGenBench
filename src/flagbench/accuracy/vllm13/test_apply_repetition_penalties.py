import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("apply_repetition_penalties")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096), (5333, 8192)])
@parametrize("use_output_mask", [True, False])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("penalty_mode", [0, 1])
def test_accuracy_apply_repetition_penalties(shape, use_output_mask, dtype, penalty_mode):
    # ===== Accuracy Test =====
    # Create inputs
    num_seqs, vocab_size = shape
    torch.manual_seed(42)

    # Logits
    logits = torch.randn(num_seqs, vocab_size, device='cuda', dtype=dtype)

    # Prompt mask
    prompt_mask = (torch.rand(num_seqs, vocab_size, device='cuda') < 0.3).to(torch.bool)

    # Output mask
    if use_output_mask:
        output_mask = (torch.rand(num_seqs, vocab_size, device='cuda') < 0.2).to(torch.bool)
    else:
        output_mask = torch.zeros(num_seqs, vocab_size, device='cuda', dtype=torch.bool)

    # Repetition penalties per sequence
    if penalty_mode == 0:
        # Uniform in [1.0, 2.0]
        repetition_penalties = (1.0 + torch.rand(num_seqs, device='cuda', dtype=dtype)).contiguous()
    else:
        # Patterned penalties including 1.0 (no-op), <1.0 (boost), and >1.0 (penalize)
        base_vals = torch.tensor([0.9, 1.0, 1.2, 1.5, 2.0], device='cuda', dtype=dtype)
        reps = (num_seqs + base_vals.numel() - 1) // base_vals.numel()
        repetition_penalties = base_vals.tile((reps,))[:num_seqs].contiguous()

    # Call baseline: in-place on logits clone
    ref_logits = logits.clone()
    flagbench.baseline.apply_repetition_penalties(ref_logits, prompt_mask, output_mask, repetition_penalties)

    # Call triton: in-place on logits clone
    act_logits = logits.clone()
    flagbench.triton.apply_repetition_penalties(act_logits, prompt_mask, output_mask, repetition_penalties)

    # Compare: mutated tensors should match
    assert_close(act_logits, ref_logits, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if max(num_seqs, vocab_size) < 1024:
        return None

    # Prepare fresh data for benchmarking
    bench_logits = torch.randn(num_seqs, vocab_size, device='cuda', dtype=dtype)
    # Use the same masks and penalties; lambda must create fresh tensors each call due to in-place op

    # Benchmark baseline
    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.apply_repetition_penalties(
            bench_logits.clone(), prompt_mask, output_mask, repetition_penalties
        ),
        warmup=25, rep=100
    )

    # Benchmark triton
    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.apply_repetition_penalties(
            bench_logits.clone(), prompt_mask, output_mask, repetition_penalties
        ),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)