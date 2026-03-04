"""
vLLM apply_repetition_penalties baseline wrapper.
"""
import torch
from vllm import _custom_ops


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor
) -> None:
    """Apply repetition penalties to logits in-place."""
    _custom_ops.apply_repetition_penalties(
        logits,
        prompt_mask,
        output_mask,
        repetition_penalties
    )
