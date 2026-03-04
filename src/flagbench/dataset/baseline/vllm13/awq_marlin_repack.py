"""
vLLM awq_marlin_repack baseline wrapper.
"""
import torch
from vllm import _custom_ops


def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False
) -> torch.Tensor:
    """Wrapper for vLLM awq_marlin_repack implementation."""
    return _custom_ops.awq_marlin_repack(
        b_q_weight,
        size_k,
        size_n,
        num_bits,
        is_a_8bit
    )
