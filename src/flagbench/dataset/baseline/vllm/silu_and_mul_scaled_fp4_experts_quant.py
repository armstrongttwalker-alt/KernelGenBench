"""
vLLM silu_and_mul_scaled_fp4_experts_quant baseline wrapper.
"""
import torch
from vllm import _custom_ops


def silu_and_mul_scaled_fp4_experts_quant(
    input_tensor: torch.Tensor,
    input_global_scale: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    topk: int
) -> tuple:
    """Fused SiLU+Mul+NVFP4 quantization for MoE intermediate activations."""
    return _custom_ops.silu_and_mul_scaled_fp4_experts_quant(
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
        topk
    )
