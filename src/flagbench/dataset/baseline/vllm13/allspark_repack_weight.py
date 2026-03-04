"""
vLLM allspark_repack_weight baseline wrapper.
"""
import torch
from vllm import _custom_ops


def allspark_repack_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    has_zp: bool = False
) -> tuple:
    """Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format"""
    return _custom_ops.allspark_repack_weight(
        qweight,
        scale,
        zero_point,
        has_zp
    )
