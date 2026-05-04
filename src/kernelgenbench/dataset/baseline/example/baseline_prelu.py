import torch

def non_torch_prelu(self: torch.Tensor, weight: torch.Tensor):
    """Baseline implementation using torch operations"""
    # Handle scalar weight
    if weight.numel() == 1 or weight.dim() == 0:
        return torch.where(self > 0, self, weight * self)

    # Handle per-channel weight
    # weight shape: (C,), self shape: (N, C, ...)
    if self.dim() < 2:
        raise ValueError("prelu: per-channel weight requires input with at least 2 dims (N, C, ...)")

    if weight.numel() != self.size(1):
        raise ValueError(f"prelu: weight.numel() ({weight.numel()}) must equal input.size(1) ({self.size(1)})")

    # Reshape weight for broadcasting: (1, C, 1, 1, ...)
    weight = weight.view(1, -1, *([1] * (self.dim() - 2)))
    return torch.where(self > 0, self, weight * self)
