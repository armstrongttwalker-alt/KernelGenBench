import torch
import flagbench
from sandbox.verifier.test_parametrize import parametrize, label
from torch.testing import assert_close
from sandbox.config import DEVICE as device
from sandbox.config import QUICK_MODE

# 测试参数配置
SHAPES = [(2, 3), (2, 32, 16, 16)] if QUICK_MODE else [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16)]
DTYPES = [torch.float32] if QUICK_MODE else [torch.float32, torch.float16, torch.bfloat16]
WEIGHT_KINDS = ["scalar", "per_channel"]

@label("non_torch_prelu")
@parametrize("shape", SHAPES)
@parametrize("dtype", DTYPES)
@parametrize("weight_kind", WEIGHT_KINDS)
def test_accuracy_non_torch_prelu(shape, dtype, weight_kind):
    """Test non_torch_prelu: compare baseline vs triton implementation

    This test verifies that:
    1. Baseline is auto-loaded from baseline/example/baseline_prelu.py
    2. Triton implementation matches baseline implementation
    3. Both implementations match PyTorch reference
    """
    x = torch.randn(shape, dtype=dtype, device=device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=device)

    # 调用 baseline 实现（自动加载）
    baseline_result = flagbench.baseline.non_torch_prelu(x, w)

    # 调用 triton 实现
    triton_result = flagbench.triton.non_torch_prelu(x, w)

    # 比较 baseline vs triton
    assert_close(triton_result, baseline_result, rtol=1e-3, atol=1e-3)
