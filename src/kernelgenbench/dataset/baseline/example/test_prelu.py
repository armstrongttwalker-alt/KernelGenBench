import torch
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from torch.testing import assert_close

device = "cuda"

@label("non_torch_prelu")
@parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16), (2, 128, 64, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("weight_kind", ["scalar", "per_channel"])
def test_non_torch_prelu(shape, dtype, weight_kind):
    x = torch.randn(shape, dtype=dtype, device=device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=device)

    # 统一调用 kernelgenbench.triton.non_torch_prelu
    # DISPATCH_TORCH_LIB=0 时调用 baseline
    # DISPATCH_TORCH_LIB=1 时调用 triton
    result = kernelgenbench.triton.non_torch_prelu(x, w)

    # 与 PyTorch 参考实现比较
    ref = torch.nn.functional.prelu(x, w)
    assert_close(result, ref, rtol=1e-3, atol=1e-3)
