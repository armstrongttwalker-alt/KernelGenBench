import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("cgeru")
@parametrize("m, n", [
    (1, 1),
    (1, 32),
    (32, 1),
    (15, 160),
    (160, 1024),
    (495, 5333),
    (33, 273),
    (16, 256),
    (5333, 497),
])
@parametrize("alpha", [
    1.0,
    0.0,
    0.001,
    -0.999,
    100.001,
    -111.999,
    0.5,
    -0.5,
    0.3 + 0.7j,
    -0.25 - 0.5j,
])
@parametrize("dtype", [torch.complex64])
def test_cgeru_cublas_baseline(m, n, alpha, dtype):
    # Initialize input tensors
    x = torch.complex(
        torch.randn(m, dtype=torch.float32, device="cuda"),
        torch.randn(m, dtype=torch.float32, device="cuda"),
    ).to(dtype)
    y = torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    ).to(dtype)
    A0 = torch.complex(
        torch.randn(m, n, dtype=torch.float32, device="cuda"),
        torch.randn(m, n, dtype=torch.float32, device="cuda"),
    ).to(dtype)

    # Stride parameters (unit strides)
    incx = 1
    incy = 1
    lda = max(1, m)

    # Clone A for independent runs
    A_ref = A0.clone()
    A_act = A0.clone()

    # Baseline call
    ref_out = flagbench.baseline.cgeru(m, n, alpha, x, incx, y, incy, A_ref, lda)

    # Triton implementation call
    act_out = flagbench.triton.cgeru(m, n, alpha, x, incx, y, incy, A_act, lda)

    # Compare results
    assert_close(act_out, ref_out, dtype)