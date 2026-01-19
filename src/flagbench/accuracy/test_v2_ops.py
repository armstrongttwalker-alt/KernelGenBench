#!/usr/bin/env python3
"""
Auto-generated test file for v2 operators.

This file contains test functions extracted from evaluation results.
"""

import torch
import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS


# ========== log_sigmoid_backward ==========
@label("log_sigmoid_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log_sigmoid_backward_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    buffer = torch.sigmoid(self)

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_buffer = buffer.clone()
    ref_out = torch.ops.aten.log_sigmoid_backward(ref_grad_output, ref_self, ref_buffer)

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_buffer = buffer.clone()
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log_sigmoid_backward(act_grad_output, act_self, act_buffer)

    assert_close(act_out, ref_out, dtype=dtype)


@label("log_sigmoid_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log_sigmoid_backward_grad_input(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    buffer = torch.sigmoid(self)

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_buffer = buffer.clone()
    ref_out_tensor = torch.empty_like(ref_self)
    ref_out = torch.ops.aten.log_sigmoid_backward.grad_input(
        ref_grad_output, ref_self, ref_buffer, grad_input=ref_out_tensor
    )

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_buffer = buffer.clone()
    act_out_tensor = torch.empty_like(act_self)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log_sigmoid_backward.grad_input(
            act_grad_output, act_self, act_buffer, grad_input=act_out_tensor
        )

    assert_close(act_out, ref_out, dtype=dtype)

# ========== mish_backward ==========
@label("mish_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mish_backward_tensor(shape, dtype):
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    self_tensor = torch.randn(shape, dtype=dtype, device=device)

    ref_grad = grad_output.clone()
    ref_self = self_tensor.clone()
    ref_out = torch.ops.aten.mish_backward(ref_grad, ref_self)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mish_backward(grad_output, self_tensor)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== reflection_pad1d_backward ==========
@label("reflection_pad1d_backward")
@parametrize("shape", [(2, 3, 8), (4, 16, 64), (32, 64, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("padding", [(1, 1), (2, 3), (5, 5)])
def test_reflection_pad1d_backward_tensor(shape, dtype, padding):
    N, C, W = shape
    pad_l, pad_r = padding
    W_out = W + pad_l + pad_r

    self = torch.randn((N, C, W), dtype=dtype, device=device)
    grad_output = torch.randn((N, C, W_out), dtype=dtype, device=device)

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()

    ref_out = torch.ops.aten.reflection_pad1d_backward(ref_grad_output, ref_self, padding)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.reflection_pad1d_backward(grad_output, self, padding)

    assert_close(act_out, ref_out, dtype=dtype)


@label("reflection_pad1d_backward")
@parametrize("shape", [(2, 3, 8), (4, 16, 64), (32, 64, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("padding", [(1, 1), (2, 3), (5, 5)])
def test_reflection_pad1d_backward_grad_input(shape, dtype, padding):
    N, C, W = shape
    pad_l, pad_r = padding
    W_out = W + pad_l + pad_r

    self = torch.randn((N, C, W), dtype=dtype, device=device)
    grad_output = torch.randn((N, C, W_out), dtype=dtype, device=device)

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()

    ref_out_buf = torch.empty_like(ref_self)
    act_out_buf = torch.empty_like(self)

    ref_out = torch.ops.aten.reflection_pad1d_backward.grad_input(ref_grad_output, ref_self, padding, grad_input=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.reflection_pad1d_backward.grad_input(grad_output, self, padding, grad_input=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== rrelu_with_noise_backward ==========
@label("rrelu_with_noise_backward")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("training", [True, False])
def test_rrelu_with_noise_backward_tensor(shape, dtype, training):
    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    noise = (lower + (upper - lower) * torch.rand(shape, dtype=dtype, device=device))
    self_is_result = False

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_noise = noise.clone()

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()

    ref_out = torch.ops.aten.rrelu_with_noise_backward(ref_grad_output, ref_self, ref_noise, lower, upper, training, self_is_result)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rrelu_with_noise_backward(act_grad_output, act_self, act_noise, lower, upper, training, self_is_result)

    assert_close(act_out, ref_out, dtype=dtype)


@label("rrelu_with_noise_backward")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("training", [True, False])
def test_rrelu_with_noise_backward_out(shape, dtype, training):
    lower = 0.1
    upper = 0.3
    self = torch.randn(shape, dtype=dtype, device=device)
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    noise = (lower + (upper - lower) * torch.rand(shape, dtype=dtype, device=device))
    self_is_result = True

    ref_self = self.clone()
    ref_grad_output = grad_output.clone()
    ref_noise = noise.clone()
    ref_out_buf = torch.empty_like(ref_self)

    act_self = self.clone()
    act_grad_output = grad_output.clone()
    act_noise = noise.clone()
    act_out_buf = torch.empty_like(act_self)

    ref_out = torch.ops.aten.rrelu_with_noise_backward.out(ref_grad_output, ref_self, ref_noise, lower, upper, training, self_is_result, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rrelu_with_noise_backward.out(act_grad_output, act_self, act_noise, lower, upper, training, self_is_result, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== select_backward ==========
@label("select_backward")
@parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@parametrize("dim", [0, -1])
@parametrize("index_mode", ["first", "mid", "neg1"])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_default(input_sizes, dim, index_mode, dtype):
    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad_ref = torch.randn(out_shape, dtype=dtype, device=device)
    grad_act = grad_ref.clone()

    ref_out = torch.ops.aten.select_backward(grad_ref, list(input_sizes), dim, index)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.select_backward(grad_act, list(input_sizes), dim, index)

    assert_close(act_out, ref_out, dtype=dtype)


@label("select_backward")
@parametrize("input_sizes", [(2, 3), (128, 256), (512, 512)])
@parametrize("dim", [0, -1])
@parametrize("index_mode", ["first", "mid", "neg1"])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_select_backward_out(input_sizes, dim, index_mode, dtype):
    ndim = len(input_sizes)
    dim_c = dim % ndim
    size_d = input_sizes[dim_c]
    if index_mode == "first":
        index = 0
    elif index_mode == "mid":
        index = size_d // 2
    else:
        index = -1
    out_shape = list(input_sizes)
    del out_shape[dim_c]
    out_shape = tuple(out_shape) if len(out_shape) > 0 else ()
    grad_ref = torch.randn(out_shape, dtype=dtype, device=device)
    grad_act = grad_ref.clone()

    ref_out_buf = torch.empty(input_sizes, dtype=dtype, device=device)
    ref_out = torch.ops.aten.select_backward.out(grad_ref, list(input_sizes), dim, index, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out_buf = torch.empty(input_sizes, dtype=dtype, device=device)
        act_out = torch.ops.aten.select_backward.out(grad_act, list(input_sizes), dim, index, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== smooth_l1_loss_backward ==========
@label("smooth_l1_loss_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_backward_tensor(shape, dtype, reduction, beta):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)
    if reduction == 0:
        grad_output = torch.randn(shape, dtype=dtype, device=device)
    else:
        grad_output = torch.randn((), dtype=dtype, device=device)

    ref_grad_output = grad_output.clone()
    ref_self = self.clone()
    ref_target = target.clone()

    ref_out = torch.ops.aten.smooth_l1_loss_backward(ref_grad_output, ref_self, ref_target, reduction, float(beta))

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.smooth_l1_loss_backward(grad_output, self, target, reduction, float(beta))

    assert_close(act_out, ref_out, dtype=dtype)


@label("smooth_l1_loss_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss_backward_grad_input(shape, dtype, reduction, beta):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)
    if reduction == 0:
        grad_output = torch.randn(shape, dtype=dtype, device=device)
    else:
        grad_output = torch.randn((), dtype=dtype, device=device)

    ref_grad_output = grad_output.clone()
    ref_self = self.clone()
    ref_target = target.clone()

    ref_out_buf = torch.empty_like(ref_self)
    ref_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
        ref_grad_output, ref_self, ref_target, reduction, float(beta), grad_input=ref_out_buf
    )

    act_out_buf = torch.empty_like(self)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.smooth_l1_loss_backward.grad_input(
            grad_output, self, target, reduction, float(beta), grad_input=act_out_buf
        )

    assert_close(act_out, ref_out, dtype=dtype)

# ========== softplus_backward ==========
@label("softplus_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("beta", [0.5, 1.0, 2.0])
@parametrize("threshold", [10.0, 20.0])
def test_softplus_backward_tensor(shape, dtype, beta, threshold):
    x = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)

    ref_x = x.clone()
    ref_grad = grad.clone()

    ref_out = torch.ops.aten.softplus_backward(ref_grad, ref_x, beta, threshold)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.softplus_backward(grad, x, beta, threshold)

    assert_close(act_out, ref_out, dtype=dtype)


@label("softplus_backward")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("beta", [0.5, 1.0, 2.0])
@parametrize("threshold", [10.0, 20.0])
def test_softplus_backward_grad_input_out(shape, dtype, beta, threshold):
    x = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)

    ref_x = x.clone()
    ref_grad = grad.clone()
    ref_out_buf = torch.empty_like(ref_x)

    ref_out = torch.ops.aten.softplus_backward.grad_input(
        ref_grad, ref_x, beta, threshold, grad_input=ref_out_buf
    )

    act_out_buf = torch.empty_like(x)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.softplus_backward.grad_input(
            grad, x, beta, threshold, grad_input=act_out_buf
        )

    assert_close(act_out, ref_out, dtype=dtype)
    assert_close(act_out_buf, ref_out_buf, dtype=dtype)

# ========== upsample_nearest2d_backward ==========
@label("upsample_nearest2d_backward")
@parametrize("case", [
    (2, 3, 4, 5, 8, 10),
    (4, 8, 16, 32, 24, 48),
    (1, 32, 64, 128, 128, 256),
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("use_scales", [False, True])
def test_upsample_nearest2d_backward_tensor(case, dtype, use_scales):
    N, C, in_h, in_w, out_h, out_w = case
    grad_output = torch.randn((N, C, out_h, out_w), dtype=dtype, device=device)
    scales_h = float(out_h) / float(in_h) if use_scales else None
    scales_w = float(out_w) / float(in_w) if use_scales else None

    ref_grad_output = grad_output.clone()
    ref_out = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad_output, [out_h, out_w], [N, C, in_h, in_w], scales_h, scales_w
    )

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, [out_h, out_w], [N, C, in_h, in_w], scales_h, scales_w
        )

    assert_close(act_out, ref_out, dtype=dtype)


@label("upsample_nearest2d_backward")
@parametrize("case", [
    (2, 3, 4, 5, 8, 10),
    (4, 8, 16, 32, 24, 48),
    (1, 32, 64, 128, 128, 256),
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("use_scales", [False, True])
def test_upsample_nearest2d_backward_grad_input(case, dtype, use_scales):
    N, C, in_h, in_w, out_h, out_w = case
    grad_output = torch.randn((N, C, out_h, out_w), dtype=dtype, device=device)
    ref_grad_input = torch.empty((N, C, in_h, in_w), dtype=dtype, device=device)
    act_grad_input = torch.empty_like(ref_grad_input)

    scales_h = float(out_h) / float(in_h) if use_scales else None
    scales_w = float(out_w) / float(in_w) if use_scales else None

    ref_grad_output = grad_output.clone()
    ref_out = torch.ops.aten.upsample_nearest2d_backward.grad_input(
        ref_grad_output, [out_h, out_w], [N, C, in_h, in_w], scales_h, scales_w, grad_input=ref_grad_input
    )

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.upsample_nearest2d_backward.grad_input(
            grad_output, [out_h, out_w], [N, C, in_h, in_w], scales_h, scales_w, grad_input=act_grad_input
        )

    assert_close(act_out, ref_out, dtype=dtype)

# ========== erfc ==========
@label("erfc")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfc_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.erfc(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.erfc(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("erfc")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_erfc_out_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.erfc.out(ref_x, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.empty_like(x)
        torch.ops.aten.erfc.out(x, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)


@label("erfc")
@parametrize("a", [-5, -1, 0, 2, 7])
def test_erfc_int(a):
    ref_out = torch.ops.aten.erfc(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.erfc(a)
    assert_close(act_out, ref_out, dtype=torch.float32)


@label("erfc")
@parametrize("a", [-3.0, -0.5, 0.0, 1.5, 10.25])
def test_erfc_float(a):
    ref_out = torch.ops.aten.erfc(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.erfc(a)
    assert_close(act_out, ref_out, dtype=torch.float32)


@label("erfc")
@parametrize("a", [False, True])
def test_erfc_scalar(a):
    ref_out = torch.ops.aten.erfc(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.erfc(a)
    assert_close(act_out, ref_out, dtype=torch.float32)

# ========== hardsigmoid ==========
@label("hardsigmoid")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_tensor(shape, dtype):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()

    ref_out = torch.ops.aten.hardsigmoid(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.hardsigmoid(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("hardsigmoid")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hardsigmoid_out(shape, dtype):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    ref_out = torch.ops.aten.hardsigmoid.out(ref_x, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.hardsigmoid.out(x, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== heaviside ==========
@label("heaviside")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_tensor(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=device)
    values_tensor = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = self_tensor.clone()
    ref_values = values_tensor.clone()

    ref_out = torch.ops.aten.heaviside(ref_self, ref_values)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.heaviside(self_tensor, values_tensor)

    assert_close(act_out, ref_out, dtype=dtype)


@label("heaviside")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_heaviside_out(shape, dtype):
    self_tensor = torch.randn(shape, dtype=dtype, device=device)
    values_tensor = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.rand(shape, device=device) < 0.1
    self_tensor[mask] = 0.0

    ref_self = self_tensor.clone()
    ref_values = values_tensor.clone()
    ref_out_buf = torch.empty_like(ref_self)

    ref_out = torch.ops.aten.heaviside.out(ref_self, ref_values, out=ref_out_buf)

    act_out_buf = torch.empty_like(self_tensor)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.heaviside.out(self_tensor, values_tensor, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== log10 ==========
@label("log10")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log10_tensor(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=device) + 0.01
    ref_x = x.clone()

    ref_out = torch.ops.aten.log10(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log10(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("log10")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_log10_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=device) + 0.01
    ref_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    ref_out = torch.ops.aten.log10.out(ref_x, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log10.out(x, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)


@label("log10")
@parametrize("a", [1, 2, 10, 1000, 1000000])
def test_log10_int(a):
    ref_out = torch.ops.aten.log10.int(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log10.int(a)
    assert_close(act_out, ref_out, dtype=torch.float32)


@label("log10")
@parametrize("a", [0.1, 1.0, 3.14, 10.0, 123.456])
def test_log10_float(a):
    ref_out = torch.ops.aten.log10.float(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log10.float(a)
    assert_close(act_out, ref_out, dtype=torch.float32)


@label("log10")
@parametrize("a", [complex(1.0, 0.0), complex(0.1, 2.0), complex(3.0, -4.0), complex(10.0, 0.5)])
def test_log10_complex(a):
    ref_out = torch.ops.aten.log10.complex(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.log10.complex(a)
    assert_close(act_out, ref_out, dtype=torch.complex64)


@label("log10")
@parametrize("a", [1, 10.0, True, 3.5])
def test_log10_scalar(a):
    ref_out = torch.ops.aten.log10.Scalar(a)
    with flagbench.use_gems(REGISTERED_OP_OPS):
        act_out = torch.ops.aten.log10.Scalar(a)
    assert_close(act_out, ref_out, dtype=torch.float32)

# ========== logit ==========
@label("logit")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("eps", [None, 1e-6, 1e-3])
def test_logit_tensor(shape, dtype, eps):
    if eps is None:
        base = torch.rand(shape, dtype=torch.float32, device=device) * 0.998 + 0.001
    else:
        base = torch.rand(shape, dtype=torch.float32, device=device)
        flat = base.view(-1)
        if flat.numel() >= 2:
            flat[0] = 0.0
            flat[1] = 1.0
    input_tensor = base.to(dtype)

    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.logit(ref_input, eps)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.logit(input_tensor, eps)

    assert_close(act_out, ref_out, dtype=dtype)


@label("logit")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("eps", [None, 1e-6, 1e-3])
def test_logit_out(shape, dtype, eps):
    if eps is None:
        base = torch.rand(shape, dtype=torch.float32, device=device) * 0.998 + 0.001
    else:
        base = torch.rand(shape, dtype=torch.float32, device=device)
        flat = base.view(-1)
        if flat.numel() >= 2:
            flat[0] = 0.0
            flat[1] = 1.0
    input_tensor = base.to(dtype)

    ref_input = input_tensor.clone()
    ref_out_buf = torch.empty(shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.logit.out(ref_input, eps, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out_buf = torch.empty(shape, dtype=dtype, device=device)
        act_out = torch.ops.aten.logit.out(input_tensor, eps, out=act_out_buf)

    assert_close(act_out_buf, ref_out_buf, dtype=dtype)

# ========== mish ==========
@label("mish")
@parametrize("shape", [(2, 3), (128, 256), (512, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mish_tensor(shape, dtype):
    x_ref = torch.randn(shape, device=device, dtype=dtype)
    x_act = x_ref.clone()

    ref_out = torch.ops.aten.mish(x_ref)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mish(x_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("mish")
@parametrize("shape", [(2, 3), (128, 256), (512, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mish_out(shape, dtype):
    x_ref = torch.randn(shape, device=device, dtype=dtype)
    x_act = x_ref.clone()

    ref_out = torch.empty_like(x_ref)
    act_out = torch.empty_like(x_act)

    ref_ret = torch.ops.aten.mish.out(x_ref, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_ret = torch.ops.aten.mish.out(x_act, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)
    assert_close(act_ret, ref_ret, dtype=dtype)

# ========== prelu ==========
@label("prelu")
@parametrize("shape", [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16), (2, 128, 64, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("weight_kind", ["scalar", "per_channel"])
def test_prelu_tensor(shape, dtype, weight_kind):
    x = torch.randn(shape, dtype=dtype, device=device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=device)

    ref_x = x.clone()
    ref_w = w.clone()

    ref_out = torch.ops.aten.prelu(ref_x, ref_w)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.prelu(x, w)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== rrelu_with_noise ==========
@label("rrelu_with_noise")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("training", [False, True])
@parametrize("bounds", [(0.125, 1.0 / 3.0), (0.1, 0.4)])
def test_rrelu_with_noise_tensor(shape, dtype, training, bounds):
    lower, upper = bounds
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_input = x.clone()
    act_input = x.clone()
    ref_noise = torch.empty_like(ref_input)
    act_noise = torch.empty_like(act_input)

    if training:
        ref_gen = torch.Generator(device=device)
        ref_gen.manual_seed(123)
        act_gen = torch.Generator(device=device)
        act_gen.manual_seed(123)
    else:
        ref_gen = None
        act_gen = None

    ref_out = torch.ops.aten.rrelu_with_noise(ref_input, ref_noise, lower, upper, training, ref_gen)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rrelu_with_noise(act_input, act_noise, lower, upper, training, act_gen)

    assert_close(act_out, ref_out, dtype=dtype)


@label("rrelu_with_noise")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("training", [False, True])
@parametrize("bounds", [(0.125, 1.0 / 3.0), (0.1, 0.4)])
def test_rrelu_with_noise_out(shape, dtype, training, bounds):
    lower, upper = bounds
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_input = x.clone()
    act_input = x.clone()
    ref_noise = torch.empty_like(ref_input)
    act_noise = torch.empty_like(act_input)
    ref_out_buf = torch.empty_like(ref_input)
    act_out_buf = torch.empty_like(act_input)

    if training:
        ref_gen = torch.Generator(device=device)
        ref_gen.manual_seed(123)
        act_gen = torch.Generator(device=device)
        act_gen.manual_seed(123)
    else:
        ref_gen = None
        act_gen = None

    torch.ops.aten.rrelu_with_noise.out(ref_input, ref_noise, lower, upper, training, ref_gen, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.rrelu_with_noise.out(act_input, act_noise, lower, upper, training, act_gen, out=act_out_buf)

    assert_close(act_out_buf, ref_out_buf, dtype=dtype)

# ========== square ==========
@label("square")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_square_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)

    ref_x = x.clone()
    ref_out = torch.ops.aten.square(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.square(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("square")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("out_layout", ["contiguous", "noncontiguous"])
def test_square_out(shape, dtype, out_layout):
    x = torch.randn(shape, dtype=dtype, device=device)

    if out_layout == "contiguous":
        ref_out_buf = torch.empty_like(x)
        act_out_buf = torch.empty_like(x)
    else:
        ref_base = torch.empty((shape[0], shape[1] * 2), dtype=dtype, device=device)
        act_base = torch.empty((shape[0], shape[1] * 2), dtype=dtype, device=device)
        ref_out_buf = ref_base[:, ::2]
        act_out_buf = act_base[:, ::2]

    ref_res = torch.ops.aten.square.out(x.clone(), out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_res = torch.ops.aten.square.out(x, out=act_out_buf)

    assert_close(act_res, ref_res, dtype=dtype)

# ========== affine_grid_generator ==========
@label("affine_grid_generator")
@parametrize("size", [
    [1, 3, 4, 4],
    [2, 3, 64, 64],
    [4, 3, 128, 256],
    [1, 2, 4, 4, 4],
    [2, 1, 8, 16, 16],
    [1, 1, 16, 32, 32],
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("align_corners", [True, False])
def test_affine_grid_generator_tensor(size, dtype, align_corners):
    k = len(size) - 2
    N = size[0]
    theta_shape = (N, k, k + 1)

    theta = torch.randn(theta_shape, dtype=dtype, device=device)
    ref_theta = theta.clone()
    act_theta = theta.clone()

    ref_out = torch.ops.aten.affine_grid_generator(ref_theta, size, align_corners)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.affine_grid_generator(act_theta, size, align_corners)

    assert_close(act_out, ref_out, dtype=dtype)


@label("affine_grid_generator")
@parametrize("size", [
    [1, 3, 4, 4],
    [2, 3, 64, 64],
    [4, 3, 128, 256],
    [1, 2, 4, 4, 4],
    [2, 1, 8, 16, 16],
    [1, 1, 16, 32, 32],
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("align_corners", [True, False])
def test_affine_grid_generator_out(size, dtype, align_corners):
    device = device
    k = len(size) - 2
    N = size[0]
    theta_shape = (N, k, k + 1)
    out_shape = [size[0]] + size[2:] + [k]

    theta = torch.randn(theta_shape, dtype=dtype, device=device)
    ref_theta = theta.clone()
    act_theta = theta.clone()

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.affine_grid_generator.out(ref_theta, size, align_corners, out=ref_out_buf)

    act_out_buf = torch.empty_like(ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.affine_grid_generator.out(act_theta, size, align_corners, out=act_out_buf)

    assert_close(act_out_buf, ref_out_buf, dtype=dtype)

# ========== bernoulli ==========
@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bernoulli_self(shape, dtype):
    probs = torch.rand(shape, dtype=dtype, device=device)
    ref_input = probs.clone()
    act_input = probs.clone()

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(1234)
    ref_out = torch.ops.aten.bernoulli(ref_input, generator=gen_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(1234)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.bernoulli(act_input, generator=gen_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bernoulli_out(shape, dtype):
    probs = torch.rand(shape, dtype=dtype, device=device)
    ref_input = probs.clone()
    act_input = probs.clone()

    out_ref = torch.empty(shape, dtype=dtype, device=device)
    out_act = torch.empty(shape, dtype=dtype, device=device)

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(4321)
    ref_out = torch.ops.aten.bernoulli.out(ref_input, generator=gen_ref, out=out_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(4321)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.bernoulli.out(act_input, generator=gen_act, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("p", [0.1, 0.5, 0.9])
def test_bernoulli_p_scalar(shape, dtype, p):
    tmpl = torch.empty(shape, dtype=dtype, device=device)
    ref_tmpl = tmpl.clone()
    act_tmpl = tmpl.clone()

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(1111)
    ref_out = torch.ops.aten.bernoulli.p(ref_tmpl, float(p), generator=gen_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(1111)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.bernoulli.p(act_tmpl, float(p), generator=gen_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bernoulli_tensor_p(shape, dtype):
    tmpl = torch.empty(shape, dtype=dtype, device=device)
    p_tensor = torch.rand(shape, dtype=torch.float32, device=device)
    ref_tmpl = tmpl.clone()
    act_tmpl = tmpl.clone()
    ref_p = p_tensor.clone()
    act_p = p_tensor.clone()

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(2222)
    ref_out = torch.ops.aten.bernoulli.Tensor(ref_tmpl, ref_p, generator=gen_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(2222)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.bernoulli.Tensor(act_tmpl, act_p, generator=gen_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bernoulli_tensor_out(shape, dtype):
    tmpl = torch.empty(shape, dtype=dtype, device=device)
    p_tensor = torch.rand(shape, dtype=torch.float32, device=device)
    ref_tmpl = tmpl.clone()
    act_tmpl = tmpl.clone()
    ref_p = p_tensor.clone()
    act_p = p_tensor.clone()

    out_ref = torch.empty(shape, dtype=dtype, device=device)
    out_act = torch.empty(shape, dtype=dtype, device=device)

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(3333)
    ref_out = torch.ops.aten.bernoulli.Tensor_out(ref_tmpl, ref_p, generator=gen_ref, out=out_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(3333)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.bernoulli.Tensor_out(act_tmpl, act_p, generator=gen_act, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("bernoulli")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("p", [None, 0.3, 0.8])
def test_bernoulli_float_out(shape, dtype, p):
    tmpl = torch.empty(shape, dtype=dtype, device=device)
    ref_tmpl = tmpl.clone()
    act_tmpl = tmpl.clone()

    out_ref = torch.empty(shape, dtype=dtype, device=device)
    out_act = torch.empty(shape, dtype=dtype, device=device)

    gen_ref = torch.Generator(device=device)
    gen_ref.manual_seed(4444)
    if p is None:
        ref_out = torch.ops.aten.bernoulli.float_out(ref_tmpl, generator=gen_ref, out=out_ref)
    else:
        ref_out = torch.ops.aten.bernoulli.float_out(ref_tmpl, float(p), generator=gen_ref, out=out_ref)

    gen_act = torch.Generator(device=device)
    gen_act.manual_seed(4444)
    with flagbench.use_gems(REGISTERED_OPS):
        if p is None:
            act_out = torch.ops.aten.bernoulli.float_out(act_tmpl, generator=gen_act, out=out_act)
        else:
            act_out = torch.ops.aten.bernoulli.float_out(act_tmpl, float(p), generator=gen_act, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== empty_strided ==========
@label("empty_strided")
@parametrize("size", [(1024,), (2, 3), (128, 256), (512, 512), (8, 16, 4)])
@parametrize("stride_kind", ["contig", "reversed", "padded2"])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_strided(size, stride_kind, dtype):
    def compute_stride(sz, mode):
        n = len(sz)
        # contiguous
        contig = [0] * n
        r = 1
        for i in range(n - 1, -1, -1):
            contig[i] = r
            r *= int(sz[i])
        if mode == "contig":
            return tuple(contig)
        if mode == "padded2":
            return tuple(int(s) * 2 for s in contig)
        if mode == "reversed":
            order = list(range(n - 1, -1, -1))
            s2 = [0] * n
            s2[order[-1]] = 1
            for i in range(n - 2, -1, -1):
                dim = order[i]
                next_dim = order[i + 1]
                s2[dim] = s2[next_dim] * int(sz[next_dim])
            return tuple(s2)
        return tuple(contig)

    stride = compute_stride(size, stride_kind)

    ref_out = torch.ops.aten.empty_strided(size, stride, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.empty_strided(size, stride, dtype=dtype, device=device)

    assert ref_out.size() == act_out.size()
    assert ref_out.stride() == act_out.stride()
    assert ref_out.dtype == act_out.dtype
    assert ref_out.device == act_out.device

    base = torch.randn(size, dtype=dtype, device=device)
    ref_out.copy_(base)
    act_out.copy_(base)

    assert_close(act_out, ref_out, dtype=dtype)


@label("empty_strided")
@parametrize("size", [(1024,), (2, 3), (128, 256), (512, 512), (8, 16, 4)])
@parametrize("stride_kind", ["contig", "reversed", "padded2"])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_empty_strided_out(size, stride_kind, dtype):
    def compute_stride(sz, mode):
        n = len(sz)
        contig = [0] * n
        r = 1
        for i in range(n - 1, -1, -1):
            contig[i] = r
            r *= int(sz[i])
        if mode == "contig":
            return tuple(contig)
        if mode == "padded2":
            return tuple(int(s) * 2 for s in contig)
        if mode == "reversed":
            order = list(range(n - 1, -1, -1))
            s2 = [0] * n
            s2[order[-1]] = 1
            for i in range(n - 2, -1, -1):
                dim = order[i]
                next_dim = order[i + 1]
                s2[dim] = s2[next_dim] * int(sz[next_dim])
            return tuple(s2)
        return tuple(contig)

    stride = compute_stride(size, stride_kind)

    ref_out_tensor = torch.empty(0, device=device, dtype=dtype)
    act_out_tensor = torch.empty(0, device=device, dtype=dtype)

    ref_out = torch.ops.aten.empty_strided.out(size, stride, out=ref_out_tensor)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.empty_strided.out(size, stride, out=act_out_tensor)

    assert ref_out.size() == act_out.size()
    assert ref_out.stride() == act_out.stride()
    assert ref_out.dtype == act_out.dtype
    assert ref_out.device == act_out.device

    base = torch.randn(size, dtype=dtype, device=device)
    ref_out.copy_(base)
    act_out.copy_(base)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== new_empty_strided ==========
@label("new_empty_strided")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("stride_kind", ["contig", "revcontig", "expanded"])
@parametrize("pass_dtype", [True, False])
def test_new_empty_strided_tensor(shape, dtype, stride_kind, pass_dtype):
    # prepare self tensor
    self_ref = torch.randn((1,), dtype=dtype, device=device)
    self_act = self_ref.clone()

    # compute stride based on kind
    size = list(shape)
    # contiguous
    contig = []
    acc = 1
    for dim in reversed(size):
        contig.insert(0, acc)
        acc *= dim
    if stride_kind == "contig":
        stride = contig
    elif stride_kind == "revcontig":
        # column-major like
        rev = []
        acc2 = 1
        for dim in size:
            rev.append(acc2)
            acc2 *= dim
        stride = rev
    else:
        stride = [s * 2 for s in contig]

    dtype_arg = dtype if pass_dtype else None

    ref_out = torch.ops.aten.new_empty_strided(self_ref, size, stride, dtype=dtype_arg)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.new_empty_strided(self_act, size, stride, dtype=dtype_arg)

    # fill both outputs with deterministic pattern
    numel = 1
    for s in size:
        numel *= s
    pattern = torch.arange(numel, dtype=dtype, device=device).reshape(size)
    ref_out.copy_(pattern)
    act_out.copy_(pattern)

    assert_close(act_out, ref_out, dtype=dtype)


@label("new_empty_strided")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("stride_kind", ["contig", "revcontig", "expanded"])
def test_new_empty_strided_out(shape, dtype, stride_kind):
    # prepare self tensor
    self_ref = torch.randn((1,), dtype=dtype, device=device)
    self_act = self_ref.clone()

    # compute stride based on kind
    size = list(shape)
    contig = []
    acc = 1
    for dim in reversed(size):
        contig.insert(0, acc)
        acc *= dim
    if stride_kind == "contig":
        stride = contig
    elif stride_kind == "revcontig":
        rev = []
        acc2 = 1
        for dim in size:
            rev.append(acc2)
            acc2 *= dim
        stride = rev
    else:
        stride = [s * 2 for s in contig]

    ref_out_buf = torch.empty((1,), dtype=dtype, device=device)
    act_out_buf = torch.empty((1,), dtype=dtype, device=device)

    ref_out = torch.ops.aten.new_empty_strided.out(self_ref, size, stride, out=ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.new_empty_strided.out(self_act, size, stride, out=act_out_buf)

    # fill both outputs with deterministic pattern
    numel = 1
    for s in size:
        numel *= s
    pattern = torch.arange(numel, dtype=dtype, device=device).reshape(size)
    ref_out.copy_(pattern)
    act_out.copy_(pattern)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== new_ones ==========
@label("new_ones")
@parametrize("self_shape", [(2, 3), (128, 256)])
@parametrize("size", [(2, 3), (128, 256), (32, 16, 8), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_new_ones_default(self_shape, size, dtype):
    self_tensor = torch.randn(self_shape, dtype=torch.float32, device=device)

    ref_self = self_tensor.clone()
    ref_out = torch.ops.aten.new_ones(ref_self, size, dtype=dtype)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.new_ones(self_tensor, size, dtype=dtype)

    assert_close(act_out, ref_out, dtype=dtype)


@label("new_ones")
@parametrize("self_shape", [(2, 3), (64, 64)])
@parametrize("size", [(2, 3), (128, 256), (16, 8, 4), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_new_ones_out(self_shape, size, dtype):
    self_tensor = torch.randn(self_shape, dtype=torch.float32, device=device)

    ref_self = self_tensor.clone()
    ref_out_buf = torch.empty(size, device=device, dtype=dtype)
    ref_out = torch.ops.aten.new_ones.out(ref_self, size, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out_buf = torch.empty(size, device=device, dtype=dtype)
        act_out = torch.ops.aten.new_ones.out(self_tensor, size, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== poisson ==========
@label("poisson")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("use_gen", [True, False])
def test_poisson_tensor(shape, dtype, use_gen):
    rates = torch.rand(shape, device=device, dtype=torch.float32) * 10.0
    rates = rates.to(dtype)

    ref_input = rates.clone()
    act_input = rates.clone()

    if use_gen:
        base_gen = torch.Generator(device=device)
        base_gen.manual_seed(1234)
        ref_gen = torch.Generator(device=device)
        ref_gen.set_state(base_gen.get_state())
        act_gen = torch.Generator(device=device)
        act_gen.set_state(base_gen.get_state())

        ref_out = torch.ops.aten.poisson(ref_input, ref_gen)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.poisson(act_input, act_gen)
    else:
        rng_state = torch.cuda.get_rng_state()
        ref_out = torch.ops.aten.poisson(ref_input)
        torch.cuda.set_rng_state(rng_state)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.poisson(act_input)

    assert_close(act_out, ref_out, dtype=dtype)


@label("poisson")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("use_gen", [True, False])
def test_poisson_out(shape, dtype, use_gen):
    rates = torch.rand(shape, device=device, dtype=torch.float32) * 10.0
    rates = rates.to(dtype)

    ref_input = rates.clone()
    act_input = rates.clone()

    out_ref = torch.empty_like(ref_input)
    out_act = torch.empty_like(act_input)

    if use_gen:
        base_gen = torch.Generator(device=device)
        base_gen.manual_seed(5678)
        ref_gen = torch.Generator(device=device)
        ref_gen.set_state(base_gen.get_state())
        act_gen = torch.Generator(device=device)
        act_gen.set_state(base_gen.get_state())

        torch.ops.aten.poisson.out(ref_input, ref_gen, out=out_ref)
        with flagbench.use_gems(REGISTERED_OPS):
            torch.ops.aten.poisson.out(act_input, act_gen, out=out_act)
    else:
        rng_state = torch.cuda.get_rng_state()
        torch.ops.aten.poisson.out(ref_input, out=out_ref)
        torch.cuda.set_rng_state(rng_state)
        with flagbench.use_gems(REGISTERED_OPS):
            torch.ops.aten.poisson.out(act_input, out=out_act)

    assert_close(out_act, out_ref, dtype=dtype)

# ========== scalar_tensor ==========
@label("scalar_tensor")
@parametrize("val", [-7, -1, 0, 3, 12345, -1.5, 2.75, 3.14159])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scalar_tensor_default(val, dtype):
    ref_out = torch.ops.aten.scalar_tensor(val, dtype=dtype, device=device, layout=None, pin_memory=None)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.scalar_tensor(val, dtype=dtype, device=device, layout=None, pin_memory=None)
    assert_close(act_out, ref_out, dtype=dtype)


@label("scalar_tensor")
@parametrize("val", [-7, -1, 0, 3, 12345, -1.5, 2.75, 3.14159])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scalar_tensor_out(val, dtype):
    ref_out_buf = torch.empty((), dtype=dtype, device=device)
    act_out_buf = torch.empty((), dtype=dtype, device=device)

    ref_out = torch.ops.aten.scalar_tensor.out(val, out=ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.scalar_tensor.out(val, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== acosh ==========
@label("acosh")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_acosh_tensor(shape, dtype):
    base = torch.rand(shape, device=device, dtype=torch.float32)
    input_tensor = (base * 9.0 + 1.0).to(dtype)

    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.acosh(ref_input)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.acosh(input_tensor)

    assert_close(act_out, ref_out, dtype=dtype)


@label("acosh")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_acosh_out_tensor(shape, dtype):
    base = torch.rand(shape, device=device, dtype=torch.float32)
    input_tensor = (base * 9.0 + 1.0).to(dtype)

    ref_input = input_tensor.clone()
    out_ref = torch.empty_like(ref_input)
    ref_out = torch.ops.aten.acosh.out(ref_input, out=out_ref)

    out_act = torch.empty_like(input_tensor)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.acosh.out(input_tensor, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)


@label("acosh")
@parametrize("a", [1, 2, 10, 256])
def test_acosh_int(a):
    dtype = torch.float32
    ref_out = torch.ops.aten.acosh.int(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.acosh.int(a)
    assert_close(act_out, ref_out, dtype=dtype)


@label("acosh")
@parametrize("a", [1.0, 1.5, 3.0, 10.0, 100.0])
def test_acosh_float(a):
    dtype = torch.float32
    ref_out = torch.ops.aten.acosh.float(a)
    with flagbench.use_gems(REGISTERED_OPOPs):
        act_out = torch.ops.aten.acosh.float(a)
    assert_close(act_out, ref_out, dtype=dtype)


@label("acosh")
@parametrize("a", [complex(1.5, 0.2), complex(3.0, -0.5), complex(10.0, 0.0), complex(2.0, 2.0)])
def test_acosh_complex(a):
    dtype = torch.complex64
    ref_out = torch.ops.aten.acosh.complex(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.acosh.complex(a)
    assert_close(act_out, ref_out, dtype=dtype)


@label("acosh")
@parametrize("a,dtype", [(True, torch.float32), (2.0, torch.float32)])
def test_acosh_scalar(a, dtype):
    ref_out = torch.ops.aten.acosh.Scalar(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.acosh.Scalar(a)
    assert_close(act_out, ref_out, dtype=dtype)

# ========== asin ==========
@label("asin")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_asin_tensor(shape, dtype):
    x = (torch.rand(shape, device=device, dtype=dtype) * 2) - 1
    ref_x = x.clone()

    ref_out = torch.ops.aten.asin(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.asin(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("asin")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_asin_out(shape, dtype):
    x = (torch.rand(shape, device=device, dtype=dtype) * 2) - 1
    ref_x = x.clone()

    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.asin.out(ref_x, out=ref_out)

    act_out = torch.empty_like(x)
    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.asin.out(x, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)


@label("asin")
@parametrize("a", [-1, 0, 1])
def test_asin_int(a):
    ref_out = torch.ops.aten.asin.int(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.asin.int(a)
    assert_close(act_out, ref_out, dtype=torch.float64)


@label("asin")
@parametrize("a", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_asin_float(a):
    ref_out = torch.ops.aten.asin.float(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.asin.float(a)
    assert_close(act_out, ref_out, dtype=torch.float64)


@label("asin")
@parametrize("a", [0 + 0j, 0.5 + 0.5j, -0.5 + 0.75j])
def test_asin_complex(a):
    ref_out = torch.ops.aten.asin.complex(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.asin.complex(a)
    assert_close(act_out, ref_out, dtype=torch.complex128)


@label("asin")
@parametrize("a", [0, 1.0, -0.5, 0.5 + 0.25j])
def test_asin_scalar(a):
    ref_out = torch.ops.aten.asin.Scalar(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.asin.Scalar(a)
    dtype = torch.complex128 if isinstance(a, complex) else torch.float64
    assert_close(act_out, ref_out, dtype=dtype)

# ========== cosh ==========
@label("cosh")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cosh_tensor(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=device).uniform_(-3, 3)
    ref_x = x.clone()

    ref_out = torch.ops.aten.cosh(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.cosh(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("cosh")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cosh_out_tensor(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=device).uniform_(-3, 3)
    ref_x = x.clone()

    ref_out_buf = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.cosh.out(ref_x, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out_buf = torch.empty_like(x)
        act_out = torch.ops.aten.cosh.out(x, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)


@label("cosh")
@parametrize("a", [-3, -1, 0, 1, 7])
def test_cosh_int_scalar(a):
    ref_out = torch.ops.aten.cosh(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.cosh(a)

    assert_close(act_out, ref_out, dtype=torch.float64)


@label("cosh")
@parametrize("a", [-2.5, -0.75, 0.0, 0.75, 3.25])
def test_cosh_float_scalar(a):
    ref_out = torch.ops.aten.cosh(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.cosh(a)

    assert_close(act_out, ref_out, dtype=torch.float64)


@label("cosh")
@parametrize("a", [complex(0.0, 0.0), complex(1.0, 0.5), complex(-2.0, -1.0), complex(0.5, -1.5)])
def test_cosh_complex_scalar(a):
    ref_out = torch.ops.aten.cosh(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.cosh(a)

    assert_close(act_out, ref_out, dtype=torch.complex128)


@label("cosh")
@parametrize("a", [True, False])
def test_cosh_Scalar_bool(a):
    ref_out = torch.ops.aten.cosh(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.cosh(a)

    assert_close(act_out, ref_out, dtype=torch.float64)

# ========== floor ==========
@label("floor")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_floor_tensor(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.floor(ref_input)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.floor(input_tensor)

    assert_close(act_out, ref_out, dtype=dtype)


@label("floor")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_floor_out(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ref_input = input_tensor.clone()

    ref_out = torch.empty_like(ref_input)
    act_out = torch.empty_like(input_tensor)

    torch.ops.aten.floor.out(ref_input, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.floor.out(input_tensor, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)


@label("floor")
@parametrize("a", [0, 1, -1, 42, -37, 123456])
def test_floor_int(a):
    ref_out = torch.ops.aten.floor.int(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.floor.int(a)

    assert_close(act_out, ref_out, dtype=torch.int64)


@label("floor")
@parametrize("a", [0.0, 0.5, -0.5, 3.7, -2.3, 123.999])
def test_floor_float(a):
    ref_out = torch.ops.aten.floor.float(a)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.floor.float(a)

    assert_close(act_out, ref_out, dtype=torch.int64)


@label("floor")
@parametrize("value,dtype", [(0.0, torch.float32), (3.7, torch.float32), (-2.3, torch.float32), (5, torch.int64), (-7, torch.int64)])
def test_floor_scalar(value, dtype):
    ref_out = torch.ops.aten.floor.Scalar(value)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.floor.Scalar(value)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== i0 ==========
@label("i0")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_i0_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=torch.float32, device=device).to(dtype) * 3.0
    ref_inp = inp.clone()

    ref_out = torch.ops.aten.i0(ref_inp)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.i0(inp)

    assert_close(act_out, ref_out, dtype=dtype)


@label("i0")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_i0_out(shape, dtype):
    inp = torch.randn(shape, dtype=torch.float32, device=device).to(dtype) * 3.0

    ref_inp = inp.clone()
    ref_out = torch.empty(shape, dtype=dtype, device=device)
    torch.ops.aten.i0.out(ref_inp, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_inp = inp
        act_out = torch.empty(shape, dtype=dtype, device=device)
        torch.ops.aten.i0.out(act_inp, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== polygamma ==========
@label("polygamma")
@parametrize("n", [0, 1, 2])
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_polygamma_tensor(shape, dtype, n):
    x = torch.rand(shape, device=device, dtype=dtype) * 4.0 + 0.5
    ref_x = x.clone()
    ref_out = torch.ops.aten.polygamma(n, ref_x)
    act_x = x.clone()
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.polygamma(n, act_x)
    assert_close(act_out, ref_out, dtype=dtype)


@label("polygamma")
@parametrize("n", [0, 1, 2])
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_polygamma_out(shape, dtype, n):
    x = torch.rand(shape, device=device, dtype=dtype) * 4.0 + 0.5
    ref_x = x.clone()
    out_ref = torch.empty_like(ref_x)
    torch.ops.aten.polygamma.out(n, ref_x, out=out_ref)
    act_x = x.clone()
    out_act = torch.empty_like(act_x)
    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.polygamma.out(n, act_x, out=out_act)
    assert_close(out_act, out_ref, dtype=dtype)

# ========== rsub ==========
@label("rsub")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha", [1.0, 0.5])
def test_rsub_tensor(shape, dtype, alpha):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out = torch.ops.aten.rsub(ref_self, ref_other, alpha=alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rsub(self, other, alpha=alpha)

    assert_close(act_out, ref_out, dtype=dtype)


@label("rsub")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("other", [-2.0, 0.0, 3.25])
@parametrize("alpha", [1.0, 0.5])
def test_rsub_scalar(shape, dtype, other, alpha):
    self = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()

    ref_out = torch.ops.aten.rsub(ref_self, other, alpha=alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rsub(self, other, alpha=alpha)

    assert_close(act_out, ref_out, dtype=dtype)


@label("rsub")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha", [1.0, 0.5])
def test_rsub_tensor_out(shape, dtype, alpha):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out_buf = torch.empty(shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.rsub(ref_self, ref_other, alpha=alpha, out=ref_out_buf)

    act_out_buf = torch.empty(shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rsub(self, other, alpha=alpha, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)


@label("rsub")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("other", [-2.0, 0.0, 3.25])
@parametrize("alpha", [1.0, 0.5])
def test_rsub_scalar_out(shape, dtype, other, alpha):
    self = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()

    ref_out_buf = torch.empty(shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.rsub(ref_self, other, alpha=alpha, out=ref_out_buf)

    act_out_buf = torch.empty(shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rsub(self, other, alpha=alpha, out=act_out_buf)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== sgn ==========
@label("sgn")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sgn_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.sgn(ref_x)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.sgn(x)

    assert_close(act_out, ref_out, dtype=dtype)


@label("sgn")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sgn_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.empty_like(ref_x)
    act_out = torch.empty_like(x)

    ref_ret = torch.ops.aten.sgn.out(ref_x, out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_ret = torch.ops.aten.sgn.out(x, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== special_entr ==========
@label("special_entr")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_entr_tensor(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=device) - 0.2
    flat = x.view(-1)
    if flat.numel() > 0:
        flat[0] = 0
    ref_input = x.clone()
    ref_out = torch.ops.aten.special_entr(ref_input)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.special_entr(x.clone())
    assert_close(act_out, ref_out, dtype=dtype)


@label("special_entr")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_special_entr_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=device) - 0.2
    flat = x.view(-1)
    if flat.numel() > 0:
        flat[0] = 0
    ref_input = x.clone()
    ref_out = torch.empty_like(ref_input)
    ref_out = torch.ops.aten.special_entr.out(ref_input, out=ref_out)
    act_out = torch.empty_like(x)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.special_entr.out(x.clone(), out=act_out)
    assert_close(act_out, ref_out, dtype=dtype)

# ========== amin ==========
@label("amin")
@parametrize("shape", [(2, 3), (128, 256), (512, 320)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [None, 0, 1, -1, [0, 1]])
@parametrize("keepdim", [False, True])
def test_amin_tensor_reduce_2d(shape, dtype, dim, keepdim):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    if dim is None and not keepdim:
        ref_out = torch.ops.aten.amin(ref_x)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin(x)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        ref_out = torch.ops.aten.amin(ref_x, use_dim, keepdim)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin(x, use_dim, keepdim)

    assert_close(act_out, ref_out, dtype=dtype)


@label("amin")
@parametrize("shape", [(2, 3, 4), (16, 17, 8), (32, 64, 128)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [None, 0, 1, 2, -1, [0, 2], [1, 2], [0, 1, 2]])
@parametrize("keepdim", [False, True])
def test_amin_tensor_reduce_3d(shape, dtype, dim, keepdim):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    if dim is None and not keepdim:
        ref_out = torch.ops.aten.amin(ref_x)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin(x)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        ref_out = torch.ops.aten.amin(ref_x, use_dim, keepdim)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin(x, use_dim, keepdim)

    assert_close(act_out, ref_out, dtype=dtype)


@label("amin")
@parametrize("shape", [(2, 3), (128, 256), (512, 320)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [None, 0, 1, -1, [0, 1]])
@parametrize("keepdim", [False, True])
def test_amin_out_reduce_2d(shape, dtype, dim, keepdim):
    def compute_out_shape(shape, dims, keepdim):
        if dims is None:
            # reduce-all default with keepdim=False
            return ()
        if isinstance(dims, int):
            dims = [dims]
        dims = [(d + len(shape)) % len(shape) for d in dims]
        if keepdim:
            out_shape = list(shape)
            for d in dims:
                out_shape[d] = 1
            return tuple(out_shape)
        else:
            remaining = [i for i in range(len(shape)) if i not in set(dims)]
            return tuple(shape[i] for i in remaining)

    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    if dim is None and not keepdim:
        out_shape = compute_out_shape(shape, None, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=device)

        ref_out = torch.ops.aten.amin.out(ref_x, out=ref_out_t)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin.out(x, out=act_out_t)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        out_shape = compute_out_shape(shape, use_dim, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=device)

        ref_out = torch.ops.aten.amin.out(ref_x, use_dim, keepdim, out=ref_out_t)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin.out(x, use_dim, keepdim, out=act_out_t)

    assert_close(act_out, ref_out, dtype=dtype)


@label("amin")
@parametrize("shape", [(2, 3, 4), (16, 17, 8), (32, 64, 128)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [None, 0, 1, 2, -1, [0, 2], [1, 2], [0, 1, 2]])
@parametrize("keepdim", [False, True])
def test_amin_out_reduce_3d(shape, dtype, dim, keepdim):
    def compute_out_shape(shape, dims, keepdim):
        if dims is None:
            return ()
        if isinstance(dims, int):
            dims = [dims]
        dims = [(d + len(shape)) % len(shape) for d in dims]
        if keepdim:
            out_shape = list(shape)
            for d in dims:
                out_shape[d] = 1
            return tuple(out_shape)
        else:
            remaining = [i for i in range(len(shape)) if i not in set(dims)]
            return tuple(shape[i] for i in remaining)

    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    if dim is None and not keepdim:
        out_shape = compute_out_shape(shape, None, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=device)

        ref_out = torch.ops.aten.amin.out(ref_x, out=ref_out_t)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin.out(x, out=act_out_t)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        out_shape = compute_out_shape(shape, use_dim, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=device)

        ref_out = torch.ops.aten.amin.out(ref_x, use_dim, keepdim, out=ref_out_t)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.amin.out(x, use_dim, keepdim, out=act_out_t)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== binary_cross_entropy_with_logits ==========
@label("binary_cross_entropy_with_logits")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("use_weight", [False, True])
@parametrize("use_pos_weight", [False, True])
def test_binary_cross_entropy_with_logits_tensor(shape, dtype, reduction, use_weight, use_pos_weight):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    weight = (torch.rand(shape, dtype=dtype, device=device) + 0.5) if use_weight else None
    pos_weight = (torch.rand(shape, dtype=dtype, device=device) + 0.5) if use_pos_weight else None

    ref_self = self.clone()
    ref_target = target.clone()
    ref_weight = weight.clone() if weight is not None else None
    ref_pos_weight = pos_weight.clone() if pos_weight is not None else None

    ref_out = torch.ops.aten.binary_cross_entropy_with_logits(ref_self, ref_target, ref_weight, ref_pos_weight, reduction)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction)

    assert_close(act_out, ref_out, dtype=dtype)


@label("binary_cross_entropy_with_logits")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("use_weight", [False, True])
@parametrize("use_pos_weight", [False, True])
def test_binary_cross_entropy_with_logits_out(shape, dtype, reduction, use_weight, use_pos_weight):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = torch.rand(shape, dtype=dtype, device=device)
    weight = (torch.rand(shape, dtype=dtype, device=device) + 0.5) if use_weight else None
    pos_weight = (torch.rand(shape, dtype=dtype, device=device) + 0.5) if use_pos_weight else None

    out_shape = shape if reduction == 0 else ()
    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_target = target.clone()
    ref_weight = weight.clone() if weight is not None else None
    ref_pos_weight = pos_weight.clone() if pos_weight is not None else None

    ref_out = torch.ops.aten.binary_cross_entropy_with_logits.out(
        ref_self, ref_target, ref_weight, ref_pos_weight, reduction, out=ref_out_buf
    )

    act_out_buf = torch.empty(out_shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.binary_cross_entropy_with_logits.out(
            self, target, weight, pos_weight, reduction, out=act_out_buf
        )

    assert_close(act_out, ref_out, dtype=dtype)

# ========== fmax ==========
@label("fmax")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fmax_tensor(shape, dtype):
    self_base = torch.randn(shape, dtype=dtype, device=device)
    other_base = torch.randn(shape, dtype=dtype, device=device)

    if self_base.numel() >= 1:
        self_base.view(-1)[0] = float("nan")
    if other_base.numel() >= 2:
        other_base.view(-1)[1] = float("nan")

    ref_self = self_base.clone()
    ref_other = other_base.clone()
    act_self = self_base.clone()
    act_other = other_base.clone()

    ref_out = torch.ops.aten.fmax(ref_self, ref_other)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.fmax(act_self, act_other)

    assert_close(act_out, ref_out, dtype=dtype, equal_nan=True)


@label("fmax")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fmax_out(shape, dtype):
    self_base = torch.randn(shape, dtype=dtype, device=device)
    other_base = torch.randn(shape, dtype=dtype, device=device)

    if self_base.numel() >= 1:
        self_base.view(-1)[0] = float("nan")
    if other_base.numel() >= 2:
        other_base.view(-1)[1] = float("nan")

    ref_self = self_base.clone()
    ref_other = other_base.clone()
    act_self = self_base.clone()
    act_other = other_base.clone()

    ref_out_tensor = torch.empty_like(ref_self)
    act_out_tensor = torch.empty_like(act_self)

    ref_out = torch.ops.aten.fmax.out(ref_self, ref_other, out=ref_out_tensor)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.fmax.out(act_self, act_other, out=act_out_tensor)

    assert_close(act_out, ref_out, dtype=dtype, equal_nan=True)

# ========== huber_loss ==========
@label("huber_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_tensor(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device=device)
    target_tensor = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.ops.aten.huber_loss(ref_self, ref_target, reduction, float(delta))

    with flagbench.use_gems(REGISTERED_OPS):
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = torch.ops.aten.huber_loss(act_self, act_target, reduction, float(delta))

    assert_close(act_out, ref_out, dtype=dtype)


@label("huber_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_out(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device=device)
    target_tensor = torch.randn(shape, dtype=dtype, device=device)

    if reduction == 0:
        out_shape = shape
    else:
        out_shape = ()

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.empty(out_shape, dtype=dtype, device=device)
    torch.ops.aten.huber_loss.out(ref_self, ref_target, reduction, float(delta), out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = torch.empty(out_shape, dtype=dtype, device=device)
        torch.ops.aten.huber_loss.out(act_self, act_target, reduction, float(delta), out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== logaddexp2 ==========
@label("logaddexp2")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp2_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out = torch.ops.aten.logaddexp2(ref_self, ref_other)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.logaddexp2(self, other)

    assert_close(act_out, ref_out, dtype=dtype)


@label("logaddexp2")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logaddexp2_out(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out = torch.empty(shape, dtype=dtype, device=device)
    torch.ops.aten.logaddexp2.out(ref_self, ref_other, out=ref_out)

    act_out = torch.empty(shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.logaddexp2.out(self, other, out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== margin_ranking_loss ==========
@label("margin_ranking_loss")
@parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("margin", [0.0, 0.5, 1.0])
@parametrize("reduction", [0, 1, 2])
def test_margin_ranking_loss_tensor(shape, dtype, margin, reduction):
    input1 = torch.randn(shape, dtype=dtype, device=device)
    input2 = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(dtype)

    ref_input1 = input1.clone()
    ref_input2 = input2.clone()
    ref_target = target.clone()

    ref_out = torch.ops.aten.margin_ranking_loss(ref_input1, ref_input2, ref_target, margin, reduction)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.margin_ranking_loss(input1, input2, target, margin, reduction)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== pairwise_distance ==========
@label("pairwise_distance")
@parametrize("shape", [(2, 3), (128, 256), (512, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pairwise_distance_defaults(shape, dtype):
    x1 = torch.randn(shape, dtype=dtype, device=device)
    x2 = torch.randn(shape, dtype=dtype, device=device)

    ref_x1 = x1.clone()
    ref_x2 = x2.clone()

    ref_out = torch.ops.aten.pairwise_distance(ref_x1, ref_x2)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.pairwise_distance(x1, x2)

    assert_close(act_out, ref_out, dtype=dtype)


@label("pairwise_distance")
@parametrize("shape", [(2, 3), (128, 256), (512, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("p", [1.0, 2.0, 3.0])
@parametrize("eps", [1e-6, 1e-4])
@parametrize("keepdim", [False, True])
def test_pairwise_distance_explicit_args(shape, dtype, p, eps, keepdim):
    x1 = torch.randn(shape, dtype=dtype, device=device)
    x2 = torch.randn(shape, dtype=dtype, device=device)

    ref_x1 = x1.clone()
    ref_x2 = x2.clone()

    ref_out = torch.ops.aten.pairwise_distance(ref_x1, ref_x2, p, eps, keepdim)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.pairwise_distance(x1, x2, p, eps, keepdim)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== renorm ==========
@label("renorm")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("p", [1.0, 2.0])
@parametrize("dim", [0, 1])
@parametrize("maxnorm", [0.5, 1.0])
def test_renorm_tensor(shape, dtype, p, dim, maxnorm):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()
    ref_out = torch.ops.aten.renorm(ref_x, p, dim, maxnorm)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.renorm(x, p, dim, maxnorm)
    assert_close(act_out, ref_out, dtype=dtype)


@label("renorm")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("p", [1.0, 2.0])
@parametrize("dim", [0, 1])
@parametrize("maxnorm", [0.5, 1.0])
def test_renorm_out_tensor(shape, dtype, p, dim, maxnorm):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)
    ref_out = torch.ops.aten.renorm.out(ref_x, p, dim, maxnorm, out=ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.renorm.out(x, p, dim, maxnorm, out=act_out_buf)
    assert_close(act_out, ref_out, dtype=dtype)

# ========== soft_margin_loss ==========
@label("soft_margin_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_tensor(shape, dtype, reduction):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1

    ref_self = self.clone()
    ref_target = target.clone()
    ref_out = torch.ops.aten.soft_margin_loss(ref_self, ref_target, reduction)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.soft_margin_loss(self, target, reduction)

    assert_close(act_out, ref_out, dtype=dtype)


@label("soft_margin_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
def test_soft_margin_loss_out(shape, dtype, reduction):
    self = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1

    if reduction == 0:
        out_ref = torch.empty(shape, dtype=dtype, device=device)
        out_act = torch.empty(shape, dtype=dtype, device=device)
    else:
        out_ref = torch.empty((), dtype=dtype, device=device)
        out_act = torch.empty((), dtype=dtype, device=device)

    ref_self = self.clone()
    ref_target = target.clone()
    ref_out = torch.ops.aten.soft_margin_loss.out(ref_self, ref_target, reduction, out=out_ref)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.soft_margin_loss.out(self, target, reduction, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== as_strided ==========
@label("as_strided")
@parametrize("base_shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("case", ["contig", "transpose", "subsample", "offset_cropped", "reshape_like"])
def test_as_strided_2d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=device)
    ref_x = x.clone()

    s0, s1 = ref_x.stride()
    h, w = base_shape
    total = ref_x.numel()

    if case == "contig":
        size = [h, w]
        stride = [s0, s1]
        storage_offset = None
    elif case == "transpose":
        size = [w, h]
        stride = [s1, s0]
        storage_offset = None
    elif case == "subsample":
        step0 = 2 if h >= 2 else 1
        step1 = 2 if w >= 2 else 1
        size = [max(h // step0, 1), max(w // step1, 1)]
        stride = [s0 * step0, s1 * step1]
        storage_offset = None
    elif case == "offset_cropped":
        off = (s0 if h > 1 else 0) + (s1 if w > 1 else 0)
        size = [max(h - 1, 1), max(w - 1, 1)]
        stride = [s0, s1]
        storage_offset = off if off > 0 else 0
    elif case == "reshape_like":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = None
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.as_strided(x, size, stride, storage_offset)

    assert_close(act_out, ref_out, dtype=dtype)


@label("as_strided")
@parametrize("base_shape", [(6,), (64,), (1024,)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("case", ["contig", "skip_step", "offset_start", "reshape2d"])
def test_as_strided_1d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=device)
    ref_x = x.clone()

    (N,) = base_shape
    (s,) = ref_x.stride()
    total = N

    if case == "contig":
        size = [N]
        stride = [s]
        storage_offset = None
    elif case == "skip_step":
        step = 2 if N >= 2 else 1
        size = [max(N // step, 1)]
        stride = [s * step]
        storage_offset = None
    elif case == "offset_start":
        off = 1 if N > 1 else 0
        size = [max(N - off, 1)]
        stride = [s]
        storage_offset = off
    elif case == "reshape2d":
        if total % 4 == 0:
            k = 4
        elif total % 2 == 0:
            k = 2
        else:
            k = 1
        size = [total // k, k]
        stride = [k, 1]
        storage_offset = None
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.as_strided(x, size, stride, storage_offset)

    assert_close(act_out, ref_out, dtype=dtype)


@label("as_strided")
@parametrize("base_shape", [(2, 3, 4), (8, 16, 8), (16, 32, 16)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("case", ["contig", "swap_last_two", "subsample", "offset_plane"])
def test_as_strided_3d(base_shape, dtype, case):
    x = torch.randn(base_shape, dtype=dtype, device=device)
    ref_x = x.clone()

    d0, d1, d2 = base_shape
    s0, s1, s2 = ref_x.stride()

    if case == "contig":
        size = [d0, d1, d2]
        stride = [s0, s1, s2]
        storage_offset = None
    elif case == "swap_last_two":
        size = [d0, d2, d1]
        stride = [s0, s2, s1]
        storage_offset = None
    elif case == "subsample":
        step0 = 2 if d0 >= 2 else 1
        step1 = 2 if d1 >= 2 else 1
        step2 = 1
        size = [max(d0 // step0, 1), max(d1 // step1, 1), max(d2 // step2, 1)]
        stride = [s0 * step0, s1 * step1, s2 * step2]
        storage_offset = None
    elif case == "offset_plane":
        off = (s0 if d0 > 1 else 0) + (s2 if d2 > 1 else 0)
        size = [max(d0 - 1, 1), d1, max(d2 - 1, 1)]
        stride = [s0, s1, s2]
        storage_offset = off if off > 0 else 0
    else:
        raise AssertionError("unknown case")

    ref_out = torch.ops.aten.as_strided(ref_x, size, stride, storage_offset)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.as_strided(x, size, stride, storage_offset)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== im2col ==========
@label("im2col")
@parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_tensor(shape, dtype, kernel_size, dilation, padding, stride):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.im2col(ref_x, kernel_size, dilation, padding, stride)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.im2col(x, kernel_size, dilation, padding, stride)

    assert_close(act_out, ref_out, dtype=dtype)


@label("im2col")
@parametrize("shape", [(3, 8, 8), (16, 64, 64), (32, 128, 128)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "kernel_size, dilation, padding, stride",
    [
        ((3, 3), (1, 1), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (0, 0), (2, 2)),
        ((5, 4), (2, 2), (2, 1), (1, 2)),
        ((1, 1), (1, 1), (0, 0), (1, 1)),
    ],
)
def test_im2col_out(shape, dtype, kernel_size, dilation, padding, stride):
    def compute_out_shape(c, h, w, k, d, p, s):
        kH, kW = k
        dH, dW = d
        pH, pW = p
        sH, sW = s
        out_h = (h + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        out_w = (w + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return (c * kH * kW, out_h * out_w)

    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    C, H, W = shape
    out_shape = compute_out_shape(C, H, W, kernel_size, dilation, padding, stride)

    out_ref = torch.empty(out_shape, dtype=dtype, device=device)
    out_act = torch.empty(out_shape, dtype=dtype, device=device)

    ref_out = torch.ops.aten.im2col.out(ref_x, kernel_size, dilation, padding, stride, out=out_ref)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.im2col.out(x, kernel_size, dilation, padding, stride, out=out_act)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== reshape ==========
@label("reshape")
@parametrize(
    "in_shape,out_shape",
    [
        ((2, 3), (3, 2)),
        ((2, 3), (1, 6)),
        ((2, 3), (-1,)),
        ((128, 256), (256, 128)),
        ((128, 256), (32, -1)),
        ((128, 256), (-1, 128)),
        ((64, 64, 64), (-1,)),
        ((64, 64, 64), (64, 4096)),
        ((512, 512), (-1,)),
        ((256, 512), (512, 256)),
    ],
)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_contiguous(in_shape, out_shape, dtype):
    input_tensor = torch.randn(in_shape, dtype=dtype, device=device)
    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.reshape(ref_input, out_shape)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.reshape(input_tensor, out_shape)

    assert_close(act_out, ref_out, dtype=dtype)


@label("reshape")
@parametrize(
    "base_shape,out_shape,transform",
    [
        ((32, 64), (-1,), "transpose01"),
        ((64, 128), (128, 64), "transpose01"),
        ((8, 16, 32), (256, 16), "permute201"),
        ((64, 64, 64), (4096, 64), "permute120"),
    ],
)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_noncontiguous(base_shape, out_shape, transform, dtype):
    base = torch.randn(base_shape, dtype=dtype, device=device)
    ref_input = base.clone()
    act_input = base.clone()

    if transform == "transpose01":
        ref_input = ref_input.transpose(0, 1)
        act_input = act_input.transpose(0, 1)
    elif transform == "permute201":
        ref_input = ref_input.permute(2, 0, 1)
        act_input = act_input.permute(2, 0, 1)
    elif transform == "permute120":
        ref_input = ref_input.permute(1, 2, 0)
        act_input = act_input.permute(1, 2, 0)

    ref_out = torch.ops.aten.reshape(ref_input, out_shape)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.reshape(act_input, out_shape)

    assert_close(act_out, ref_out, dtype=dtype)

# ========== rot90 ==========
@label("rot90")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("k", [0, 1, 2, 3, -1])
@parametrize("dims", [[0, 1]])
def test_rot90_tensor_2d(shape, dtype, k, dims):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.clone()
    ref_out = torch.ops.aten.rot90(ref_inp, k=k, dims=dims)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rot90(inp, k=k, dims=dims)
    assert_close(act_out, ref_out, dtype=dtype)

@label("rot90")
@parametrize("shape", [(4, 5, 6), (32, 64, 16), (128, 32, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("k", [0, 1, 2, 3])
@parametrize("dims", [[0, 1], [0, 2], [1, 2]])
def test_rot90_tensor_nd(shape, dtype, k, dims):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.clone()
    ref_out = torch.ops.aten.rot90(ref_inp, k=k, dims=dims)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rot90(inp, k=k, dims=dims)
    assert_close(act_out, ref_out, dtype=dtype)

@label("rot90")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("k", [0, 1, 2, 3, -1])
@parametrize("dims", [[0, 1]])
def test_rot90_out_2d(shape, dtype, k, dims):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.clone()
    out_ref = torch.empty(0, device=device, dtype=dtype)
    ref_out = torch.ops.aten.rot90.out(ref_inp, k=k, dims=dims, out=out_ref)
    out_act = torch.empty(0, device=device, dtype=dtype)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rot90.out(inp, k=k, dims=dims, out=out_act)
    assert_close(act_out, ref_out, dtype=dtype)

@label("rot90")
@parametrize("shape", [(4, 5, 6), (32, 64, 16), (128, 32, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("k", [0, 1, 2, 3])
@parametrize("dims", [[0, 1], [0, 2], [1, 2]])
def test_rot90_out_nd(shape, dtype, k, dims):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.clone()
    out_ref = torch.empty(0, device=device, dtype=dtype)
    ref_out = torch.ops.aten.rot90.out(ref_inp, k=k, dims=dims, out=out_ref)
    out_act = torch.empty(0, device=device, dtype=dtype)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.rot90.out(inp, k=k, dims=dims, out=out_act)
    assert_close(act_out, ref_out, dtype=dtype)

# ========== t ==========
@label("t")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("contig", [True, False])
def test_t_tensor(shape, dtype, contig):
    if contig:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        base = torch.randn((shape[1], shape[0]), dtype=dtype, device=device)
        input_tensor = base.t()
    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.t(ref_input)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.t(input_tensor)
    assert_close(act_out, ref_out, dtype=dtype)

# ========== unsafe_split ==========
@label("unsafe_split")
@parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [0, 1, -1])
@parametrize("split_size", [2, 64, 1000])
def test_unsafe_split_tensor(shape, dtype, dim, split_size):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.unsafe_split(ref_x, split_size, dim)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.unsafe_split(x, split_size, dim)

    assert len(act_out) == len(ref_out)
    for a, r in zip(act_out, ref_out):
        assert_close(a, r, dtype=dtype)


@label("unsafe_split")
@parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("dim", [0, 1, -1])
@parametrize("split_size", [2, 64, 1000])
def test_unsafe_split_tensor_out(shape, dtype, dim, split_size):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    d = dim % len(shape)
    size_d = shape[d]
    num_chunks = (size_d + split_size - 1) // split_size
    out_shapes = []
    for i in range(num_chunks):
        chunk = split_size if (i + 1) * split_size <= size_d else (size_d - i * split_size)
        s = list(shape)
        s[d] = chunk
        out_shapes.append(tuple(s))

    ref_out_list = [torch.empty(s, dtype=dtype, device=device) for s in out_shapes]
    act_out_list = [torch.empty(s, dtype=dtype, device=device) for s in out_shapes]

    torch.ops.aten.unsafe_split(ref_x, split_size, dim, out=ref_out_list)

    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.unsafe_split(x, split_size, dim, out=act_out_list)

    assert len(act_out_list) == len(ref_out_list)
    for a, r in zip(act_out_list, ref_out_list):
        assert_close(a, r, dtype=dtype)

# ========== unsafe_split_with_sizes ==========
@label("unsafe_split_with_sizes")
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "shape_split_dim",
    [
        ((2, 3), [1, 1], 0),
        ((2, 3), [1, 2], 1),
        ((128, 256), [64, 64], 0),
        ((128, 256), [100, 156], 1),
        ((512, 512), [128, 128, 256], 0),
        ((512, 512), [256, 256], 1),
        ((64, 32, 16), [4, 4, 4, 4], 2),
        ((64, 32, 16), [10, 10, 12], 1),
        ((64, 32, 16), [16], -1),
    ],
)
def test_unsafe_split_with_sizes_tensor(shape_split_dim, dtype):
    shape, split_sizes, dim = shape_split_dim
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.unsafe_split_with_sizes(ref_x, split_sizes, dim)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.unsafe_split_with_sizes(x, split_sizes, dim)

    assert len(act_out) == len(ref_out)
    for a, r in zip(act_out, ref_out):
        assert_close(a, r, dtype=dtype)


@label("unsafe_split_with_sizes")
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "shape_split_dim",
    [
        ((2, 3), [1, 1], 0),
        ((2, 3), [1, 2], 1),
        ((128, 256), [64, 64], 0),
        ((128, 256), [100, 156], 1),
        ((512, 512), [128, 128, 256], 0),
        ((512, 512), [256, 256], 1),
        ((64, 32, 16), [4, 4, 4, 4], 2),
        ((64, 32, 16), [10, 10, 12], 1),
        ((64, 32, 16), [16], -1),
    ],
)
def test_unsafe_split_with_sizes_out(shape_split_dim, dtype):
    shape, split_sizes, dim = shape_split_dim
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ndim = len(shape)
    dim_eff = dim if dim >= 0 else dim + ndim
    piece_shapes = [tuple(shape[:dim_eff] + (s,) + shape[dim_eff + 1 :]) for s in split_sizes]

    ref_out_bufs = [torch.empty(ps, dtype=dtype, device=device) for ps in piece_shapes]
    act_out_bufs = [torch.empty(ps, dtype=dtype, device=device) for ps in piece_shapes]

    torch.ops.aten.unsafe_split_with_sizes.out(ref_x, split_sizes, dim, out=ref_out_bufs)
    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.unsafe_split_with_sizes.out(x, split_sizes, dim, out=act_out_bufs)

    assert len(act_out_bufs) == len(ref_out_bufs)
    for a, r in zip(act_out_bufs, ref_out_bufs):
        assert_close(a, r, dtype=dtype)

# ========== unsqueeze ==========
@label(f"unsqueeze")
@parametrize(
    "shape_dim",
    [
        ((), 0),
        ((), -1),
        ((2, 3), -3),
        ((2, 3), -1),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), 2),
        ((128, 256), -3),
        ((128, 256), -1),
        ((128, 256), 0),
        ((128, 256), 2),
        ((512, 512), -3),
        ((512, 512), 0),
        ((512, 512), 2),
        ((64, 32, 16), -4),
        ((64, 32, 16), -1),
        ((64, 32, 16), 0),
        ((64, 32, 16), 2),
        ((64, 32, 16), 3),
        ((4,), -2),
        ((4,), -1),
        ((4,), 0),
        ((4,), 1),
    ],
)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unsqueeze_tensor(shape_dim, dtype):
    shape, dim = shape_dim
    input_tensor = torch.randn(shape, dtype=dtype, device=device)

    ref_input = input_tensor.clone()
    ref_out = torch.ops.aten.unsqueeze(ref_input, dim)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.unsqueeze(input_tensor, dim)

    assert_close(act_out, ref_out, dtype=dtype)
