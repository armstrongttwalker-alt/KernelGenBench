import logging
import random

import numpy as np
import pytest
import torch

import flagbench

from sandbox.utils.accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from sandbox.config import TO_CPU
from sandbox.config import DEVICE as device
from sandbox.register import REGISTERED_OPS
from sandbox.verifier.test_parametrize import parametrize, label




@label("log_normal")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("mean_std", [(0.0, 1.0), (1.0, 0.5), (-0.5, 2.0)])
@parametrize("use_gen", [False, True])
def test_log_normal_tensor(shape, dtype, mean_std, use_gen):
    mean, std = mean_std
    input_tensor = torch.empty(shape, dtype=dtype, device=device)
    ref_input = input_tensor.clone()

    if use_gen:
        gen_ref = torch.Generator(device=device).manual_seed(12345)
        gen_act = torch.Generator(device=device).manual_seed(12345)
        ref_out = torch.ops.aten.log_normal(ref_input, mean, std, generator=gen_ref)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.log_normal(input_tensor, mean, std, generator=gen_act)
    else:
        torch.manual_seed(2025)
        torch.cuda.manual_seed(2025)
        ref_out = torch.ops.aten.log_normal(ref_input, mean, std, generator=None)
        torch.manual_seed(2025)
        torch.cuda.manual_seed(2025)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.log_normal(input_tensor, mean, std, generator=None)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("log_normal")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("mean_std", [(0.0, 1.0), (1.0, 0.5), (-0.5, 2.0)])
@parametrize("use_gen", [False, True])
def test_log_normal_out(shape, dtype, mean_std, use_gen):
    mean, std = mean_std
    input_tensor = torch.empty(shape, dtype=dtype, device=device)
    out_tensor = torch.empty_like(input_tensor)

    ref_input = input_tensor.clone()
    ref_out_buf = out_tensor.clone()

    if use_gen:
        gen_ref = torch.Generator(device=device).manual_seed(54321)
        gen_act = torch.Generator(device=device).manual_seed(54321)
        ref_out = torch.ops.aten.log_normal.out(ref_input, mean, std, generator=gen_ref, out=ref_out_buf)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.log_normal.out(input_tensor, mean, std, generator=gen_act, out=out_tensor)
    else:
        torch.manual_seed(30303)
        torch.cuda.manual_seed(30303)
        ref_out = torch.ops.aten.log_normal.out(ref_input, mean, std, generator=None, out=ref_out_buf)
        torch.manual_seed(30303)
        torch.cuda.manual_seed(30303)
        with flagbench.use_gems(REGISTERED_OPS):
            act_out = torch.ops.aten.log_normal.out(input_tensor, mean, std, generator=None, out=out_tensor)

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


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

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("unfold_backward")
@parametrize("case", [
    ((2, 3), 1, 2, 1),
    ((3, 5), 0, 2, 1),
    ((128, 256), 1, 15, 2),
    ((64, 64, 128), 2, 16, 8),
    ((32, 33, 34, 35), 3, 5, 2),
    ((256, 1024), 1, 64, 32),
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unfold_backward_tensor(case, dtype):
    input_sizes, dim, size, step = case
    S = input_sizes[dim]
    num_windows = (S - size) // step + 1
    grad_shape = list(input_sizes)
    grad_shape[dim] = num_windows
    grad_shape.append(size)

    grad_in = torch.randn(tuple(grad_shape), dtype=dtype, device=device)
    ref_grad_in = grad_in.clone()

    ref_out = torch.ops.aten.unfold_backward(ref_grad_in, list(input_sizes), dim, size, step)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.unfold_backward(grad_in, list(input_sizes), dim, size, step)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("unfold_backward")
@parametrize("case", [
    ((2, 3), 1, 2, 1),
    ((128, 256), 1, 15, 2),
    ((64, 64, 128), 2, 16, 8),
    ((32, 33, 34, 35), 0, 4, 3),
])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unfold_backward_out(case, dtype):
    input_sizes, dim, size, step = case
    S = input_sizes[dim]
    num_windows = (S - size) // step + 1
    grad_shape = list(input_sizes)
    grad_shape[dim] = num_windows
    grad_shape.append(size)

    grad_in = torch.randn(tuple(grad_shape), dtype=dtype, device=device)
    ref_grad_in = grad_in.clone()

    out_ref = torch.empty(tuple(input_sizes), dtype=dtype, device=device)
    out_act = torch.empty(tuple(input_sizes), dtype=dtype, device=device)

    ref_out = torch.ops.aten.unfold_backward.out(ref_grad_in, list(input_sizes), dim, size, step, out=out_ref)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.unfold_backward.out(grad_in, list(input_sizes), dim, size, step, out=out_act)

    gems_assert_close(act_out, ref_out, dtype=dtype)



@label("logit_backward")
@parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("eps", [None, 0.01, 0.1])
def test_logit_backward_tensor(shape, dtype, eps):
    self_tensor = torch.rand(shape, device=device, dtype=dtype) * 0.9 + 0.05
    grad_output = torch.randn(shape, device=device, dtype=dtype)

    ref_self = self_tensor.clone()
    ref_grad = grad_output.clone()

    ref_out = torch.ops.aten.logit_backward(ref_grad, ref_self, eps)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.logit_backward(grad_output, self_tensor, eps)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("logit_backward")
@parametrize("shape", [(2, 3), (128, 256), (1024, 256)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("eps", [None, 0.01, 0.1])
def test_logit_backward_out(shape, dtype, eps):
    self_tensor = torch.rand(shape, device=device, dtype=dtype) * 0.9 + 0.05
    grad_output = torch.randn(shape, device=device, dtype=dtype)

    ref_self = self_tensor.clone()
    ref_grad = grad_output.clone()

    ref_out_buf = torch.empty_like(ref_self)
    act_out_buf = torch.empty_like(self_tensor)

    ref_out = torch.ops.aten.logit_backward.grad_input(ref_grad, ref_self, eps, grad_input=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.logit_backward.grad_input(grad_output, self_tensor, eps, grad_input=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)



@label("convolution")
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "cfg",
    [
        dict(ndim=1, N=2, Cin=3, Cout=4, spatial=(64,), ksize=(3,), stride=(1,), pad=(1,), dil=(1,), transposed=False, outpad=(0,), groups=1, bias=True),
        dict(ndim=2, N=2, Cin=16, Cout=32, spatial=(32, 32), ksize=(3, 3), stride=(2, 2), pad=(1, 1), dil=(1, 1), transposed=False, outpad=(0, 0), groups=1, bias=False),
        dict(ndim=3, N=1, Cin=8, Cout=8, spatial=(16, 16, 16), ksize=(3, 3, 3), stride=(1, 1, 1), pad=(1, 1, 1), dil=(1, 1, 1), transposed=False, outpad=(0, 0, 0), groups=2, bias=True),
        dict(ndim=1, N=2, Cin=4, Cout=6, spatial=(32,), ksize=(4,), stride=(2,), pad=(1,), dil=(1,), transposed=True, outpad=(0,), groups=1, bias=True),
        dict(ndim=2, N=2, Cin=32, Cout=16, spatial=(16, 16), ksize=(3, 3), stride=(2, 2), pad=(1, 1), dil=(1, 1), transposed=True, outpad=(1, 1), groups=2, bias=False),
        dict(ndim=2, N=2, Cin=32, Cout=64, spatial=(128, 128), ksize=(3, 3), stride=(1, 1), pad=(1, 1), dil=(1, 1), transposed=False, outpad=(0, 0), groups=1, bias=True),
    ],
)
def test_convolution_tensor(dtype, cfg):
    N = cfg["N"]
    Cin = cfg["Cin"]
    Cout = cfg["Cout"]
    spatial = cfg["spatial"]
    ksize = cfg["ksize"]
    stride = list(cfg["stride"])
    pad = list(cfg["pad"])
    dil = list(cfg["dil"])
    transposed = cfg["transposed"]
    outpad = list(cfg["outpad"])
    groups = cfg["groups"]
    use_bias = cfg["bias"]

    input = torch.randn((N, Cin, *spatial), dtype=dtype, device=device)
    if not transposed:
        weight = torch.randn((Cout, Cin // groups, *ksize), dtype=dtype, device=device)
    else:
        weight = torch.randn((Cin, Cout // groups, *ksize), dtype=dtype, device=device)
    bias = torch.randn((Cout,), dtype=dtype, device=device) if use_bias else None

    ref_input = input.clone()
    ref_weight = weight.clone()
    ref_bias = bias.clone() if bias is not None else None

    ref_out = torch.ops.aten.convolution(ref_input, ref_weight, ref_bias, stride, pad, dil, transposed, outpad, groups)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.convolution(input.clone(), weight.clone(), bias.clone() if bias is not None else None, stride, pad, dil, transposed, outpad, groups)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("convolution")
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize(
    "cfg",
    [
        dict(ndim=1, N=2, Cin=3, Cout=4, spatial=(64,), ksize=(3,), stride=(1,), pad=(1,), dil=(1,), transposed=False, outpad=(0,), groups=1, bias=True),
        dict(ndim=2, N=2, Cin=16, Cout=32, spatial=(32, 32), ksize=(3, 3), stride=(2, 2), pad=(1, 1), dil=(1, 1), transposed=False, outpad=(0, 0), groups=1, bias=False),
        dict(ndim=3, N=1, Cin=8, Cout=8, spatial=(16, 16, 16), ksize=(3, 3, 3), stride=(1, 1, 1), pad=(1, 1, 1), dil=(1, 1, 1), transposed=False, outpad=(0, 0, 0), groups=2, bias=True),
        dict(ndim=1, N=2, Cin=4, Cout=6, spatial=(32,), ksize=(4,), stride=(2,), pad=(1,), dil=(1,), transposed=True, outpad=(0,), groups=1, bias=True),
        dict(ndim=2, N=2, Cin=32, Cout=16, spatial=(16, 16), ksize=(3, 3), stride=(2, 2), pad=(1, 1), dil=(1, 1), transposed=True, outpad=(1, 1), groups=2, bias=False),
        dict(ndim=2, N=2, Cin=32, Cout=64, spatial=(128, 128), ksize=(3, 3), stride=(1, 1), pad=(1, 1), dil=(1, 1), transposed=False, outpad=(0, 0), groups=1, bias=True),
    ],
)
def test_convolution_out_tensor(dtype, cfg):
    N = cfg["N"]
    Cin = cfg["Cin"]
    Cout = cfg["Cout"]
    spatial = cfg["spatial"]
    ksize = cfg["ksize"]
    stride = list(cfg["stride"])
    pad = list(cfg["pad"])
    dil = list(cfg["dil"])
    transposed = cfg["transposed"]
    outpad = list(cfg["outpad"])
    groups = cfg["groups"]
    use_bias = cfg["bias"]

    input = torch.randn((N, Cin, *spatial), dtype=dtype, device=device)
    if not transposed:
        weight = torch.randn((Cout, Cin // groups, *ksize), dtype=dtype, device=device)
    else:
        weight = torch.randn((Cin, Cout // groups, *ksize), dtype=dtype, device=device)
    bias = torch.randn((Cout,), dtype=dtype, device=device) if use_bias else None

    ref_input = input.clone()
    ref_weight = weight.clone()
    ref_bias = bias.clone() if bias is not None else None

    ref_temp = torch.ops.aten.convolution(ref_input, ref_weight, ref_bias, stride, pad, dil, transposed, outpad, groups)
    ref_out = torch.empty_like(ref_temp)
    ref_out = torch.ops.aten.convolution.out(ref_input, ref_weight, ref_bias, stride, pad, dil, transposed, outpad, groups, out=ref_out)

    act_input = input.clone()
    act_weight = weight.clone()
    act_bias = bias.clone() if bias is not None else None

    act_out = torch.empty_like(ref_temp)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.convolution.out(act_input, act_weight, act_bias, stride, pad, dil, transposed, outpad, groups, out=act_out)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("linalg_cross")
@parametrize(
    "case",
    [
        ((2, 3), -1),
        ((2, 3), 1),
        ((128, 3), 1),
        ((512, 3), 1),
        ((64, 128, 3), -1),
        ((64, 3, 128), 1),
        ((32, 32, 3), 2),
        ((128, 256, 3), 2),
    ],
)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linalg_cross_tensor(case, dtype):
    shape, dim = case
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out = torch.ops.aten.linalg_cross(ref_self, ref_other, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.linalg_cross(self, other, dim=dim)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("linalg_cross")
@parametrize(
    "case",
    [
        ((2, 3), -1),
        ((2, 3), 1),
        ((128, 3), 1),
        ((512, 3), 1),
        ((64, 128, 3), -1),
        ((64, 3, 128), 1),
        ((32, 32, 3), 2),
        ((128, 256, 3), 2),
    ],
)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linalg_cross_out(case, dtype):
    shape, dim = case
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ref_self = self.clone()
    ref_other = other.clone()

    ref_out_buf = torch.empty(shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.linalg_cross.out(ref_self, ref_other, dim=dim, out=ref_out_buf)

    act_out_buf = torch.empty(shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.linalg_cross.out(self, other, dim=dim, out=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("avg_pool3d")
@parametrize("shape", [(1, 2, 8, 8, 8), (2, 8, 16, 17, 18), (4, 16, 32, 33, 34)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("kernel_stride_pad", [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),
    ((3, 3, 3), (1, 1, 1), (1, 1, 1)),
    ((3, 2, 2), (2, 1, 2), (1, 0, 1)),
])
@parametrize("ceil_mode", [False, True])
@parametrize("count_include_pad", [True, False])
@parametrize("divisor_override", [None, 1, 7])
def test_avg_pool3d_tensor(shape, dtype, kernel_stride_pad, ceil_mode, count_include_pad, divisor_override):
    ks, stride, pad = kernel_stride_pad
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out = torch.ops.aten.avg_pool3d(ref_x, ks, stride, pad, ceil_mode, count_include_pad, divisor_override)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.avg_pool3d(x, ks, stride, pad, ceil_mode, count_include_pad, divisor_override)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("avg_pool3d")
@parametrize("shape", [(1, 2, 8, 8, 8), (2, 8, 16, 17, 18), (4, 16, 32, 33, 34)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("kernel_stride_pad", [
    ((2, 2, 2), (2, 2, 2), (0, 0, 0)),
    ((3, 3, 3), (1, 1, 1), (1, 1, 1)),
    ((4, 3, 2), (2, 3, 1), (0, 1, 0)),
])
@parametrize("ceil_mode", [False, True])
@parametrize("count_include_pad", [True, False])
@parametrize("divisor_override", [None, 3])
def test_avg_pool3d_out(shape, dtype, kernel_stride_pad, ceil_mode, count_include_pad, divisor_override):
    ks, stride, pad = kernel_stride_pad

    def _outdim(L, k, s, p, ceil):
        return ((L + 2 * p - k + (s - 1 if ceil else 0)) // s) + 1

    N, C, D, H, W = shape
    Do = _outdim(D, ks[0], stride[0], pad[0], ceil_mode)
    Ho = _outdim(H, ks[1], stride[1], pad[1], ceil_mode)
    Wo = _outdim(W, ks[2], stride[2], pad[2], ceil_mode)
    out_shape = (N, C, Do, Ho, Wo)

    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = x.clone()

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=device)
    ref_out = torch.ops.aten.avg_pool3d.out(ref_x, ks, stride, pad, ceil_mode, count_include_pad, divisor_override, out=ref_out_buf)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out_buf = torch.empty(out_shape, dtype=dtype, device=device)
        act_out = torch.ops.aten.avg_pool3d.out(x, ks, stride, pad, ceil_mode, count_include_pad, divisor_override, out=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("round")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_tensor(shape, dtype):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()
    ref_out = torch.ops.aten.round(ref_x)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.round(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("round")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("decimals", [-2, 0, 2])
def test_round_tensor_decimals(shape, dtype, decimals):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()
    ref_out = torch.ops.aten.round.decimals(ref_x, decimals=decimals)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.round.decimals(x, decimals=decimals)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("round")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_round_tensor_out(shape, dtype):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    ref_out = torch.ops.aten.round.out(ref_x, out=ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.round.out(x, out=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("round")
@parametrize("shape", [(2, 3), (128, 256), (1024, 1024)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("decimals", [-2, 0, 2])
def test_round_tensor_decimals_out(shape, dtype, decimals):
    x = torch.randn(shape, device=device, dtype=dtype)
    ref_x = x.clone()
    ref_out_buf = torch.empty_like(ref_x)
    act_out_buf = torch.empty_like(x)

    ref_out = torch.ops.aten.round.decimals_out(ref_x, decimals=decimals, out=ref_out_buf)
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.round.decimals_out(x, decimals=decimals, out=act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("round")
@parametrize("a", [-10, -3, -1, 0, 1, 2, 7, 15])
def test_round_int_scalar(a):
    ref = torch.ops.aten.round.int(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act = torch.ops.aten.round.int(a)
    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    gems_assert_close(act_t, ref_t, dtype=torch.float32)


@label("round")
@parametrize("a", [-3.75, -2.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.25, 2.5, 3.75])
def test_round_float_scalar(a):
    ref = torch.ops.aten.round.float(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act = torch.ops.aten.round.float(a)
    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    gems_assert_close(act_t, ref_t, dtype=torch.float32)


@label("round")
@parametrize("a", [-5, -1.5, -0.5, 0, 0.5, 1.2, 2, 3.5, 10.0])
def test_round_scalar_overload(a):
    ref = torch.ops.aten.round.Scalar(a)
    with flagbench.use_gems(REGISTERED_OPS):
        act = torch.ops.aten.round.Scalar(a)
    ref_t = torch.tensor(ref, dtype=torch.float32, device=device)
    act_t = torch.tensor(act, dtype=torch.float32, device=device)
    gems_assert_close(act_t, ref_t, dtype=torch.float32)


@label("baddbmm")
@parametrize("shape", [(2, 3, 4, 5), (8, 64, 32, 16), (16, 128, 64, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha_beta", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.3)])
def test_baddbmm_tensor(shape, dtype, alpha_beta):
    B, N, M, P = shape
    alpha, beta = alpha_beta
    self = torch.randn((B, N, P), device=device, dtype=dtype)
    batch1 = torch.randn((B, N, M), device=device, dtype=dtype)
    batch2 = torch.randn((B, M, P), device=device, dtype=dtype)

    ref_self = self.clone()
    ref_batch1 = batch1.clone()
    ref_batch2 = batch2.clone()

    ref_out = torch.ops.aten.baddbmm(ref_self, ref_batch1, ref_batch2, beta=beta, alpha=alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.baddbmm(self, batch1, batch2, beta=beta, alpha=alpha)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("baddbmm")
@parametrize("shape", [(2, 3, 4, 5), (8, 64, 32, 16), (16, 128, 64, 64)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha_beta", [(1.0, 1.0), (0.5, 2.0), (2.0, 0.3)])
def test_baddbmm_out(shape, dtype, alpha_beta):
    B, N, M, P = shape
    alpha, beta = alpha_beta
    self = torch.randn((B, N, P), device=device, dtype=dtype)
    batch1 = torch.randn((B, N, M), device=device, dtype=dtype)
    batch2 = torch.randn((B, M, P), device=device, dtype=dtype)

    ref_self = self.clone()
    ref_batch1 = batch1.clone()
    ref_batch2 = batch2.clone()
    ref_out = torch.empty((B, N, P), device=device, dtype=dtype)

    ref_res = torch.ops.aten.baddbmm.out(ref_self, ref_batch1, ref_batch2, beta=beta, alpha=alpha, out=ref_out)

    act_out = torch.empty((B, N, P), device=device, dtype=dtype)
    with flagbench.use_gems(REGISTERED_OPS):
        act_res = torch.ops.aten.baddbmm.out(self, batch1, batch2, beta=beta, alpha=alpha, out=act_out)

    gems_assert_close(act_res, ref_res, dtype=dtype)


@label("addbmm")
@parametrize("shape", [(4, 8, 16, 12), (16, 64, 32, 48), (32, 128, 64, 96)])  # (b, n, p, m)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha", [1.0, 0.5])
@parametrize("beta", [1.0, 0.0])
def test_addbmm_tensor(shape, dtype, alpha, beta):
    b, n, p, m = shape
    input_2d = torch.randn((n, m), dtype=dtype, device=device)
    batch1 = torch.randn((b, n, p), dtype=dtype, device=device)
    batch2 = torch.randn((b, p, m), dtype=dtype, device=device)

    ref_input_2d = input_2d.clone()
    ref_batch1 = batch1.clone()
    ref_batch2 = batch2.clone()

    ref_out = torch.ops.aten.addbmm(ref_input_2d, ref_batch1, ref_batch2, beta=beta, alpha=alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.addbmm(input_2d, batch1, batch2, beta=beta, alpha=alpha)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@label("addbmm")
@parametrize("shape", [(4, 8, 16, 12), (16, 64, 32, 48), (32, 128, 64, 96)])  # (b, n, p, m)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("alpha", [1.0, 0.5])
@parametrize("beta", [1.0, 0.0])
def test_addbmm_out(shape, dtype, alpha, beta):
    b, n, p, m = shape
    input_2d = torch.randn((n, m), dtype=dtype, device=device)
    batch1 = torch.randn((b, n, p), dtype=dtype, device=device)
    batch2 = torch.randn((b, p, m), dtype=dtype, device=device)

    ref_input_2d = input_2d.clone()
    ref_batch1 = batch1.clone()
    ref_batch2 = batch2.clone()

    ref_out_buf = torch.empty_like(ref_input_2d)
    torch.ops.aten.addbmm.out(ref_input_2d, ref_batch1, ref_batch2, beta=beta, alpha=alpha, out=ref_out_buf)

    act_out_buf = torch.empty_like(input_2d)
    with flagbench.use_gems(REGISTERED_OPS):
        torch.ops.aten.addbmm.out(input_2d, batch1, batch2, beta=beta, alpha=alpha, out=act_out_buf)

    gems_assert_close(act_out_buf, ref_out_buf, dtype=dtype)

