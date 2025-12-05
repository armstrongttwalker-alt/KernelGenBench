import random
import time

import numpy as np
import pytest
import torch

import flagbench

from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    CONTIGUOUS_SHAPE_STRIDES_2D,
    FLOAT_DTYPES,
    INT_DTYPES,
    IRREGULAR_SHAPE_STRIDES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SHAPE_STRIDES,
    SkipVersion,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from sandbox.config import QUICK_MODE
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
# Make sure every thread has same seed.
random.seed(time.time() // 100)

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS_SHAPE = (
    [(True, DIMS_LIST[0], REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([True, False] * 2, DIMS_LIST, REDUCTION_SHAPES + [(7, 4, 11, 1)]))
)
SMOOTH_IGNORE_SHAPE = (
    [(0.1, 1, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([0, 0.1, 1], [1, 200, -100], REDUCTION_SHAPES))
)
SMOOTH_SHAPE = (
    [(0.1, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([1, 0.1, 0], REDUCTION_SHAPES))
)
DIM_SHAPE_STRIDES = (
    [(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])]
    if QUICK_MODE
    else list(
        (random.randint(0, len(shape) - 1), shape, stride)
        for shape, stride in SHAPE_STRIDES
    )
)
REGULAR_DIM_SHAPE_STRIDES = (
    [(1, *CONTIGUOUS_SHAPE_STRIDES_2D[1])]
    if QUICK_MODE
    else list(
        (random.randint(0, len(shape) - 1), shape, stride)
        for shape, stride in CONTIGUOUS_SHAPE_STRIDES_2D
    )
)
IRREGULAR_DIM_SHAPE_STRIDES = [(3, *IRREGULAR_SHAPE_STRIDES)]

THRESHOLD_SHAPE = (
    [(0.3, REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(zip([0.3, 0.5, 0.7], REDUCTION_SHAPES))
)
CROSS_ENTROPY_LOSS_REDUCTION = ["mean"] if QUICK_MODE else ["mean", "none", "sum"]


# @pytest.mark.amax
@label("amax")
@parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.amax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


# TODO: There are some bugs in argmax with large size.
# @pytest.mark.argmax
@label("argmax")
@parametrize("shape", REDUCTION_SMALL_SHAPES)
@parametrize("dim", DIM_LIST)
@parametrize("keepdim", [True, False])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_argmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.argmin
@label("argmin")
@parametrize("shape", REDUCTION_SMALL_SHAPES)
@parametrize("dim", DIM_LIST + [None])
@parametrize("keepdim", [True, False])
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_argmin(shape, dim, keepdim, dtype):
    if dtype in INT_DTYPES:
        inp = torch.randint(-1024, 1024, size=shape, device=device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.argmin(ref_inp, dim=dim, keepdim=keepdim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.argmin(inp, dim=dim, keepdim=keepdim)

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.CrossEntropyLoss
# @label("CrossEntropyLoss")
@label("cross_entropy")
@parametrize("label_smoothing, ignore_index, shape", SMOOTH_IGNORE_SHAPE)
@parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@parametrize("weight", [True, False])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_indices(
    shape, dtype, weight, ignore_index, reduction, label_smoothing
):
    dim = 1
    up_limit = shape[dim] - 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, up_limit, target_shape, device=device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)

    if weight:
        wgt = torch.randn(shape[dim], dtype=dtype, device=device)
        ref_wgt = to_reference(wgt, True)
    else:
        wgt = None
        ref_wgt = None
    ref_criterion = torch.nn.CrossEntropyLoss(
        weight=ref_wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_criterion = torch.nn.CrossEntropyLoss(
        weight=wgt,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    ref_out = ref_criterion(ref_inp, ref_target)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = res_criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.CrossEntropyLoss
@label("CrossEntropyLoss")
@parametrize("label_smoothing, shape", SMOOTH_SHAPE)
@parametrize("reduction", CROSS_ENTROPY_LOSS_REDUCTION)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cross_entropy_loss_probabilities(
    shape, dtype, reduction, label_smoothing
):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    target = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[dim], dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)
    ref_weight = to_reference(weight, True)
    ref_criterion = torch.nn.CrossEntropyLoss(
        weight=ref_weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    res_criterion = torch.nn.CrossEntropyLoss(
        weight=weight,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    ref_out = ref_criterion(ref_inp, ref_target)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = res_criterion(inp, target)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.NLLLoss
@label("NLLLoss")
@parametrize("reduction", ["mean", "none", "sum"])
@parametrize("weight", [True, False])
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("ignore_index", [1, 200, -100])
def test_accuracy_nll_loss(shape, dtype, ignore_index, reduction, weight):
    dim = 1
    target_shape = list(shape)
    del target_shape[dim]

    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    target = torch.randint(0, shape[dim], target_shape, device=device)
    if weight:
        weight = torch.randn(shape[dim], dtype=dtype, device=device)
    else:
        weight = None
    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target)
    ref_weight = to_reference(weight, True)

    ref_out = torch.nn.functional.nll_loss(
        ref_inp, ref_target, ref_weight, reduction=reduction, ignore_index=ignore_index
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.nll_loss(
            inp, target, weight, reduction=reduction, ignore_index=ignore_index
        )
    reduce_dim = 1 if reduction == "none" else target.numel()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, equal_nan=True)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flagbench.use_gems(REGISTERED_OPS):
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


CUMSUM_SHAPES = (
    [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.cumsum
@label("cumsum")
@parametrize("shape", CUMSUM_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cumsum(shape, dtype):
    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=device).to(dtype)
        ref_inp = to_reference(inp)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp, True)

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cumsum(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


CUMMIN_SHAPES = (
    [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)


# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="AssertionError")
# @pytest.mark.skipif
# @label("skipif")
# (
#     SkipVersion("triton", "<3.0"),
#     reason="Skipping when associative_scan only support single tensor input.",
# )
# @pytest.mark.cummin
@label("cummin")
@parametrize("shape", CUMMIN_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cummin(shape, dtype):
    dim = 1 if shape == REDUCTION_SHAPES[-1] else -1
    if dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cummin(ref_inp, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cummin(inp, dim=dim)
    gems_assert_close(res_out.values, ref_out.values, dtype, reduce_dim=shape[dim])
    gems_assert_equal(res_out.indices, ref_out.indices)


NONZERO_SHAPES = [(2, 32)] if QUICK_MODE else REDUCTION_SHAPES + [(2637,)]


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.nonzero
@label("nonzero")
@parametrize("shape", NONZERO_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=device).to(
            dtype
        )
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, False)

    ref_out = torch.nonzero(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nonzero(inp)

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.count_nonzero
@label("count_nonzero")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + [torch.bool])
def test_accuracy_count_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=device).to(
            dtype
        )
    elif dtype in INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, False)
    dim = random.choice([None] + list(range(inp.ndim)))
    ref_out = torch.count_nonzero(ref_inp, dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.count_nonzero(inp, dim)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.log_softmax
@label("log_softmax")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
# @parametrize("dim", [0, 1] if flag_gems.vendor_name == "cambricon" else [1])
@parametrize("dim", [1])
def test_accuracy_log_softmax(shape, dtype, dim):
    # if flag_gems.vendor_name == "cambricon":
    #     torch.manual_seed(42)
    #     torch.mlu.manual_seed_all(42)
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.log_softmax(ref_inp, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.log_softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


# TODO: failed at (1, 2) (200, 40999, 3)
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="Unsupported")
# @pytest.mark.softmax
@label("softmax")
@parametrize(
    "shape", [(1, 256)] if QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("dim", DIM_LIST)
def test_accuracy_softmax(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim])


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.softmax
@label("softmax")
@parametrize(
    "shape", [(1, 256)] if QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("dim", DIM_LIST)
def test_accuracy_softmax_with_neg_inf(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.softmax(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


# @pytest.mark.var_mean
@label("var_mean")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dim", DIMS_LIST)
@parametrize("correction", [1] if QUICK_MODE else [0, 1])
@parametrize("keepdim", [True] if QUICK_MODE else [True, False])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_varmean(shape, dim, correction, keepdim, dtype):
    if shape[0] == 1:  # TODO: res is inf, while ref is nan
        shape = (2, 2)
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_var, ref_mean = torch.var_mean(
        ref_inp, dim, correction=correction, keepdim=keepdim
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_var, res_mean = torch.var_mean(
            inp, dim, correction=correction, keepdim=keepdim
        )

    gems_assert_close(res_mean, ref_mean, dtype)
    gems_assert_close(res_var, ref_var, dtype)


# @pytest.mark.scatter
@label("scatter")
@parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@parametrize("dim", [0, 1, 2])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_scatter_src(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    src = torch.randn(src_shape, dtype=dtype, device=device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.scatter(inp, dim, index, src)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.scatter
@label("scatter")
@parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@parametrize("dim", [0, 1, 2])
@parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_scatter_add(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    src = torch.randn(src_shape, dtype=dtype, device=device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="add")
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.scatter(inp, dim, index, src, reduce="add")

    gems_assert_close(res_out, ref_out, dtype)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "hygon", reason="RuntimeError")
# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="RuntimeError")
# @pytest.mark.scatter
@label("scatter")
@parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@parametrize("dim", [0, 1, 2])
@parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_scatter_mul(src_shape, inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    src = torch.randn(src_shape, dtype=dtype, device=device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    import random

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_src = to_reference(src)
    ref_out = torch.scatter(ref_inp, dim, ref_index, ref_src, reduce="multiply")
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.scatter(inp, dim, index, src, reduce="multiply")

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.gather
# @pytest.mark.parametrize(
#     "inp_shape",
#     [(32, 8, 4)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16), (128, 32, 256)],
# )
# @pytest.mark.parametrize("dim", [0, 1, 2])
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@label("gather")
@parametrize(
    "inp_shape",
    [(32, 8, 4)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16), (128, 32, 256)],
)
@parametrize("dim", [0, 1, 2])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gather(inp_shape, dim, dtype):
    inp = torch.randn(
        inp_shape, dtype=dtype, device=flagbench.device, requires_grad=True
    )
    size_dim = inp_shape[dim]

    import random

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flagbench.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.gather(inp, dim, index)

    gems_assert_equal(res_out, ref_out)

    if dtype in (torch.bfloat16,):
        return

    # out_grad = torch.randn_like(res_out)
    # ref_grad = to_reference(out_grad)

    # (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    # with flagbench.use_gems(REGISTERED_OPS):
    #     (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    # res_in_grad = to_reference(res_in_grad)
    # gems_assert_equal(res_in_grad, ref_in_grad)

# @pytest.mark.gather
# @label("gather")
# @parametrize(
#     "inp_shape",
#     [(32, 8, 4)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16), (128, 32, 256)],
# )
# @parametrize("dim", [0, 1, 2])
# @parametrize("dtype", FLOAT_DTYPES)
# def test_accuracy_gather(inp_shape, dim, dtype):
#     inp = torch.randn(
#         inp_shape, dtype=dtype, device=device, requires_grad=True
#     )
#     size_dim = inp_shape[dim]

#     import random

#     index_shape = [
#         random.randint(1, inp_shape[0]),
#         random.randint(1, inp_shape[1]),
#         random.randint(1, inp_shape[2]),
#     ]
#     index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

#     m, n, o = index_shape

#     index_size_dim = index_shape[dim]
#     # make unique indices
#     for i in range(1 if dim == 0 else m):
#         for j in range(1 if dim == 1 else n):
#             for k in range(1 if dim == 2 else o):
#                 ii = [i, j, k]
#                 ii[dim] = slice(0, index.size(dim) + 1)
#                 index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

#     ref_inp = to_reference(inp)
#     ref_index = to_reference(index)
#     ref_out = torch.gather(ref_inp, dim, ref_index)

#     with flagbench.use_gems(REGISTERED_OPS):
#         res_out = torch.gather(inp, dim, index)

#     gems_assert_equal(res_out, ref_out)

#     if dtype in (torch.bfloat16,):
#         return

#     out_grad = torch.randn_like(res_out)
#     ref_grad = to_reference(out_grad)

#     (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
#     with flagbench.use_gems(REGISTERED_OPS):
#         (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
#     res_in_grad = to_reference(res_in_grad)
#     gems_assert_equal(res_in_grad, ref_in_grad)


# @pytest.mark.select_scatter
@label("select_scatter")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dim", DIM_LIST)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_select_scatter(shape, dim, dtype):
    import random

    index = random.randint(0, shape[dim] - 1)
    inp = torch.randn(shape, dtype=dtype, device=device)

    src_shape = list(inp.shape)
    del src_shape[dim]
    src = torch.randn(src_shape, dtype=dtype, device=device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.select_scatter
@label("select_scatter")
def test_accuracy_select_scatter_with_self_overlapping_input():
    dim = 0
    index = 1
    inp = torch.randn((1, 4), device=device).broadcast_to((3, 4))
    src = torch.randn((4,), device=device)

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_out = torch.select_scatter(ref_inp, dim=dim, index=index, src=ref_src)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.select_scatter(inp, dim=dim, index=index, src=src)

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.slice_scatter
@label("slice_scatter")
# @parametrize(("dim", "shape", "stride"), REGULAR_DIM_SHAPE_STRIDES)
# @parametrize("dtype", FLOAT_DTYPES)
# @parametrize("start", [16, 64])
# @parametrize("end", [1024, 256])
# @parametrize("step", [1, 2])
# def test_accuracy_slice_scatter(shape, stride, dim, dtype, start, end, step):
#     inp = torch.empty_strided(shape, stride, dtype=dtype, device=device)
#     inp.copy_(1)

#     valid_shape = list(inp.shape)
#     size = valid_shape[dim]

#     start = start % size
#     end = end % (size + 1)

#     if end < start:
#         end, start = start, end
#     elif end == start:
#         end = size

#     valid_shape[dim] = (end - start + step - 1) // step

#     src = torch.rand(valid_shape, dtype=dtype, device=device)

#     ref_inp = to_reference(inp)
#     ref_src = to_reference(src)
#     ref_out = torch.slice_scatter(
#         ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
#     )

#     if flag_gems.vendor_name == "kunlunxin":
#         from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

#         res_out = kl_ops.slice_scatter(
#             inp, dim=dim, src=src, start=start, end=end, step=step
#         )
#     else:
#         res_out = flag_gems.ops.slice_scatter(
#             inp, dim=dim, src=src, start=start, end=end, step=step
#         )

#     gems_assert_equal(res_out, ref_out)


# # @pytest.mark.slice_scatter
@label("slice_scatter")
# def test_accuracy_slice_scatter_with_self_overlapping_input():
#     inp = torch.randn((3, 1), device=device).broadcast_to((3, 8))
#     src = torch.rand((3, 4), device=device)

#     start = 0
#     end = 8
#     step = 2
#     dim = 1
#     ref_inp = to_reference(inp)
#     ref_src = to_reference(src)
#     ref_out = torch.slice_scatter(
#         ref_inp, dim=dim, src=ref_src, start=start, end=end, step=step
#     )
#     if flag_gems.vendor_name == "kunlunxin":
#         from flag_gems.runtime.backend._kunlunxin import ops as kl_ops

#         res_out = kl_ops.slice_scatter(
#             inp, dim=dim, src=src, start=start, end=end, step=step
#         )
#     else:
#         res_out = flag_gems.ops.slice_scatter(
#             inp, dim=dim, src=src, start=start, end=end, step=step
#         )

#     gems_assert_equal(res_out, ref_out)


# TODO: failed at (200, 40999, 3)
# @pytest.mark.index_add
@label("index_add")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dim", DIM_LIST)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    alpha = 2

    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@label("index_fill")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dim", DIM_LIST)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("value", [0.0, 1.23, -5.6, 3.14])
def test_accuracy_index_fill(shape, dim, dtype, value):
    inp = torch.randn(shape, dtype=dtype, device=device)

    # 生成索引
    index_max = inp.shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=device)

    # 转换为 reference
    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_fill(ref_inp, dim, ref_index, value)

    # 真正执行
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.index_fill(inp, dim, index, value)

    # 对比结果
    gems_assert_close(res_out, ref_out, dtype=dtype)



# @pytest.mark.index_select
@label("index_select")
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dim", DIM_LIST)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_index_select(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(
        0, index_size, [floor(index_size * 0.8)], device=device
    )

    ref_inp = to_reference(inp)
    ref_index = to_reference(index)
    ref_out = torch.index_select(ref_inp, dim, ref_index)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.index_select(inp, dim, index)

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.masked_select
@label("masked_select")
@parametrize("threshold, shape", THRESHOLD_SHAPE)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_select(shape, dtype, threshold):
    inp = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.randn(shape, dtype=dtype, device=device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    ref_out = torch.masked_select(ref_inp, ref_mask)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.masked_select(inp, mask)

    gems_assert_equal(res_out, ref_out)


SHAPE_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((32, 16, 1024), (1024, 16, 8)),
    ((64, 64, 64), (128, 64, 7)),
    ((32, 12, 9), (17, 12, 3)),
    ((32, 6, 6), (64, 6, 2)),
]


# @pytest.mark.skip
# @label("skip")("conv1d introduces failures, disable it temporarily")
# @pytest.mark.conv1d
@label("conv1d")
@parametrize("shape, kernel", SHAPE_CONV1D)
@parametrize("stride", [2])
@parametrize("padding", [1])
@parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy_conv1d(shape, kernel, stride, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)
    weight = torch.randn(kernel, dtype=dtype, device=device)
    ref_weight = to_reference(weight, True)
    ref_out = torch.nn.functional.conv1d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, dilation=1
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.conv1d(
            inp, weight, bias=None, stride=stride, padding=padding, dilation=1
        )
    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((2, 2, 3, 3), (1, 2, 2, 2), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ((18, 16, 4, 4), (16, 16, 2, 2), 1),
    ((9, 16, 4, 4), (128, 4, 2, 2), 4),
    ((32, 16, 8, 8), (32, 4, 4, 4), 4),
    ((18, 16, 4, 4), (16, 8, 2, 2), 2),
    ((9, 16, 4, 4), (128, 8, 2, 2), 2),
    ((32, 8, 8, 8), (32, 8, 3, 3), 1),
    ((18, 16, 5, 5), (16, 16, 3, 3), 1),
    ((9, 16, 7, 7), (128, 4, 3, 3), 4),
    ((32, 16, 9, 9), (32, 4, 5, 5), 4),
    ((18, 16, 11, 11), (16, 8, 3, 3), 2),
    ((9, 16, 6, 6), (128, 8, 3, 3), 2),
]


# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="RuntimeError")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.conv2d
# @label("conv2d")
# @parametrize("shape, kernel,groups", SHAPE_CONV2D)
# @parametrize("stride", [1, 2])
# @parametrize("padding", [0, 1])
# @parametrize("dtype", [torch.float16, torch.float32])
# @parametrize("dilation", [1, 2])
# @parametrize("bias", [True, False])
# def test_accuracy_conv2d(shape, kernel, stride, padding, groups, dtype, dilation, bias):
#     inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
#     ref_inp = to_reference(inp, True)
#     torch.backends.cudnn.allow_tf32 = False
#     weight = torch.randn(
#         kernel, dtype=dtype, device=device, requires_grad=True
#     )
#     if bias is True:
#         bias = torch.randn(
#             [weight.shape[0]], dtype=dtype, device=device, requires_grad=True
#         )
#         bias_ref = to_reference(bias, True)
#     else:
#         bias = None
#         bias_ref = None

#     ref_weight = to_reference(weight, True)
#     ref_out = torch.nn.functional.conv2d(
#         ref_inp,
#         ref_weight,
#         bias=bias_ref,
#         groups=groups,
#         stride=stride,
#         padding=padding,
#         dilation=dilation,
#     ).to(dtype)
#     with flagbench.use_gems(REGISTERED_OPS):
#         res_out = torch.nn.functional.conv2d(
#             inp,
#             weight,
#             bias=bias,
#             groups=groups,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#         )

#     gems_assert_close(res_out, ref_out, dtype)

#     out_grad = torch.randn_like(ref_out).to(device)

#     ref_grad = to_reference(out_grad, True)
#     if bias is not None:
#         (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
#             ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
#         )
#         (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
#             res_out, (inp, weight, bias), out_grad
#         )
#     else:
#         (ref_in_grad, ref_weight_grad) = torch.autograd.grad(
#             ref_out, (ref_inp, ref_weight), ref_grad
#         )
#         (res_in_grad, res_weight_grad) = torch.autograd.grad(
#             res_out, (inp, weight), out_grad
#         )

#     gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=weight.shape[2])

#     gems_assert_close(
#         res_weight_grad, ref_weight_grad, dtype, reduce_dim=weight.shape[0]
#     )
#     if bias is not None:
#         gems_assert_close(res_bias_grad, ref_bias_grad, dtype)


SHAPE_DEPTHWISE = [
    ((32, 4, 8, 8), (32, 1, 2, 2), (2, 2)),
    ((18, 16, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 32, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 16, 8, 8), (32, 1, 4, 4), (4, 4)),
    ((18, 8, 4, 4), (16, 1, 2, 2), (2, 2)),
    ((9, 4, 4, 4), (128, 1, 2, 2), (2, 2)),
    ((32, 4, 8, 8), (32, 1, 3, 3), (3, 3)),
    ((18, 16, 13, 13), (16, 1, 5, 5), (5, 5)),
    ((9, 32, 8, 8), (128, 1, 3, 3), (3, 3)),
    ((32, 16, 9, 9), (32, 1, 5, 5), (5, 5)),
    ((18, 8, 7, 7), (16, 1, 3, 3), (3, 3)),
    ((9, 4, 6, 6), (128, 1, 3, 3), (3, 3)),
]


# test for depthwise depends on specific device
# # @pytest.mark.skip
# @label("skip")("conv_depthwise2d introduces failures, disable it temporarily")
# # @pytest.mark.conv_depthwise2d
# @label("conv_depthwise2d")
# @parametrize("shape_input, shape_weight,kernel ", SHAPE_DEPTHWISE)
# @parametrize("stride", [2])
# @parametrize("padding", [2])
# @parametrize("dtype", [torch.float32])
# def test_accuracy_depthwise2d(
#     shape_input, shape_weight, kernel, stride, padding, dtype
# ):
#     inp = torch.randn(
#         shape_input, dtype=dtype, device=device, requires_grad=True
#     )
#     ref_inp = to_reference(inp, False)
#     torch.backends.cudnn.allow_tf32 = False
#     weight = torch.randn(shape_weight, dtype=dtype, device=device)
#     ref_weight = to_reference(weight, False)
#     ref_out = torch._C._nn._conv_depthwise2d(
#         ref_inp,
#         ref_weight,
#         kernel,
#         bias=None,
#         stride=stride,
#         padding=padding,
#         dilation=1,
#     )

#     res_out = flag_gems._conv_depthwise2d(
#         inp, weight, kernel, bias=None, stride=stride, padding=padding, dilation=1
#     )
#     gems_assert_close(res_out, ref_out, dtype)


INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28,), ((2**16,),), (2**16,)),
    ((32, 32), ((8,), (8,)), (8,)),
    ((32, 32), ((8,), (2, 8)), (8,)),
    ((32, 32), ((2, 8),), (32,)),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,)),
    ((512, 512, 512), ((2, 128), (128,), (128,)), (128,)),
    ((512, 512, 512), ((2, 128),), (512,)),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
        (2, 8, 64),
    ),
)


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=device))
    return indices


# @pytest.mark.index_put
@label("index_put")
@parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_FALSE
)
@parametrize("dtype", FLOAT_DTYPES)
def test_index_put_acc_false(input_shape, indices_shape, values_shape, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=device, requires_grad=False
    )

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    with flagbench.use_gems(REGISTERED_OPS):
        out = torch.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


INDEX_PUT_SHAPE_ACC_TRUE = (
    ((2**28,), ((2**16,),), (2**16,)),
    ((32, 32), ((8,), (8,)), (8,)),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,)),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8)),
)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.index_put
@label("index_put")
@parametrize(
    "input_shape, indices_shape, values_shape", INDEX_PUT_SHAPE_ACC_TRUE
)
@parametrize("dtype", [torch.float16, torch.float32])
def test_index_put_acc_true(input_shape, indices_shape, values_shape, dtype):
    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, accumulate)
    values = torch.randn(
        values_shape, dtype=dtype, device=device, requires_grad=False
    )

    ref_inp = to_reference(inp)
    ref_indices = [to_reference(index) for index in indices]
    ref_values = to_reference(values)
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    with flagbench.use_gems(REGISTERED_OPS):
        out = torch.index_put(inp, indices, values, accumulate)
    gems_assert_close(out, ref_out, dtype)


# @pytest.mark.mse_loss
@label("mse_loss")
@parametrize("reduction", ["mean", "none", "sum"])
@parametrize("shape", REDUCTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mse_loss(shape, dtype, reduction):
    # if flag_gems.vendor_name == "kunlunxin":
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)

    dim = 1
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)

    ref_inp = to_reference(inp, True)
    ref_target = to_reference(target, True)

    ref_out = torch.nn.functional.mse_loss(ref_inp, ref_target, reduction=reduction)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.mse_loss(inp, target, reduction=reduction)
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True, reduce_dim=shape[dim])
