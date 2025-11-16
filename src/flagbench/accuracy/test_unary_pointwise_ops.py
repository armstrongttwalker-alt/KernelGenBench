import pytest
import torch

import flagbench
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
    unsqueeze_tensor,
    unsqueeze_tuple,
)
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label


# @pytest.mark.abs
@label("abs")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.abs_
@label("abs_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.abs_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.abs_(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.bitwise_not
@label("bitwise_not")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            device
        )
    else:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.bitwise_not_
@label("bitwise_not_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot_(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device=device
        )
    ref_inp = to_reference(inp.clone())

    ref_out = ref_inp.bitwise_not_()  # NOTE: there is no torch.bitwse_not_
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = inp.bitwise_not_()

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.cos
@label("cos")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.cos_
@label("cos_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cos_(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.exp
@label("exp")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.exp_
@label("exp_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.exp_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.exp_(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.gelu
@label("gelu")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu(shape, dtype, approximate):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.gelu(inp, approximate=approximate)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.gelu_
@label("gelu_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_(shape, dtype, approximate):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp.clone(), True)

    # NOTE: we cannot apply inplace operation on leaf nodes that requires gradient
    ref_out = torch.ops.aten.gelu_.default(ref_inp * 2.0, approximate=approximate)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.ops.aten.gelu_.default(inp * 2.0, approximate=approximate)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.isinf
@label("isinf")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.isinf(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.isnan
@label("isnan")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isnan(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.isnan(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.neg
@label("neg")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.neg_
@label("neg_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.neg_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.neg_(inp)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.reciprocal
@label("reciprocal")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.reciprocal_
@label("reciprocal_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.reciprocal_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.reciprocal_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# @pytest.mark.elu
@label("elu")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_elu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    alpha = torch.rand(1).item()

    ref_inp = to_reference(inp, True)
    ref_out = torch.nn.functional.elu(ref_inp, alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.elu(inp, alpha)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.relu
@label("relu")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.relu(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.relu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.relu_
@label("relu_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.relu_(ref_inp * 2.0)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.relu_(inp * 2.0)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.rsqrt
@label("rsqrt")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.rsqrt_
@label("rsqrt_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.rsqrt_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.rsqrt_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# @pytest.mark.sigmoid
@label("sigmoid")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.sigmoid(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.sigmoid_
@label("sigmoid_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.sigmoid_(ref_inp * 1.0)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.sigmoid_(inp * 1.0)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


SPECIAL_VALUES = [float("-inf"), float("inf"), -300]


# @pytest.mark.log_sigmoid
@label("log_sigmoid")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=device
        )
        inp = torch.cat((inp, special_inputs))
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.logsigmoid(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.logsigmoid(inp)
    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.silu
@label("silu")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.silu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.silu_
@label("silu_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.nn.functional.silu(ref_inp * 1.0, inplace=True)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.silu(inp * 1.0, inplace=True)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.sin
@label("sin")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.sin_
@label("sin_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.sin_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.sin_(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.tanh
@label("tanh")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.tanh(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.tanh(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.tanh_
@label("tanh_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.tanh_(ref_inp * 1.0)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.tanh_(inp * 1.0)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


# @pytest.mark.triu
@label("triu")
@parametrize("shape, diagonal", SHAPE_DIAGONAL)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = unsqueeze_tensor(inp, 2)
    ref_inp = to_reference(inp)

    ref_out = torch.triu(ref_inp, diagonal)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.erf
@label("erf")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_erf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.erf(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.inplace
@label("inplace")
# @pytest.mark.erf_
@label("erf_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_erf_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.erf_(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.erf_(inp)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.isfinite
@label("isfinite")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", ALL_FLOAT_DTYPES)
def test_accuracy_isfinite(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.masked_fill(inp, inp > 1.0, float("inf"))
    inp = torch.masked_fill(inp, inp < -1.0, float("-inf"))
    inp = torch.masked_fill(inp, (inp > -0.1) & (inp < 0.1), float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isfinite(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.isfinite(inp)
    gems_assert_equal(res_out, ref_out)


def get_max_ndim(shape, dims):
    max_ndim = max(len(shape), len(dims))
    for dim in dims:
        dim = dim + 1 if dim >= 0 else -dim
        if dim > max_ndim:
            max_ndim = dim
    return max_ndim


FLIP_DIMS = [(0,), (-2,), (2,), (0, 2), (2, 1), (0, -1, 1)]


# @pytest.mark.flip
@label("flip")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("dims", FLIP_DIMS)
def test_accuracy_flip_general(shape, dtype, dims):
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=device).to(dtype)
    max_ndim = get_max_ndim(shape, dims)
    inp = unsqueeze_tensor(inp, max_ndim)
    ref_inp = to_reference(inp, False)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.flip
@label("flip")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@parametrize("dims", FLIP_DIMS)
def test_accuracy_flip_with_non_dense_input(shape, dtype, dims):
    max_ndim = get_max_ndim(shape, dims)
    shape = unsqueeze_tuple(shape, max(max_ndim, 2))

    shape_dialted = tuple(item * 2 for item in shape)
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape_dialted, dtype=dtype, device=device)[::2, ::2]
    else:
        inp = torch.randint(-1000, 1000, shape_dialted, device=device).to(
            dtype
        )[::2, ::2]
    ref_inp = to_reference(inp, False)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.masked_fill
@label("masked_fill")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("threshold", [0.3, 0.5, 0.7])
@parametrize(
    "value",
    [
        torch.tensor(1024, device=device),
        torch.scalar_tensor(1024, device=device),
        1024,
    ],
)
def test_accuracy_masked_fill(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=device)
    mask = torch.randn(shape, dtype=dtype, device=device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_out = torch.masked_fill(ref_inp, ref_mask, to_reference(value))
    else:
        ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.masked_fill(inp, mask, value)

    gems_assert_equal(res_out, ref_out)


# @pytest.mark.masked_fill
@label("masked_fill_")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("threshold", [0.3, 0.5, 0.7])
@parametrize(
    "value",
    [
        torch.tensor(1024, device=device),
        torch.scalar_tensor(1024, device=device),
        1024,
    ],
)
def test_accuracy_masked_fill_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=device)
    mask = torch.randn(shape, dtype=dtype, device=device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flagbench.use_gems(REGISTERED_OPS):
        inp.masked_fill_(mask, value)

    gems_assert_equal(inp, ref_inp)


TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


# @pytest.mark.tile
@label("tile")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dims", TILE_DIMS)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)

    ref_out = torch.tile(ref_inp, dims)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.tile(inp, dims)

    gems_assert_close(res_out, ref_out, dtype)


REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]


# @pytest.mark.repeat
@label("repeat")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("sizes", REPEAT_SIZES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat(shape, sizes, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = to_reference(inp)
    sizes = unsqueeze_tuple(sizes, inp.ndim)

    ref_out = ref_inp.repeat(*sizes)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = inp.repeat(*sizes)

    gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.logical_not
@label("logical_not")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_not(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    elif dtype in ALL_INT_DTYPES:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            device
        )
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(device)

    ref_inp = to_reference(inp)
    ref_out = torch.logical_not(ref_inp)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.logical_not(inp)

    gems_assert_equal(res_out, ref_out)
