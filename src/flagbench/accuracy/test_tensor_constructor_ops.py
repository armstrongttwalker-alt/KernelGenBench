import pytest
import torch

import flagbench
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    DISTRIBUTION_SHAPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)
from sandbox.config import TO_CPU

from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label


# @pytest.mark.rand
@label("rand")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand(shape, dtype):
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.rand(shape, dtype=dtype, device=device)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


# @pytest.mark.randn
@label("randn")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn(shape, dtype):
    # if flag_gems.vendor_name == "cambricon":
    #     torch.manual_seed(42)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


# @pytest.mark.rand_like
@label("rand_like")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rand_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.rand_like(x)
    assert (res_out <= 1.0).all()
    assert (res_out >= 0.0).all()


# @pytest.mark.randn_like
@label("randn_like")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_randn_like(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.randn_like(x)
    mean = torch.mean(res_out.to("cpu"))
    std = torch.std(res_out.to("cpu"))
    assert torch.abs(mean) < 0.01
    assert torch.abs(std - 1) < 0.01


# @pytest.mark.zeros
@label("zeros")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_zeros(shape, dtype):
    # without dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.zeros(shape, device=device)
    gems_assert_equal(res_out, torch.zeros(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.zeros(shape, dtype=dtype, device=device)
    gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


# @pytest.mark.ones
@label("ones")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
def test_accuracy_ones(shape, dtype):
    # without dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.ones(shape, device=device)
    gems_assert_equal(res_out, torch.ones(shape, device="cpu" if TO_CPU else device))

    # with dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.ones(shape, dtype=dtype, device=device)
    gems_assert_equal(
        res_out, torch.ones(shape, dtype=dtype, device="cpu" if TO_CPU else device)
    )


# @pytest.mark.full
@label("full")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@parametrize("fill_value", [3.1415926, 2, False])
def test_accuracy_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value, device="cpu" if TO_CPU else device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.full(shape, fill_value, device=device)
    gems_assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(
        shape, fill_value, dtype=dtype, device="cpu" if TO_CPU else device
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.full(shape, fill_value, dtype=dtype, device=device)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.zeros_like
@label("zeros_like")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_zeros_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cpu" if TO_CPU else device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.zeros_like(x)
    out = torch.zeros_like(x)
    ref_out = to_reference(out)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.ones_like
@label("ones_like")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ones_like(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device="cpu" if TO_CPU else device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.ones_like(x)
    out = torch.ones_like(x)
    ref_out = to_reference(out)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.full_like
@label("full_like")
@parametrize("shape", POINTWISE_SHAPES)
@parametrize("dtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@parametrize("xdtype", BOOL_TYPES + ALL_INT_DTYPES + ALL_FLOAT_DTYPES)
@parametrize("fill_value", [3.1415926, 2, False])
def test_accuracy_full_like(shape, dtype, xdtype, fill_value):
    x = torch.empty(size=shape, dtype=xdtype, device="cpu" if TO_CPU else device)

    # without dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.full_like(x, fill_value)
    gems_assert_equal(res_out, torch.full_like(x, fill_value))

    # with dtype
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.full_like(x, fill_value, dtype=dtype)
    gems_assert_equal(res_out, torch.full_like(x, fill_value, dtype=dtype))


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="ZeroDivisionError")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.randperm
@label("randperm")
@parametrize("n", [123, 12345, 123456])
@parametrize("dtype", ALL_INT_DTYPES)
def test_accuracy_randperm(n, dtype):
    if n > torch.iinfo(torch.int16).max and dtype == torch.int16:
        return

    ref_out = torch.randperm(n, dtype=dtype, device="cpu" if TO_CPU else device)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.randperm(n, dtype=dtype, device=device)
    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    gems_assert_equal(sorted_res, sorted_ref)
