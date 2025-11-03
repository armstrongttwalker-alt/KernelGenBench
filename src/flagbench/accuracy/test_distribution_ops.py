import numpy as np
import pytest
import scipy
import torch

import flagbench

from sandbox.utils.accuracy_utils import DISTRIBUTION_SHAPES, FLOAT_DTYPES
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.config import DEVICE as device
from sandbox.register import REGISTERED_OPS

# @pytest.mark.normal
@label("normal")
@parametrize("float", ["none", "mean", "std"])
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_normal(float, shape, dtype):
    loc = (
        3.0
        if float == "mean"
        else torch.full(
            size=shape, fill_value=3.0, dtype=dtype, device=device
        )
    )
    scale = (
        10.0
        if float == "std"
        else torch.full(
            size=shape, fill_value=10.0, dtype=dtype, device=device
        )
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.normal(loc, scale)
    mean = torch.mean(res_out)
    std = torch.std(res_out)
    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


# @pytest.mark.uniform_
@label("uniform_")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_uniform(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        x.uniform_(-3, 3)
    assert (x <= 3.0).all()
    assert (x >= -3.0).all()


# @pytest.mark.exponential_
@label("exponential_")
@parametrize("shape", DISTRIBUTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=device)
    with flagbench.use_gems(REGISTERED_OPS):
        x.exponential_()
    assert x.min() > 0


# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="TO FIX")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.multinomial
@label("multinomial")
@parametrize("shape", [(1024, 10)])
@parametrize("dtype", [torch.float16, torch.float32])
@parametrize("n_samples", [2048])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    # First use multinomial to generate a series of indices, then
    # use the index counts as the input probabilities (scaled)
    rand_indices = torch.multinomial(torch.rand(shape), n_samples, True).to(device)
    inp_counts = torch.nn.functional.one_hot(rand_indices).sum(1)
    with flagbench.use_gems(REGISTERED_OPS):
        out_indices = torch.multinomial(inp_counts.to(dtype=dtype), n_samples, True)
    out_counts = torch.nn.functional.one_hot(out_indices).sum(1)
    # Do a simple Chi-square test
    assert torch.equal(inp_counts.sum(-1), out_counts.sum(-1))
    chi2, pvalue = scipy.stats.chisquare(
        out_counts.tolist(), inp_counts.tolist(), axis=-1
    )
    assert np.sum(pvalue < 0.05) / len(pvalue) < 0.1
