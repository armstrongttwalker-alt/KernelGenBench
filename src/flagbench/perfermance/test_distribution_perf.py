import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark, unary_input_fn
from bench.sandbox.test.test_parametrize import label, parametrize, Param


def normal_input_fn(shape, cur_dtype, device):
    loc = torch.full(shape, fill_value=3.0, dtype=cur_dtype, device=device)
    scale = torch.full(shape, fill_value=10.0, dtype=cur_dtype, device=device)
    yield loc, scale


@parametrize(
    "op_name, torch_op, input_fn",
    [
        Param(
            "normal",
            torch.distributions.normal.Normal,
            normal_input_fn,
            # marks=pytest.mark.normal,
            marks="normal",
        ),
        Param(
            "uniform_",
            torch.Tensor.uniform_,
            unary_input_fn,
            # marks=pytest.mark.uniform_,
            marks="uniform_",
        ),
        Param(
            "exponential_",
            torch.Tensor.exponential_,
            unary_input_fn,
            # marks=pytest.mark.exponential_,
            marks="exponential_",
        ),
    ],
)
def test_distribution_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    return bench.run()
