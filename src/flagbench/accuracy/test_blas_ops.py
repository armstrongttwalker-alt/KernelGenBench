import pytest
import torch

import flagbench
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    FLOAT_DTYPES,
    SCALARS,
    UT_SHAPES_1D,
    gems_assert_close,
    to_reference,
)
from sandbox.config import QUICK_MODE
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
MN_SHAPES = [(1, 32)] if QUICK_MODE else [(1, 32), (160, 1024), (5333, 497)]
MNK_SHAPES = (
    [(1, 1, 32)] if QUICK_MODE else [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)]
)
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES


# @pytest.mark.addmm
@label("addmm")
# @pytest.mark.linear
@label("linear")
# @pytest.mark.matmul
@label("matmul")
@parametrize("M, N, K", MNK_SHAPES)
@parametrize("scalar", SCALARS)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=device)
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((N,), dtype=dtype, device=device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)
    ref_bias = to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


MK_SHAPES = [(3, 4)] if QUICK_MODE else [
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
]

# @pytest.mark.addmv
# @label("addmv")
# @pytest.mark.linear  # 若“linear”标签适用，可保留
@label("linear")
# @pytest.mark.mv  # 若有对应 mv 算子标签，可自定义
@label("mv")
@parametrize("M, K", MK_SHAPES)
@parametrize("beta", SCALARS)
@parametrize("alpha", SCALARS)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addmv(M, K, beta, alpha, dtype):
    # 构造输入
    self_tensor = torch.randn((M,), dtype=dtype, device=device)
    mat = torch.randn((M, K), dtype=dtype, device=device)
    vec = torch.randn((K,), dtype=dtype, device=device)
    # 转为 reference（CPU 或可靠后端）
    ref_self = to_reference(self_tensor, True)
    ref_mat = to_reference(mat, True)
    ref_vec = to_reference(vec, True)

    # 参考计算：torch.addmv(self, mat, vec, beta=beta, alpha=alpha)
    ref_out = torch.addmv(ref_self, ref_mat, ref_vec, beta=beta, alpha=alpha)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.addmv(self_tensor, mat, vec, beta=beta, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)



# @pytest.mark.bmm
@label("bmm")
@parametrize("M, N, K", MNK_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_bmm(M, N, K, dtype):
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.bmm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


# TODO: failed at (1, 1, 2)
# @pytest.mark.mm
@label("mm")
@parametrize("M, N, K", MNK_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mm(M, N, K, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=device)
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


# @pytest.mark.mv
@label("mv")
@parametrize("M, N", MN_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=device)
    vector = torch.randn((M,), dtype=dtype, device=device)
    ref_matrix = to_reference(matrix, True)
    ref_vector = to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.mv(matrix, vector)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)


# @pytest.mark.outer
@label("outer")
@parametrize(
    # "M, N", MN_SHAPES + ([(32, 131072)] if flag_gems.vendor_name == "cambricon" else [])
    "M, N", MN_SHAPES
)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=device, requires_grad=True)
    inp2 = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.outer(ref_inp1, ref_inp2)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.outer(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    ref_in1_grad, ref_in2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )
    res_in1_grad, res_in2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)
    gems_assert_close(res_in1_grad, ref_in1_grad, dtype, reduce_dim=N)
    gems_assert_close(res_in2_grad, ref_in2_grad, dtype, reduce_dim=M)


# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="Segmentation fault")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.vdot
@label("vdot")
@parametrize("M", UT_SHAPES_1D)
@parametrize(
    "is_conj", [(False, False), (False, True), (True, False), (True, True)]
)
@parametrize("dtype", FLOAT_DTYPES + [torch.cfloat])
@parametrize("stride", [1, 2])
def test_accuracy_vdot(M, is_conj, dtype, stride):
    inp1_is_conj, inp2_is_conj = is_conj

    inp1 = torch.randn(M, dtype=dtype, device=device)
    inp2 = torch.randn(M, dtype=dtype, device=device)

    inp1 = inp1[::stride]
    inp2 = inp2[::stride]

    if inp1_is_conj:
        inp1 = inp1.conj()
    if inp2_is_conj:
        inp2 = inp2.conj()

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.vdot(inp1, inp2)
    ref_out = torch.vdot(ref_inp1, ref_inp2)
    gems_assert_close(res_out, ref_out, dtype)
