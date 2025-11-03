import itertools
from typing import Optional

import numpy as np
import pytest
import torch

import flagbench
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import (
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    KRON_SHAPES,
    SPECIAL_SHAPES,
    STACK_DIM_LIST,
    STACK_SHAPES,
    UPSAMPLE_SHAPES,
    UT_SHAPES_1D,
    UT_SHAPES_2D,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from sandbox.config import TO_CPU

from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label


# @pytest.mark.dropout
@label("dropout")
# @pytest.mark.native_dropout
@label("native_dropout")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("p", [0.3, 0.6, 0.9])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_dropout(shape, p, dtype):
    # if flag_gems.vendor_name == "kunlunxin":
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)

    if TO_CPU or shape == (1,):
        shape = (32768,)
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp)

    # NOTE: ensure that scalars are float32(instead of float64)
    # in some cases, casting up then casting down have different result
    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.dropout(inp, p, True)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    res_out = to_reference(res_out)
    res_in_grad = to_reference(res_in_grad)

    exp_equal = (p * p + one_minus_p * one_minus_p) * inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    if TO_CPU:
        from sandbox.utils.accuracy_utils import RESOLUTION

        zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
        num_zero = torch.sum(zero_equal).item()
        assert abs(num_zero / inp.numel() - p) <= 0.05
        scale_equal = torch.isclose(
            res_out, ref_inp / one_minus_p, rtol=RESOLUTION[dtype]
        )
        assert torch.all(torch.logical_or(zero_equal, scale_equal))
    else:
        assert (
            abs(num_equal - exp_equal) / exp_equal <= 0.05
        ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"

        num_equal = torch.sum(torch.isclose(ref_in_grad, res_in_grad)).item()
        assert (
            abs(num_equal - exp_equal) / exp_equal <= 0.05
        ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"


def get_rope_cos_sin(max_seq_len, dim, dtype, base=10000, device=device):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.cohere.modeling_cohere.rotate_half
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere/modeling_cohere.py
def rotate_interleave(x):
    """Rotates interleave the hidden dims of the input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def torch_apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
):
    q = q.float()
    k = k.float()
    if position_ids is None:
        cos = cos[None, : q.size(-3), None, :]
        sin = sin[None, : q.size(-3), None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
        sin = sin[position_ids].unsqueeze(-2)  # [bs, seq_len, 1, dim/2]
    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_interleave
    else:
        cos = torch.cat([cos, cos], dim=-1)  # [bs, seq_len, 1, dim]
        sin = torch.cat([sin, sin], dim=-1)  # [bs, seq_len, 1, dim]
        rotate_fn = rotate_half

    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)

    return q_embed, k_embed


# # @pytest.mark.apply_rotary_pos_emb
@label("apply_rotary_pos_emb")
# @parametrize("batch_size", [2] if TO_CPU else [4, 8])
# @parametrize("max_seq_len", [16] if TO_CPU else [512, 2048])
# @parametrize("q_heads,k_heads", [(8, 1), (6, 2), (1, 1), (8, 8)])
# @parametrize("head_dim", [8] if TO_CPU else [64, 96, 128, 256])
# @parametrize("dtype", FLOAT_DTYPES)
# @parametrize("rotary_interleaved", [True, False])
# @parametrize("has_pos_id", [True, False])
# def test_apply_rotary_pos_emb(
#     batch_size,
#     max_seq_len,
#     q_heads,
#     k_heads,
#     head_dim,
#     dtype,
#     has_pos_id,
#     rotary_interleaved,
# ):
#     seq_len = torch.randint(1, max_seq_len, (1,)).item()
#     q = torch.randn(
#         (batch_size, seq_len, q_heads, head_dim), dtype=dtype, device=device
#     )
#     k = torch.randn(
#         (batch_size, seq_len, k_heads, head_dim), dtype=dtype, device=device
#     )

#     position_ids = torch.randint(
#         0, max_seq_len, (batch_size, seq_len), device=device
#     )
#     cos, sin = get_rope_cos_sin(max_seq_len, head_dim, dtype, device=device)

#     ref_q = to_reference(q, True)
#     ref_k = to_reference(k, True)
#     ref_cos = to_reference(cos, True)
#     ref_sin = to_reference(sin, True)
#     ref_position_ids = to_reference(position_ids)

#     q_embed_ref, k_embed_ref = torch_apply_rotary_pos_emb(
#         q=ref_q,
#         k=ref_k,
#         cos=ref_cos,
#         sin=ref_sin,
#         position_ids=ref_position_ids if has_pos_id else None,
#         rotary_interleaved=rotary_interleaved,
#     )

#     q_embed_out, k_embed_out = flag_gems.apply_rotary_pos_emb(
#         q=q,
#         k=k,
#         cos=cos,
#         sin=sin,
#         position_ids=position_ids if has_pos_id else None,
#         rotary_interleaved=rotary_interleaved,
#     )

#     gems_assert_close(q_embed_out, q_embed_ref, dtype)
#     gems_assert_close(k_embed_out, k_embed_ref, dtype)


# TODO: failed when EmbeddingSize is small
# @pytest.mark.embedding
@label("embedding")
@parametrize("EmbeddingSize", [1024] if TO_CPU else [4096])
@parametrize("Batch", [2] if TO_CPU else [2, 4])
@parametrize("M", [4] if TO_CPU else [4, 8])
@parametrize("N", [8] if TO_CPU else [128, 256, 4096])
@parametrize("padding_idx", [None, -1, 1, 2])
@parametrize("scale_grad_by_freq", [True, False])
@parametrize("dtype", FLOAT_DTYPES)
def test_embedding(EmbeddingSize, Batch, M, N, padding_idx, scale_grad_by_freq, dtype):
    # if flag_gems.vendor_name == "kunlunxin":
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)

    indices = torch.randint(
        0, EmbeddingSize, (Batch, M), device=device, requires_grad=False
    )
    embedding = torch.randn(
        (EmbeddingSize, N), device=device, dtype=dtype, requires_grad=True
    )
    ref_embedding = to_reference(embedding)
    ref_indices = to_reference(indices)

    ref_out = torch.nn.functional.embedding(
        ref_indices, ref_embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.embedding(
            indices, embedding, padding_idx, scale_grad_by_freq=scale_grad_by_freq
        )
    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_embedding, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, embedding, out_grad)

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


# @pytest.mark.resolve_neg
@label("resolve_neg")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    y = x.conj()
    z = y.imag
    assert z.is_neg()
    with flagbench.use_gems(REGISTERED_OPS):
        out = z.resolve_neg()
    assert not out.is_neg()


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.topk
@label("topk")
@parametrize("batch_size", [4, 8])
@parametrize("hiddensize", [128, 256])
@parametrize("topk", [5])
@parametrize("largest", [True, False])
@parametrize("dtype", FLOAT_DTYPES)
def test_topk(
    batch_size,
    hiddensize,
    topk,
    largest,
    dtype,
):
    x = torch.arange(hiddensize, dtype=dtype, device=device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = to_reference(x)
    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    with flagbench.use_gems(REGISTERED_OPS):
        res_value, res_index = torch.topk(x, topk, largest=largest)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


# @pytest.mark.resolve_conj
@label("resolve_conj")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    y = x.conj()
    assert y.is_conj()
    with flagbench.use_gems(REGISTERED_OPS):
        res_y = y.to(device=device)
        z = res_y.resolve_conj()
    assert not z.is_conj()


# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="AssertionError")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.unique
@label("unique")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", INT_DTYPES)
@parametrize("sorted", [True])
@parametrize("return_inverse", [True, False])
@parametrize("return_counts", [False, True])
def test_accuracy_unique(shape, dtype, sorted, return_inverse, return_counts):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(-10, 10, shape, device=device).to(dtype)
    ref_inp = to_reference(inp, False)

    if return_counts:
        if return_inverse:
            with flagbench.use_gems(REGISTERED_OPS):
                res_out, res_unique_order, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
            gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flagbench.use_gems(REGISTERED_OPS):
                res_out, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
        gems_assert_equal(res_counts, ref_counts)
    else:
        if return_inverse:
            with flagbench.use_gems(REGISTERED_OPS):
                res_out, res_unique_order = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
            gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flagbench.use_gems(REGISTERED_OPS):
                res_out = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()
    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.multinomial
@label("multinomial")
@parametrize("shape", UT_SHAPES_1D + UT_SHAPES_2D)
@parametrize("dtype", [torch.float16, torch.float32])
@parametrize("n_samples", [1000])
def test_accuracy_multinomial_with_replacement(shape, dtype, n_samples):
    if shape[-1] == 1:
        dist = torch.rand(size=shape, dtype=dtype, device=device)
        with flagbench.use_gems(REGISTERED_OPS):
            res_out = torch.multinomial(dist, n_samples, True)
        assert torch.all(res_out == 0)
    else:
        # Mask p% off of the categories and test the sampling results fall in the rest
        for p in (0.1, 0.5, 0.9):
            dist = torch.rand(size=shape, dtype=dtype, device=device)
            dist[torch.rand(shape) < p] = 0
            # Make sure there's at least one non-zero probability
            dist[..., -1] = 0.5
            with flagbench.use_gems(REGISTERED_OPS):
                res_out = torch.multinomial(dist, n_samples, True)
            res_dist = torch.gather(dist, -1, res_out)
            # assert torch.all(res_dist)
            assert torch.sum(res_dist == 0) / res_dist.numel() < 0.001


# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="ZeroDivisionError")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.multinomial
@label("multinomial")
@parametrize("pool", UT_SHAPES_2D)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_multinomial_without_replacement(pool, dtype):
    dist = torch.rand(size=pool, dtype=dtype, device=device)
    k = pool[-1]
    if k > 1:
        ns = [k // 2, k]
    else:
        ns = [1]
    for n in ns:
        with flagbench.use_gems(REGISTERED_OPS):
            out = torch.multinomial(dist, n, False)
        # Verifies uniqueness
        idx_cnt = torch.nn.functional.one_hot(out).sum(1)
        assert torch.all(idx_cnt <= 1)


# @pytest.mark.pad
@label("pad")
@parametrize("shape", [[1024, 1024], [64, 64, 64, 64]])
# @parametrize("dtype", [torch.float32] if TO_CPU else FLOAT_DTYPES)
@parametrize("dtype", FLOAT_DTYPES)
@parametrize("pad_mode", ["constant", "reflect", "replicate", "circular"])
@parametrize("contiguous", [True, False])
def test_pad(shape, dtype, pad_mode, contiguous):
    # if flag_gems.vendor_name == "kunlunxin":
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)

    x = torch.randn(size=shape, dtype=dtype, device=device)
    if not contiguous:
        x = x[::2, ::2]

    ref_x = to_reference(x)
    if ref_x.dtype == torch.float16:
        ref_x = ref_x.to(torch.float32)

    rank = x.ndim
    pad_params = list(
        torch.randint(0, 10, (rank * 2,), dtype=torch.int32, device="cpu")
        if pad_mode == "constant"
        else torch.randint(0, 10, (rank,), dtype=torch.int32, device="cpu")
    )
    pad_value = float(torch.randint(0, 1024, (1,), dtype=torch.int32, device="cpu"))

    if pad_mode != "constant":
        pad_params = [(pad_val + 2 - 1) // 2 * 2 for pad_val in pad_params]
        pad_value = None

    ref_pad_params = [to_reference(pad_param) for pad_param in pad_params]

    ref_out = torch.nn.functional.pad(ref_x, ref_pad_params, pad_mode, pad_value)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.nn.functional.pad(x, pad_params, pad_mode, pad_value)

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "cambricon", reason="fix")
# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="torch not supports half yet")
# @pytest.mark.upsample_bicubic2d_aa
@label("upsample_bicubic2d_aa")
@parametrize("align_corners", [False, True])
@parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.7)])
@parametrize(
    "shape",
    [
        (32, 16, 128, 128),
        (15, 37, 256, 256),
        (3, 5, 127, 127),
        (128, 192, 42, 51),
        (3, 7, 1023, 1025),
    ],
)
@parametrize("dtype", FLOAT_DTYPES)
def test_upsample_bicubic2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=device)
    ref_i = to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch._C._nn._upsample_bicubic2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch._C._nn._upsample_bicubic2d_aa(
            input, output_size=output_size, align_corners=align_corners
        )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


# @pytest.mark.upsample_nearest2d
@label("upsample_nearest2d")
@parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@parametrize("shape", UPSAMPLE_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=device)
    ref_i = to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)


# import torch.nn.functional as F
# FLOAT_DTYPES = [torch.float32, torch.float64, torch.float16]
# AVG_POOL_SHAPES = [
#     (1, 3, 32, 32),
#     (2, 16, 64, 64),
#     (4, 8, 128, 128),
# ]

# @label("avg_pool2d")
# # @parametrize("kernel_size", [(2, 2), (3, 3), (5, 5)])
# @parametrize("kernel_size, padding", [((2, 2), (0, 0)), ((3, 3), (1, 1)), ((5, 5), (2, 2))])
# @parametrize("stride", [None, (2, 2), (1, 1)])
# # @parametrize("padding", [(0, 0), (1, 1), (2, 2)])
# @parametrize("ceil_mode", [False, True])
# @parametrize("count_include_pad", [True, False])
# @parametrize("divisor_override", [None, 2, 3])
# @parametrize("shape", AVG_POOL_SHAPES)
# @parametrize("dtype", FLOAT_DTYPES)
# def test_avg_pool2d(dtype, shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
#     # 生成输入张量
#     input = torch.randn(shape, dtype=dtype, device=device)

#     # 转换为参考格式
#     ref_i = input.to(torch.float32)

#     # 定义 stride，如果为 None，则使用 kernel_size
#     actual_stride = stride if stride is not None else kernel_size

#     # 计算参考输出
#     ref_out = F.avg_pool2d(
#         ref_i,
#         kernel_size=kernel_size,
#         stride=actual_stride,
#         padding=padding,
#         ceil_mode=ceil_mode,
#         count_include_pad=count_include_pad,
#         divisor_override=divisor_override
#     ).to(dtype)

#     # 使用被测试的操作进行计算
#     # with flagbench.use_gems(REGISTERED_OPS):
#     res_out = flagbench.avg_pool2d(
#         input,
#         kernel_size=kernel_size,
#         stride=actual_stride,
#         padding=padding,
#         ceil_mode=ceil_mode,
#         count_include_pad=count_include_pad,
#         divisor_override=divisor_override
#     )

#     # 比较结果
#     gems_assert_close(res_out, ref_out, dtype)


# @pytest.mark.arange
@label("arange")
@parametrize("start", [0, 1, 3])
@parametrize("step", [1, 2, 5])
@parametrize("end", [128, 256, 1024])
@parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES + [None])
@parametrize("device", [device, None])
@parametrize(
    "pin_memory", [False, None]
)  # Since triton only target to GPU, pin_memory only used in CPU tensors.
def test_arange(start, step, end, dtype, device, pin_memory):
    if TO_CPU:
        return
    ref_out = torch.arange(
        start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
    )
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.arange(
            start, end, step, dtype=dtype, device=device, pin_memory=pin_memory
        )

    gems_assert_equal(res_out, ref_out)


# # @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="AssertionError")
# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.isin
@label("isin")
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", INT_DTYPES)
@parametrize("assume_unique", [False, True])
@parametrize("invert", [False, True])
def test_accuracy_isin(shape, dtype, assume_unique, invert):
    inp1 = torch.randint(-100, 100, shape, device=device).to(dtype)
    test_numel = inp1.numel() // 2 if inp1.numel() > 1 else 1
    test_shape = (test_numel,)
    inp2 = torch.randint(-10, 10, test_shape, device=device).to(dtype)
    inp1.ravel()[-1] = 0
    if assume_unique:
        inp1 = torch.unique(inp1.cpu()).to(device)
        inp2 = torch.unique(inp2.cpu()).to(device)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.isin(inp1, inp2, assume_unique=assume_unique, invert=invert)
    ref_out = torch.isin(ref_inp1, ref_inp2, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res_out, ref_out)

    inp1_s = inp1.ravel()[0].item()
    with flagbench.use_gems(REGISTERED_OPS):
        res1_out = torch.isin(inp1_s, inp2, assume_unique=assume_unique, invert=invert)
    ref1_out = torch.isin(inp1_s, ref_inp2, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res1_out, ref1_out)

    inp2_s = inp2.ravel()[0].item()
    with flagbench.use_gems(REGISTERED_OPS):
        res2_out = torch.isin(inp1, inp2_s, assume_unique=assume_unique, invert=invert)
    ref2_out = torch.isin(ref_inp1, inp2_s, assume_unique=assume_unique, invert=invert)
    gems_assert_equal(res2_out, ref2_out)

    inp0 = torch.tensor([], device=device)
    ref_inp0 = to_reference(inp0, False)
    with flagbench.use_gems(REGISTERED_OPS):
        res0_out = torch.isin(inp0, inp2, assume_unique=assume_unique, invert=invert)
    ref0_out = torch.isin(
        ref_inp0, ref_inp2, assume_unique=assume_unique, invert=invert
    )
    gems_assert_equal(res0_out, ref0_out)


# @pytest.mark.fill
@label("fill")
@parametrize("value", [0, 1, 9])
@parametrize("shape", SPECIAL_SHAPES)
@parametrize("dtype", FLOAT_DTYPES)
def test_fill(value, shape, dtype):
    # Test fill.Scalar
    x = torch.ones(shape, device=device, dtype=dtype)
    ref_x = to_reference(x, False)

    ref_out = torch.fill(ref_x, value)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.fill(x, value)

    gems_assert_equal(res_out, ref_out)

    # Test fill.Tensor
    value_tensor = torch.tensor(value, device=device, dtype=dtype)
    ref_value_tensor = to_reference(value_tensor, False)
    ref_out_tensor = torch.fill(ref_x, ref_value_tensor)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out_tensor = torch.fill(x, value_tensor)

    gems_assert_equal(res_out_tensor, ref_out_tensor)


CAMBRICON_STACK_SHAPES = [
    [
        (8, 8, 128),
        (8, 8, 128),
        (8, 8, 128),
    ],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]
# STACK_SHAPES_TEST = STACK_SHAPES + (
#     CAMBRICON_STACK_SHAPES if flag_gems.vendor_name == "cambricon" else []
# )
STACK_SHAPES_TEST = STACK_SHAPES


# @pytest.mark.stack
@label("stack")
@parametrize("shape", STACK_SHAPES_TEST)
@parametrize("dim", STACK_DIM_LIST)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_stack(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                device
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.stack(inp, dim)
    gems_assert_equal(res_out, ref_out)


HSTACK_SHAPES = [
    [(8,), (16,)],
    [(16, 256), (16, 128)],
    [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
]


# @pytest.mark.hstack
@label("hstack")
@parametrize("shape", HSTACK_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_hstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                device
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.hstack(ref_inp)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.hstack(inp)
    gems_assert_equal(res_out, ref_out)


HSTACK_EXCEPTION_SHAPES = [
    [(16, 256), (16,)],
    [(16, 256), (8, 128)],
]


# @pytest.mark.hstack
@label("hstack")
@parametrize("shape", HSTACK_EXCEPTION_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_exception_hstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                device
            )
            for s in shape
        ]

    with pytest.raises(RuntimeError):
        with flagbench.use_gems(REGISTERED_OPS):
            _ = torch.hstack(inp)


CAT_SHAPES = [
    [(1, 32), (8, 32)],
    [(16, 128), (32, 128)],
    [(1024, 1024), (1024, 1024)],
    [(1, 1024, 256), (8, 1024, 256), (16, 1024, 256)],
    [(16, 320, 15), (32, 320, 15), (64, 320, 15)],
    [(16, 128, 64, 64), (16, 128, 64, 64), (24, 128, 64, 64), (32, 128, 64, 64)],
]


def gen_cat_shapes_dim(shapes):
    results = []
    for tensor_shapes in shapes:
        assert all(
            [len(s) == len(tensor_shapes[0]) for s in tensor_shapes]
        ), "All tensor rank must agree."
        assert all(
            [s[-1] == tensor_shapes[0][-1] for s in tensor_shapes]
        ), "All tensor must have same shape except cat dim."
        rank = len(tensor_shapes[0])
        results.append([tensor_shapes, 0])
        for dim in range(1, rank):
            results.append(
                [[(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes], dim]
            )
            results.append(
                [
                    [(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes],
                    dim - rank,
                ]
            )
    return results


# @pytest.mark.cat
@label("cat")
@parametrize("shape, dim", gen_cat_shapes_dim(CAT_SHAPES))
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_cat(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                device
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.cat
@label("cat")
@parametrize(
    "shape, dim",
    [
        (((0, 3), (2, 3)), 0),
        (((0, 3), (0, 3)), 0),
        (((0,), (0,)), 0),
    ],
)
@parametrize("dtype", [torch.float32])
def test_accuracy_cat_empty_tensor(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.cat(inp, dim)
    gems_assert_equal(res_out, ref_out)


VSTACK_SHAPES = [
    [(3,), (3,)],
    [(3, 33), (7, 33)],
    [(13, 3, 333), (17, 3, 333), (7, 3, 333)],
    [
        (13, 3, 64, 5, 2),
        (16, 3, 64, 5, 2),
        (7, 3, 64, 5, 2),
        (4, 3, 64, 5, 2),
        (1, 3, 64, 5, 2),
    ],
]

CAMBRICON_VSTACK_SHAPES = [
    [(16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64), (16, 128, 64, 64)],
    [
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
        (32, 64, 128, 8),
    ],
]
# VSTACK_SHAPES_TEST = VSTACK_SHAPES + (
#     CAMBRICON_VSTACK_SHAPES if flag_gems.vendor_name == "cambricon" else []
# )
VSTACK_SHAPES_TEST = VSTACK_SHAPES

# @pytest.mark.vstack
@label("vstack")
@parametrize("shape", VSTACK_SHAPES_TEST)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_vstack(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                device
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.vstack(ref_inp)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.vstack(inp)
    gems_assert_equal(res_out, ref_out)


REPEAT_INTERLEAVE_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
REPEAT_INTERLEAVE_REPEATS = [2]
REPEAT_INTERLEAVE_DIM = [-1, 0, None]


# @pytest.mark.repeat_interleave
@label("repeat_interleave")
@parametrize("shape", REPEAT_INTERLEAVE_SHAPES + [(1,)])
@parametrize("dim", REPEAT_INTERLEAVE_DIM)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.repeat_interleave
@label("repeat_interleave")
@parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@parametrize("dim", REPEAT_INTERLEAVE_DIM)
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)[::2]
    repeats = 2
    ref_inp = to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.repeat_interleave(ref_inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.repeat_interleave
@label("repeat_interleave")
@parametrize("shape", UT_SHAPES_1D)
@parametrize("dtype", [torch.int32])
def test_accuracy_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device=device)
    ref_repeats = to_reference(repeats)
    ref_out = torch.repeat_interleave(ref_repeats)

    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.repeat_interleave(repeats)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.repeat_interleave
@label("repeat_interleave")
@parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@parametrize("dim", [-1, 0, 1])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat_interleave_self_tensor(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=device)
    repeats = torch.randint(0, 30, (shape[dim],), device=device)
    ref_inp = to_reference(inp)
    ref_repeats = to_reference(repeats)

    ref_out = torch.repeat_interleave(ref_inp, ref_repeats, dim)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.repeat_interleave(inp, repeats, dim)
    gems_assert_equal(res_out, ref_out)


# @pytest.mark.diag
@label("diag")
@parametrize("shape", UT_SHAPES_1D + UT_SHAPES_2D)
@parametrize("diagonal", [-2, -1, 0, 1, 2])
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag(shape, diagonal, dtype):
    # if flag_gems.vendor_name == "kunlunxin":
    #     torch.manual_seed(0)
    #     torch.cuda.manual_seed_all(0)

    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            device
        )
    else:
        inp = torch.randint(0, 0x7FFF, size=shape, dtype=dtype, device="cpu").to(
            device
        )
    ref_inp = to_reference(inp)

    ref_out = torch.diag(ref_inp, diagonal)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.diag(inp, diagonal)
    gems_assert_equal(res_out, ref_out)


def get_dim1_dim2(o_rank):
    dims = list(range(-o_rank, o_rank))
    return [
        p for p in itertools.permutations(dims, 2) if (p[0] % o_rank) != (p[1] % o_rank)
    ]


def get_diag_embed_shape_and_dims():
    shapes = [
        (1024,),
        (1024, 1024),
    ]
    # [(shape, dim1, dim2)]
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s) + 1)
        result.extend([(s, dim1, dim2) for dim1, dim2 in dim_pairs])

    return result


# @pytest.mark.diag_embed
@label("diag_embed")
@parametrize("shape, dim1, dim2", get_diag_embed_shape_and_dims())
@parametrize("offset", [-1, 0, 1])
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_diag_embed(shape, dtype, offset, dim1, dim2):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    elif dtype in INT_DTYPES:
        inp = torch.randint(
            low=0, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    else:
        inp = torch.randint(low=0, high=2, size=shape, dtype=dtype, device="cpu").to(
            device
        )

    ref_inp = to_reference(inp)

    ref_out = torch.diag_embed(ref_inp, offset, dim1, dim2)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.diag_embed(inp, offset, dim1, dim2)
    gems_assert_equal(res_out, ref_out)


def get_diagonal_backward_shape_and_dims():
    shapes = SPECIAL_SHAPES
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s))
        result.extend([(s, dim1, dim2) for dim1, dim2 in dim_pairs])

    return result


# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="MUSA error: unknown error")
# @pytest.mark.diagonal_backward
@label("diagonal_backward")
@parametrize("shape, dim1, dim2", get_diagonal_backward_shape_and_dims())
@parametrize("offset", [-1, 0, 1])
@parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_diagonal_backward(shape, dtype, dim1, dim2, offset):
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    ref_inp = to_reference(inp)

    ref_out = torch.diagonal(ref_inp, offset, dim1, dim2)
    res_out = torch.diagonal(inp, offset, dim1, dim2)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flagbench.use_gems(REGISTERED_OPS):
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    res_out = to_reference(res_out)
    res_in_grad = to_reference(res_in_grad)
    gems_assert_equal(res_out, ref_out)
    gems_assert_equal(res_in_grad, ref_in_grad)


# # @pytest.mark.skipif
# @label("skipif")(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
# @pytest.mark.sort
@label("sort")
@parametrize("batch_size", [4, 8])
@parametrize("hiddensize", [1, 256, 2048, 9333, 65536])
@parametrize("descending", [True, False])
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@parametrize("dim", [0, -1])
def test_sort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in FLOAT_DTYPES:
        x = torch.empty((hiddensize,), dtype=dtype, device=device)
        tmp = torch.tensor(0, dtype=dtype)
        inf = torch.tensor(float("inf"), dtype=dtype)
        for i in range(0, hiddensize):
            x[i] = tmp.item()
            tmp = torch.nextafter(tmp, inf)
            if tmp.item() == inf.item():
                hiddensize = i
                x = x[:hiddensize]
                break
    else:
        if device == "musa" and dtype == torch.int16:
            # arange short type on torch of mthreads not supported yet.
            x = torch.arange(hiddensize, dtype=torch.int32, device=device).to(
                dtype
            )
        else:
            x = torch.arange(hiddensize, dtype=dtype, device=device)
    y = torch.empty((batch_size, hiddensize), dtype=dtype, device=device)

    # Each row use different shuffled index.
    col_indices = torch.randperm(x.size(0))
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(0))
        y[bsz, :] = x[col_indices]
    if dim == 0:
        y = torch.movedim(y, dim, -1)
    ref_y = to_reference(y)
    ref_value, ref_index = torch.sort(ref_y, dim=dim, descending=descending)

    with flagbench.use_gems(REGISTERED_OPS):
        res_value, res_index = torch.sort(y, dim=dim, descending=descending)

    gems_assert_close(res_value, ref_value, dtype)
    gems_assert_equal(res_index, ref_index)


# @pytest.mark.skipif
# @label("skipif")(device == "musa", reason="ZeroDivisionError")
# @pytest.mark.kron
@label("kron")
@parametrize("shape", KRON_SHAPES)
@parametrize("dtype", FLOAT_DTYPES + INT_DTYPES + BOOL_TYPES)
def test_accuracy_kron(shape, dtype):
    if dtype in INT_DTYPES:
        inp1 = torch.randint(
            low=-10, high=10, size=shape[0], dtype=dtype, device=device
        )
        inp2 = torch.randint(
            low=-10, high=10, size=shape[1], dtype=dtype, device=device
        )
    elif dtype in FLOAT_DTYPES:
        inp1 = torch.randn(shape[0], dtype=dtype, device=device)
        inp2 = torch.randn(shape[1], dtype=dtype, device=device)
    else:
        inp1 = torch.randint(0, 2, size=shape[0], dtype=dtype, device=device)
        inp2 = torch.randint(0, 2, size=shape[1], dtype=dtype, device=device)

    # if flag_gems.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
    #     # Pytorch 2.0.1 Bfloat16 CPU Backend Precision Failed
    #     inp1 = torch.randn(shape[0], dtype=torch.float32, device=device)
    #     inp2 = torch.randn(shape[1], dtype=torch.float32, device=device)
    if dtype == torch.bfloat16:
        # Pytorch 2.0.1 Bfloat16 CPU Backend Precision Failed
        inp1 = torch.randn(shape[0], dtype=torch.float32, device=device)
        inp2 = torch.randn(shape[1], dtype=torch.float32, device=device)

    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.kron(ref_inp1, ref_inp2)
    with flagbench.use_gems(REGISTERED_OPS):
        res_out = torch.kron(inp1, inp2)

    gems_assert_equal(res_out, ref_out)
