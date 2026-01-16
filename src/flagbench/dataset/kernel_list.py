import torch
from torch import FunctionSchema
from enum import Enum
from typing import Dict
from .dataloader import TorchOpsLoader
from logging import getLogger
import os

logger = getLogger(__name__)


class Autograd(Enum):
    enable = True
    disable = False

    @classmethod
    def get_optional_value(cls):
        return [member.name for member in cls]

class DynamicImplInfo:
    def __init__(self):
        self._cache = {}
        self._cache_errors = []
        self.loader = TorchOpsLoader(to_str=False)
        self.namespaces = self.loader.list_namespaces()

    def get(self, api: str, *, namespace: str = "aten"):
        if "::" in api:
            namespace, api = api.split("::", 1)
        if api in self._cache:
            return self._cache[api]
        if api in self._cache_errors:
            return None
        assert namespace in self.namespaces, f"namespace {namespace} not found"
        try:
            schemas = self.loader.get_operator(namespace, api).schemas
            self._cache[api] = self._schemas_to_impl_info(schemas, namespace, api)
            return self._cache[api]
        except Exception as e:
            self._cache_errors.append(api)
            logger.error(f"get impl info for {namespace}.{api} error: {e}")
            return None
        
    def _schemas_to_impl_info(self, schemas, namespace: str, api: str):
        impl_info = []
        for overload_name, schema in schemas.items():
            impl_info.append((f"{api}{'.' + overload_name if overload_name != '' else ''}", Autograd.disable))
        return impl_info

    def __contains__(self, namespace_api_tuple: tuple[str, str] | str):
        if isinstance(namespace_api_tuple, str):
            namespace_api_tuple = ("aten", namespace_api_tuple)
        namespace, api = namespace_api_tuple
        impl_info = self.get(api, namespace=namespace)
        return impl_info is not None
    
    def __getitem__(self, namespace_api_tuple: tuple[str, str] | str):
        if isinstance(namespace_api_tuple, str):
            namespace_api_tuple = ("aten", namespace_api_tuple)
        namespace, api = namespace_api_tuple
        impl_info = self.get(api, namespace=namespace)
        if impl_info is None:
            raise KeyError(f"impl info for {namespace}.{api} not found")
        return impl_info


IMPL_INFO = {
    "abs": [("abs", Autograd.disable)],
    "abs_": [("abs_", Autograd.disable)],
    "add": [("add.Tensor", Autograd.disable)],
    "add_": [("add_.Tensor", Autograd.disable)],
    "addmm": [("addmm", Autograd.disable)],
    "addmv": [("addmv", Autograd.disable)],
    "arange": [
        ("arange.start_step", Autograd.disable),
        ("arange.start", Autograd.disable),
        ("arange", Autograd.disable),
    ],
    "batch_norm": [("batch_norm", Autograd.enable)],
    "bitwise_and": [
        ("bitwise_and.Tensor", Autograd.disable),
        ("bitwise_and.Scalar", Autograd.disable),
        ("bitwise_and.Scalar_Tensor", Autograd.disable),
    ],
    "bitwise_and_": [
        ("bitwise_and_.Tensor_", Autograd.disable),
        ("bitwise_and_.Scalar", Autograd.disable),
    ],
    "bitwise_not": [("bitwise_not", Autograd.disable)],
    "bitwise_not_": [("bitwise_not_", Autograd.disable)],
    "bitwise_or": [
        ("bitwise_or.Tensor", Autograd.disable),
        ("bitwise_or.Scalar", Autograd.disable),
        ("bitwise_or.Scalar_Tensor", Autograd.disable),
    ],
    "bitwise_or_": [
        ("bitwise_or_.Tensor", Autograd.disable),
        ("bitwise_or_.Scalar", Autograd.disable),
    ],
    "bmm": [("bmm", Autograd.disable)],
    "clamp": [
        ("clamp", Autograd.disable),
        ("clamp.Tensor", Autograd.disable),
    ],
    "clamp_": [
        ("clamp_", Autograd.disable),
        ("clamp_.Tensor", Autograd.disable),
    ],
    "cos": [("cos", Autograd.disable)],
    "cos_": [("cos_", Autograd.disable)],
    "pad": [("pad", Autograd.disable)],
    "constant_pad_nd": [("constant_pad_nd", Autograd.disable)],
    "cumsum": [("cumsum", Autograd.disable)],
    "cummin": [("cummin", Autograd.disable)],
    "div": [
        ("div.Tensor", Autograd.disable),
        ("div.Scalar", Autograd.disable),
        ("div.Tensor_mode", Autograd.disable),
        ("div.Scalar_mode", Autograd.disable),
    ],
    "div_": [
        ("div_.Tensor", Autograd.disable),
        ("div_.Scalar", Autograd.disable),
        ("div_.Tensor_mode", Autograd.disable),
        ("div_.Scalar_mode", Autograd.disable),
    ],
    "divide": [
        ("divide.Tensor", Autograd.disable),
        ("divide.Scalar", Autograd.disable),
        ("divide.Tensor_mode", Autograd.disable),
        ("divide.Scalar_mode", Autograd.disable),
    ],
    "divide_": [
        ("divide_.Tensor", Autograd.disable),
        ("divide_.Scalar", Autograd.disable),
        ("divide_.Tensor_mode", Autograd.disable),
        ("divide_.Scalar_mode", Autograd.disable),
    ],
    "true_divide": [
        ("true_divide.Tensor", Autograd.disable),
        ("true_divide.Scalar", Autograd.disable),
    ],
    "true_divide_": [
        ("true_divide_.Tensor", Autograd.disable),
        ("true_divide_.Scalar", Autograd.disable),
    ],
    "floor_divide": [
        ("floor_divide", Autograd.disable),
        ("floor_divide.Scalar", Autograd.disable),
    ],
    "floor_divide_": [
        ("floor_divide_.Tensor", Autograd.disable),
        ("floor_divide_.Scalar", Autograd.disable),
    ],
    "remainder": [
        ("remainder.Tensor", Autograd.disable),
        ("remainder.Scalar", Autograd.disable),
        ("remainder.Scalar_Tensor", Autograd.disable),
    ],
    "remainder_": [
        ("remainder_.Tensor", Autograd.disable),
        ("remainder_.Scalar", Autograd.disable),
    ],
    "native_dropout": [("native_dropout", Autograd.enable)],
    "erf": [("erf", Autograd.disable)],
    "erf_": [("erf_", Autograd.disable)],
    "embedding": [("embedding", Autograd.enable)],
    "eq": [
        ("eq.Tensor", Autograd.disable), 
        ("eq.Scalar", Autograd.disable),
    ],
    "exp": [("exp", Autograd.disable)],
    "exp_": [("exp_", Autograd.disable)],
    "exponential_": [("exponential_", Autograd.disable)],
    "ge": [
        ("ge.Tensor", Autograd.disable),
        ("ge.Scalar", Autograd.disable),
    ],
    "gelu": [("gelu", Autograd.enable)],
    "gelu_": [("gelu_", Autograd.enable)],
    "native_group_norm": [("native_group_norm", Autograd.enable)],
    "_weight_norm_interface": [("_weight_norm_interface", Autograd.enable)],
    "_weight_norm": [("_weight_norm", Autograd.enable)],
    "gt": [
        ("gt.Tensor", Autograd.disable),
        ("gt.Scalar", Autograd.disable),
    ],
    "instance_norm": [("instance_norm", Autograd.enable)],
    "isfinite": [("isfinite", Autograd.disable)],
    "isin": [
        ("isin.Tensor_Tensor", Autograd.disable),
        ("isin.Scalar_Tensor", Autograd.disable),
        ("isin.Tensor_Scalar", Autograd.disable),
    ],
    "isinf": [("isinf", Autograd.disable)],
    "isnan": [("isnan", Autograd.disable)],
    "minimum": [("minimum", Autograd.disable)],
    "maximum": [("maximum", Autograd.disable)],
    "native_layer_norm": [("native_layer_norm", Autograd.enable)],
    "le": [
        ("le.Tensor", Autograd.disable),
        ("le.Scalar", Autograd.disable),
    ],
    "lt": [
        ("lt.Tensor", Autograd.disable),
        ("lt.Scalar", Autograd.disable),
    ],
    "rms_norm": [("rms_norm", Autograd.disable)],
    "rand": [("rand", Autograd.disable)],
    "randn": [("randn", Autograd.disable)],
    "rand_like": [("rand_like", Autograd.disable)],
    "randn_like": [("randn_like", Autograd.disable)],
    "zeros": [("zeros", Autograd.disable)],
    "ones": [("ones", Autograd.disable)],
    "full": [("full", Autograd.disable)],
    "zeros_like": [("zeros_like", Autograd.disable)],
    "ones_like": [("ones_like", Autograd.disable)],
    "full_like": [("full_like", Autograd.disable)],
    "resolve_neg": [("resolve_neg", Autograd.disable)],
    "resolve_conj": [("resolve_conj", Autograd.disable)],
    "normal": [
        ("normal.Tensor_float", Autograd.disable),
        ("normal.float_Tensor", Autograd.disable),
        ("normal.Tensor_Tensor", Autograd.disable),
    ],
    "uniform_": [("uniform_", Autograd.disable)],
    "mean": [
        ("mean", Autograd.disable),
        ("mean.dim", Autograd.disable),
    ],
    "mm": [("mm", Autograd.disable)],
    "mul": [("mul.Tensor", Autograd.disable)],
    "mul_": [("mul_.Tensor", Autograd.disable)],
    "multinomial": [("multinomial", Autograd.disable)],
    "mv": [("mv", Autograd.disable)],
    "ne": [
        ("ne.Tensor", Autograd.disable),
        ("ne.Scalar", Autograd.disable),
    ],
    "neg": [("neg", Autograd.disable)],
    "neg_": [("neg_", Autograd.disable)],
    "pow": [
        ("pow.Scalar", Autograd.disable),
        ("pow.Tensor_Scalar", Autograd.disable),
        ("pow.Tensor_Tensor", Autograd.disable),
    ],
    "pow_": [
        ("pow_.Scalar", Autograd.disable),
        ("pow_.Tensor", Autograd.disable),
    ],
    "reciprocal": [("reciprocal", Autograd.disable)],
    "reciprocal_": [("reciprocal_", Autograd.disable)],
    "relu": [("relu", Autograd.enable)],
    "relu_": [("relu_", Autograd.enable)],
    "rsqrt": [("rsqrt", Autograd.disable)],
    "rsqrt_": [("rsqrt_", Autograd.disable)],
    "sigmoid": [("sigmoid", Autograd.enable)],
    "sigmoid_": [("sigmoid_", Autograd.enable)],
    "silu": [("silu", Autograd.enable)],
    "silu_": [("silu_", Autograd.enable)],
    "sin": [("sin", Autograd.disable)],
    "sin_": [("sin_", Autograd.disable)],
    "softmax": [("softmax.int", Autograd.enable)],
    "sort": [("sort", Autograd.disable)],
    "sub": [("sub.Tensor", Autograd.disable)],
    "sub_": [("sub_.Tensor", Autograd.disable)],
    "tanh": [("tanh", Autograd.enable)],
    "tanh_": [("tanh_", Autograd.enable)],
    "triu": [("triu", Autograd.disable)],
    "var_mean": [("var_mean.correction", Autograd.disable)],
    "linalg_vector_norm": [("linalg_vector_norm", Autograd.disable)],
    "where": [
        ("where.self_out", Autograd.disable),
        ("where.self", Autograd.disable),
        ("where.ScalarSelf", Autograd.disable),
        ("where.ScalarOther", Autograd.disable),
    ],
    "max": [
        ("max", Autograd.disable),
        ("max.dim", Autograd.disable),
    ],
    "min": [
        ("min", Autograd.disable),
        ("min.dim", Autograd.disable),
    ],
    "amax": [("amax", Autograd.disable)],
    "argmax": [("argmax", Autograd.disable)],
    "argmin": [("argmin", Autograd.disable)],
    "prod": [
        ("prod", Autograd.disable),
        ("prod.dim_int", Autograd.disable),
    ],
    "sum": [
        ("sum", Autograd.disable),
        ("sum.dim_IntList", Autograd.disable),
    ],
    "scaled_dot_product_attention": [("scaled_dot_product_attention", Autograd.disable)],
    "all": [
        ("all", Autograd.disable),
        ("all.dim", Autograd.disable),
        ("all.dims", Autograd.disable),
    ],
    "any": [
        ("any", Autograd.disable),
        ("any.dim", Autograd.disable),
        ("any.dims", Autograd.disable),
    ],
    "quantile": [("quantile", Autograd.disable)],
    "log_softmax": [("log_softmax.int", Autograd.enable)],
    "outer": [("outer", Autograd.enable)],
    "cross_entropy": [("cross_entropy_loss", Autograd.enable)],
    "nll_loss_forward": [("nll_loss_forward", Autograd.disable)],
    "nll_loss_backward": [("nll_loss_backward", Autograd.disable)],
    "nll_loss2d_forward": [("nll_loss2d_forward", Autograd.disable)],
    "nll_loss2d_backward": [("nll_loss2d_backward", Autograd.disable)],
    "scatter": [
        ("scatter.src", Autograd.disable),
        ("scatter.reduce", Autograd.disable),
    ],
    "gather": [("gather", Autograd.disable)],
    "gather_backward": [("gather_backward", Autograd.disable)],
    "isclose": [("isclose", Autograd.disable)],
    "allclose": [("allclose", Autograd.disable)],
    "fill": [
        ("fill.Scalar", Autograd.disable),
        ("fill.Tensor", Autograd.disable),
    ],
    "flip": [("flip", Autograd.disable)],
    "slice_scatter": [("slice_scatter", Autograd.disable)],
    "select_scatter": [("select_scatter", Autograd.disable)],
    "index_select": [("index_select", Autograd.disable)],
    "tile": [("tile", Autograd.disable)],
    "masked_fill": [
        ("masked_fill.Tensor", Autograd.disable),
        ("masked_fill.Scalar", Autograd.disable),
    ],
    "masked_fill_": [
        ("masked_fill_.Tensor", Autograd.disable),
        ("masked_fill_.Scalar", Autograd.disable),
    ],
    "_unique2": [("_unique2", Autograd.disable)],
    "_upsample_bicubic2d_aa": [("_upsample_bicubic2d_aa", Autograd.disable)],
    "upsample_nearest2d": [("upsample_nearest2d", Autograd.disable)],
    "nonzero": [("nonzero", Autograd.disable)],
    "repeat": [("repeat", Autograd.disable)],
    "masked_select": [("masked_select", Autograd.disable)],
    "stack": [("stack", Autograd.disable)],
    "hstack": [("hstack", Autograd.disable)],
    "cat": [("cat", Autograd.disable)],
    "repeat_interleave": [
        ("repeat_interleave.self_int", Autograd.disable),
        ("repeat_interleave.Tensor", Autograd.disable),
        ("repeat_interleave.self_Tensor", Autograd.disable),
    ],
    "vstack": [("vstack", Autograd.disable)],
    "randperm": [("randperm", Autograd.disable)],
    "diag": [("diag", Autograd.disable)],
    "diag_embed": [("diag_embed", Autograd.disable)],
    "diagonal_backward": [("diagonal_backward", Autograd.disable)],
    "index_add": [("index_add", Autograd.disable)],
    # "index_fill": [("index_fill", Autograd.disable)],
    "count_nonzero": [("count_nonzero", Autograd.disable)],
    "logical_or": [("logical_or", Autograd.disable)],
    "logical_and": [("logical_and", Autograd.disable)],
    "logical_xor": [("logical_xor", Autograd.disable)],
    "logical_not": [("logical_not", Autograd.disable)],
    "kron": [("kron", Autograd.disable)],
    "elu": [("elu", Autograd.disable)],
    "index_put": [("index_put", Autograd.disable)],
    "log_sigmoid": [("log_sigmoid", Autograd.disable)],
    "vdot": [("vdot", Autograd.disable)],
    "mse_loss": [("mse_loss", Autograd.disable)],
}


PYTORCH_OPERATORS = {
    'torch.abs': torch.abs,
    'torch.abs_': torch.abs_,
    'torch.add': torch.add,
    'torch.Tensor.add_': torch.Tensor.add_,
    'torch.addmm': torch.addmm,
    'torch.addmv': torch.addmv,
    'torch.all': torch.all,
    'torch.allclose': torch.allclose,
    'torch.amax': torch.amax,
    'torch.angle': torch.angle,
    'torch.any': torch.any,
    'torch.arange': torch.arange,
    'torch.argmax': torch.argmax,
    'torch.argmin': torch.argmin,
    'torch.batch_norm': torch.batch_norm,
    'torch.bitwise_and': torch.bitwise_and,
    'torch.Tensor.bitwise_and_': torch.Tensor.bitwise_and_,
    'torch.bitwise_not': torch.bitwise_not,
    'torch.Tensor.bitwise_not_': torch.Tensor.bitwise_not_,
    'torch.bitwise_or': torch.bitwise_or,
    'torch.Tensor.bitwise_or_': torch.Tensor.bitwise_or_,
    'torch.bmm': torch.bmm,
    'torch.cat': torch.cat,
    'torch.clamp': torch.clamp,
    'torch.clamp_': torch.clamp_,
    'torch.Tensor.contiguous': torch.Tensor.contiguous,
    'torch.conv1d': torch.conv1d,
    'torch.conv2d': torch.conv2d,
    'torch.cos': torch.cos,
    'torch.cos_': torch.cos_,
    'torch.count_nonzero': torch.count_nonzero,
    'torch.nn.functional.cross_entropy': torch.nn.functional.cross_entropy,
    'torch.cummax': torch.cummax,
    'torch.cummin': torch.cummin,
    # 'torch.cumsum': torch.cumsum,
    'torch.diag': torch.diag,
    'torch.diag_embed': torch.diag_embed,
    'torch.diagonal': torch.diagonal,
    'torch.div': torch.div,
    'torch.Tensor.div_': torch.Tensor.div_,
    # 'torch.dot': torch.dot,
    'torch.dropout': torch.dropout,
    'torch.nn.functional.elu': torch.nn.functional.elu,
    # 'torch.nn.functional.elu_': torch.nn.functional.elu_,
    'torch.embedding': torch.embedding,
    'torch.eq': torch.eq,
    'torch.erf': torch.erf,
    'torch.erf_': torch.erf_,
    'torch.exp': torch.exp,
    'torch.exp_': torch.exp_,
    'torch.Tensor.exponential_': torch.Tensor.exponential_,
    'torch.eye': torch.eye,
    'torch.fill': torch.fill,
    'torch.fill_': torch.fill_,
    'torch.flip': torch.flip,
    'torch.floor_divide': torch.floor_divide,
    'torch.Tensor.floor_divide_': torch.Tensor.floor_divide_,
    'torch.full': torch.full,
    'torch.full_like': torch.full_like,
    'torch.gather': torch.gather,
    'torch.ge': torch.ge,
    'torch.nn.functional.gelu': torch.nn.functional.gelu,
    'torch._C._nn.gelu_': torch._C._nn.gelu_,
    'torch.nn.functional.glu': torch.nn.functional.glu,
    'torch.group_norm': torch.group_norm,
    'torch.gt': torch.gt,
    'torch.hstack': torch.hstack,
    'torch.ops.aten.index': torch.ops.aten.index,
    'torch.index_add': torch.index_add,
    'torch.index_put': torch.index_put,
    'torch.index_put_': torch.index_put_,
    'torch.index_select': torch.index_select,
    # 'torch.instance_norm': torch.instance_norm,
    'torch.isclose': torch.isclose,
    'torch.isfinite': torch.isfinite,
    'torch.isin': torch.isin,
    'torch.isinf': torch.isinf,
    'torch.isnan': torch.isnan,
    'torch.kron': torch.kron,
    'torch.layer_norm': torch.layer_norm,
    'torch.le': torch.le,
    'torch.lerp': torch.lerp,
    'torch.Tensor.lerp_': torch.Tensor.lerp_,
    'torch.linspace': torch.linspace,
    'torch.log': torch.log,
    'torch.nn.functional.logsigmoid': torch.nn.functional.logsigmoid,
    'torch.log_softmax': torch.log_softmax,
    'torch.logical_and': torch.logical_and,
    'torch.logical_not': torch.logical_not,
    'torch.logical_or': torch.logical_or,
    'torch.logical_xor': torch.logical_xor,
    'torch.lt': torch.lt,
    'torch.masked_fill': torch.masked_fill,
    'torch.Tensor.masked_fill_': torch.Tensor.masked_fill_,
    'torch.masked_select': torch.masked_select,
    'torch.max': torch.max,
    'torch.maximum': torch.maximum,
    'torch.mean': torch.mean,
    'torch.min': torch.min,
    'torch.minimum': torch.minimum,
    'torch.mm': torch.mm,
    'torch.nn.functional.mse_loss': torch.nn.functional.mse_loss,
    'torch.mul': torch.mul,
    'torch.Tensor.mul_': torch.Tensor.mul_,
    # 'torch.multinomial': torch.multinomial,
    'torch.mv': torch.mv,
    'torch.nan_to_num': torch.nan_to_num,
    'torch.ne': torch.ne,
    'torch.neg': torch.neg,
    'torch.neg_': torch.neg_,
    'torch.nn.functional.nll_loss': torch.nn.functional.nll_loss,
    'torch.nonzero': torch.nonzero,
    'torch.normal': torch.normal,
    'torch.ones': torch.ones,
    'torch.ones_like': torch.ones_like,
    'torch.outer': torch.outer,
    'torch.nn.functional.pad': torch.nn.functional.pad,
    'torch.polar': torch.polar,
    'torch.pow': torch.pow,
    'torch.Tensor.pow_': torch.Tensor.pow_,
    'torch.prod': torch.prod,
    'torch.quantile': torch.quantile,
    'torch.rand': torch.rand,
    'torch.rand_like': torch.rand_like,
    'torch.randn': torch.randn,
    'torch.randn_like': torch.randn_like,
    'torch.randperm': torch.randperm,
    'torch.reciprocal': torch.reciprocal,
    'torch.reciprocal_': torch.reciprocal_,
    'torch.relu': torch.relu,
    'torch.relu_': torch.relu_,
    'torch.remainder': torch.remainder,
    'torch.Tensor.remainder_': torch.Tensor.remainder_,
    'torch.Tensor.repeat': torch.Tensor.repeat,
    'torch.repeat_interleave': torch.repeat_interleave,
    'torch.resolve_conj': torch.resolve_conj,
    'torch.resolve_neg': torch.resolve_neg,
    'torch.rms_norm': torch.rms_norm,
    'torch.rsqrt': torch.rsqrt,
    'torch.rsqrt_': torch.rsqrt_,
    'torch.nn.functional.scaled_dot_product_attention': torch.nn.functional.scaled_dot_product_attention,
    'torch.scatter': torch.scatter,
    'torch.Tensor.scatter_': torch.Tensor.scatter_,
    'torch.select_scatter': torch.select_scatter,
    'torch.sigmoid': torch.sigmoid,
    'torch.sigmoid_': torch.sigmoid_,
    'torch.nn.functional.silu': torch.nn.functional.silu,
    'torch._C._nn.silu_': torch._C._nn.silu_,
    'torch.sin': torch.sin,
    'torch.sin_': torch.sin_,
    'torch.slice_scatter': torch.slice_scatter,
    'torch.softmax': torch.softmax,
    'torch.sort': torch.sort,
    'torch.stack': torch.stack,
    'torch.sub': torch.sub,
    'torch.Tensor.sub_': torch.Tensor.sub_,
    'torch.sum': torch.sum,
    'torch.tanh': torch.tanh,
    'torch.tanh_': torch.tanh_,
    'torch.threshold': torch.threshold,
    'torch.tile': torch.tile,
    'torch.Tensor.to': torch.Tensor.to,
    'torch.topk': torch.topk,
    'torch.triu': torch.triu,
    'torch.Tensor.uniform_': torch.Tensor.uniform_,
    'torch.unique': torch.unique,
    'torch.nn.functional.upsample': torch.nn.functional.upsample,
    'torch.var_mean': torch.var_mean,
    'torch.vdot': torch.vdot,
    'torch.linalg.vector_norm': torch.linalg.vector_norm,
    'torch.vstack': torch.vstack,
    'torch._weight_norm': torch._weight_norm,
    'torch.where': torch.where,
    'torch.zeros': torch.zeros,
    'torch.zeros_like': torch.zeros_like,
    'torch.true_divide': torch.true_divide,
    'torch.Tensor.true_divide_': torch.Tensor.true_divide_,
    'torch.divide': torch.divide,
    'torch.Tensor.divide_': torch.Tensor.divide_, 
    'torch.index_fill': torch.index_fill, 
}

# Selected 40 operators for benchmark library
# These operators are selected based on correctness test results from gpt-5.1 evaluation
# and have corresponding performance tests
V1_OPERATORS = {
    'torch.abs': torch.abs,
    'torch.all': torch.all,
    'torch.allclose': torch.allclose,
    'torch.amax': torch.amax,
    'torch.any': torch.any,
    'torch.arange': torch.arange,
    'torch.argmax': torch.argmax,
    'torch.argmin': torch.argmin,
    'torch.bitwise_and': torch.bitwise_and,
    'torch.bitwise_not': torch.bitwise_not,
    'torch.bitwise_or': torch.bitwise_or,
    'torch.cos': torch.cos,
    'torch.count_nonzero': torch.count_nonzero,
    'torch.diag': torch.diag,
    'torch.diag_embed': torch.diag_embed,
    'torch.div': torch.div,
    'torch.embedding': torch.embedding,
    'torch.eq': torch.eq,
    'torch.fill': torch.fill,
    'torch.floor_divide': torch.floor_divide,
    'torch.full': torch.full,
    'torch.full_like': torch.full_like,
    'torch.gather': torch.gather,
    'torch.ge': torch.ge,
    'torch.gt': torch.gt,
    'torch.index_add': torch.index_add,
    'torch.isfinite': torch.isfinite,
    'torch.isinf': torch.isinf,
    'torch.isnan': torch.isnan,
    'torch.kron': torch.kron,
    'torch.mean': torch.mean,
    'torch.mul': torch.mul,
    'torch.nn.functional.scaled_dot_product_attention': torch.nn.functional.scaled_dot_product_attention,
    'torch.ones': torch.ones,
    'torch.rand': torch.rand,
    'torch.relu': torch.relu,
    'torch.resolve_conj': torch.resolve_conj,
    'torch.tanh': torch.tanh,
    'torch.vdot': torch.vdot,
    'torch.zeros_like': torch.zeros_like,
}

# Non-FlagGems operators: 10 operators from log_9 result.json
# These operators are not in FlagGems but have test functions generated
NON_FLAGGEMS_OPERATORS = {
    'torch.ops.aten.log_normal': torch.ops.aten.log_normal,
    'torch.ops.aten.bernoulli': torch.ops.aten.bernoulli,
    'torch.ops.aten.unfold_backward': torch.ops.aten.unfold_backward,
    'torch.ops.aten.logit_backward': torch.ops.aten.logit_backward,
    'torch.ops.aten.convolution': torch.ops.aten.convolution,
    'torch.ops.aten.linalg_cross': torch.ops.aten.linalg_cross,
    'torch.ops.aten.avg_pool3d': torch.ops.aten.avg_pool3d,
    'torch.ops.aten.round': torch.ops.aten.round,
    'torch.ops.aten.baddbmm': torch.ops.aten.baddbmm,
    'torch.ops.aten.addbmm': torch.ops.aten.addbmm,
}

# V2 operators: 50 operators from sampled_from_passed_ops.json
# These operators have test functions generated and extracted to test_v2_ops.py
V2_OPERATORS = {
    # Backward operators
    'torch.ops.aten.log_sigmoid_backward': torch.ops.aten.log_sigmoid_backward,
    'torch.ops.aten.mish_backward': torch.ops.aten.mish_backward,
    'torch.ops.aten.reflection_pad1d_backward': torch.ops.aten.reflection_pad1d_backward,
    'torch.ops.aten.rrelu_with_noise_backward': torch.ops.aten.rrelu_with_noise_backward,
    'torch.ops.aten.select_backward': torch.ops.aten.select_backward,
    'torch.ops.aten.smooth_l1_loss_backward': torch.ops.aten.smooth_l1_loss_backward,
    'torch.ops.aten.softplus_backward': torch.ops.aten.softplus_backward,
    'torch.ops.aten.upsample_nearest2d_backward': torch.ops.aten.upsample_nearest2d_backward,
    # Activation functions
    'torch.ops.aten.erfc': torch.ops.aten.erfc,
    'torch.ops.aten.hardsigmoid': torch.ops.aten.hardsigmoid,
    'torch.ops.aten.heaviside': torch.ops.aten.heaviside,
    'torch.ops.aten.log10': torch.ops.aten.log10,
    'torch.ops.aten.logit': torch.ops.aten.logit,
    'torch.ops.aten.mish': torch.ops.aten.mish,
    'torch.ops.aten.prelu': torch.ops.aten.prelu,
    'torch.ops.aten.rrelu_with_noise': torch.ops.aten.rrelu_with_noise,
    'torch.ops.aten.square': torch.ops.aten.square,
    # Tensor creation and manipulation
    'torch.ops.aten.affine_grid_generator': torch.ops.aten.affine_grid_generator,
    'torch.ops.aten.bernoulli': torch.ops.aten.bernoulli,
    'torch.ops.aten.empty_strided': torch.ops.aten.empty_strided,
    'torch.ops.aten.new_empty_strided': torch.ops.aten.new_empty_strided,
    'torch.ops.aten.new_ones': torch.ops.aten.new_ones,
    'torch.ops.aten.poisson': torch.ops.aten.poisson,
    'torch.ops.aten.scalar_tensor': torch.ops.aten.scalar_tensor,
    # Math operations
    'torch.ops.aten.acosh': torch.ops.aten.acosh,
    'torch.ops.aten.asin': torch.ops.aten.asin,
    'torch.ops.aten.cosh': torch.ops.aten.cosh,
    'torch.ops.aten.floor': torch.ops.aten.floor,
    'torch.ops.aten.i0': torch.ops.aten.i0,
    'torch.ops.aten.polygamma': torch.ops.aten.polygamma,
    'torch.ops.aten.rsub': torch.ops.aten.rsub,
    'torch.ops.aten.sgn': torch.ops.aten.sgn,
    'torch.ops.aten.special_entr': torch.ops.aten.special_entr,
    # Reduction and comparison operations
    'torch.ops.aten.amin': torch.ops.aten.amin,
    'torch.ops.aten.binary_cross_entropy_with_logits': torch.ops.aten.binary_cross_entropy_with_logits,
    'torch.ops.aten.fmax': torch.ops.aten.fmax,
    'torch.ops.aten.huber_loss': torch.ops.aten.huber_loss,
    'torch.ops.aten.logaddexp2': torch.ops.aten.logaddexp2,
    'torch.ops.aten.margin_ranking_loss': torch.ops.aten.margin_ranking_loss,
    'torch.ops.aten.pairwise_distance': torch.ops.aten.pairwise_distance,
    'torch.ops.aten.renorm': torch.ops.aten.renorm,
    'torch.ops.aten.soft_margin_loss': torch.ops.aten.soft_margin_loss,
    # Tensor shape operations
    'torch.ops.aten.as_strided': torch.ops.aten.as_strided,
    'torch.ops.aten.im2col': torch.ops.aten.im2col,
    'torch.ops.aten.reshape': torch.ops.aten.reshape,
    'torch.ops.aten.rot90': torch.ops.aten.rot90,
    'torch.ops.aten.t': torch.ops.aten.t,
    'torch.ops.aten.unsafe_split': torch.ops.aten.unsafe_split,
    'torch.ops.aten.unsafe_split_with_sizes': torch.ops.aten.unsafe_split_with_sizes,
    'torch.ops.aten.unsqueeze': torch.ops.aten.unsqueeze,
}


# Qwen next operators
# aten::_flash_attention_backward
# aten::_flash_attention_forward
# aten::_index_put_impl_
# aten::_local_scalar_dense
# aten::_scaled_dot_product_flash_attention
# aten::_scaled_dot_product_flash_attention_backward
# aten::_softmax
# aten::_to_copy
# aten::add
# aten::add_
# aten::arange
# aten::argmax
# aten::bitwise_not
# aten::bmm
# aten::cat
# aten::clone
# aten::contiguous
# aten::copy_
# aten::cos
# aten::cumsum
# aten::diff
# aten::div
# aten::div_
# aten::embedding
# aten::embedding_backward
# aten::embedding_dense_backward
# aten::eq
# aten::expand
# aten::expand_as
# aten::exponential_
# aten::fill_
# aten::floor_divide
# aten::full
# aten::gather
# aten::gt
# aten::index
# aten::index_put_
# aten::index_select
# aten::item
# aten::le
# aten::linear
# aten::masked_fill_
# aten::matmul
# aten::mean
# aten::mm
# aten::mul
# aten::narrow
# aten::neg
# aten::ones_like
# aten::pow
# aten::resolve_conj
# aten::resolve_neg
# aten::rsqrt
# aten::rsub
# aten::scaled_dot_product_attention
# aten::scatter
# aten::select
# aten::silu
# aten::silu_backward
# aten::sin
# aten::softmax
# aten::sort
# aten::stack
# aten::sub
# aten::sum
# aten::to
# aten::zero_
# aten::zeros
# aten::zeros_like

# Qwen next operators
QWEN_NEXT_OPERATORS = {
    # 'torch.ops.aten._flash_attention_backward': torch.ops.aten._flash_attention_backward,
    # 'torch.ops.aten._flash_attention_forward': torch.ops.aten._flash_attention_forward,
    # 'torch.ops.aten._index_put_impl_': torch.ops.aten._index_put_impl_,
    # 'torch.ops.aten._local_scalar_dense': torch.ops.aten._local_scalar_dense,
    # 'torch.ops.aten._scaled_dot_product_flash_attention': torch.ops.aten._scaled_dot_product_flash_attention,
    # 'torch.ops.aten._scaled_dot_product_flash_attention_backward': torch.ops.aten._scaled_dot_product_flash_attention_backward,
    # 'torch.ops.aten._softmax': torch.ops.aten._softmax,
    # 'torch.ops.aten._to_copy': torch.ops.aten._to_copy,
    # 'torch.ops.aten.add': torch.ops.aten.add,
    # 'torch.ops.aten.add_': torch.ops.aten.add_,
    # 'torch.ops.aten.arange': torch.ops.aten.arange,
    # 'torch.ops.aten.argmax': torch.ops.aten.argmax,
    # 'torch.ops.aten.bitwise_not': torch.ops.aten.bitwise_not,
    # 'torch.ops.aten.bmm': torch.ops.aten.bmm,
    # 'torch.ops.aten.cat': torch.ops.aten.cat,
    # 'torch.ops.aten.clone': torch.ops.aten.clone,
    # 'torch.ops.aten.contiguous': torch.ops.aten.contiguous,
    # 'torch.ops.aten.copy_': torch.ops.aten.copy_,
    # 'torch.ops.aten.cos': torch.ops.aten.cos,
    # 'torch.ops.aten.cumsum': torch.ops.aten.cumsum,
    # 'torch.ops.aten.diff': torch.ops.aten.diff,
    # 'torch.ops.aten.div': torch.ops.aten.div,
    # 'torch.ops.aten.div_': torch.ops.aten.div_,
    # 'torch.ops.aten.embedding': torch.ops.aten.embedding,
    # 'torch.ops.aten.embedding_backward': torch.ops.aten.embedding_backward,
    # 'torch.ops.aten.embedding_dense_backward': torch.ops.aten.embedding_dense_backward,
    # 'torch.ops.aten.eq': torch.ops.aten.eq,
    # 'torch.ops.aten.expand': torch.ops.aten.expand,
    # 'torch.ops.aten.expand_as': torch.ops.aten.expand_as,
    # 'torch.ops.aten.exponential_': torch.ops.aten.exponential_,
    # 'torch.ops.aten.fill_': torch.ops.aten.fill_,
    # 'torch.ops.aten.floor_divide': torch.ops.aten.floor_divide,
    # 'torch.ops.aten.full': torch.ops.aten.full,
    # 'torch.ops.aten.gather': torch.ops.aten.gather,
    'torch.ops.aten.gt': torch.ops.aten.gt,
    'torch.ops.aten.index': torch.ops.aten.index,
    'torch.ops.aten.index_put_': torch.ops.aten.index_put_,
    'torch.ops.aten.index_select': torch.ops.aten.index_select,
    'torch.ops.aten.item': torch.ops.aten.item,
    'torch.ops.aten.le': torch.ops.aten.le,
    'torch.ops.aten.linear': torch.ops.aten.linear,
    'torch.ops.aten.masked_fill_': torch.ops.aten.masked_fill_,
    'torch.ops.aten.matmul': torch.ops.aten.matmul,
    'torch.ops.aten.mean': torch.ops.aten.mean,
    'torch.ops.aten.mm': torch.ops.aten.mm,
    'torch.ops.aten.mul': torch.ops.aten.mul,
    'torch.ops.aten.narrow': torch.ops.aten.narrow,
    'torch.ops.aten.neg': torch.ops.aten.neg,
    'torch.ops.aten.ones_like': torch.ops.aten.ones_like,
    'torch.ops.aten.pow': torch.ops.aten.pow,
    'torch.ops.aten.resolve_conj': torch.ops.aten.resolve_conj,
    'torch.ops.aten.resolve_neg': torch.ops.aten.resolve_neg,
    'torch.ops.aten.rsqrt': torch.ops.aten.rsqrt,
    # 'torch.ops.aten.rsub': torch.ops.aten.rsub,
    'torch.ops.aten.scaled_dot_product_attention': torch.ops.aten.scaled_dot_product_attention,
    # 'torch.ops.aten.scatter': torch.ops.aten.scatter,
    # 'torch.ops.aten.select': torch.ops.aten.select,
    # 'torch.ops.aten.silu': torch.ops.aten.silu,
    # 'torch.ops.aten.silu_backward': torch.ops.aten.silu_backward,
    # 'torch.ops.aten.sin': torch.ops.aten.sin,
    # 'torch.ops.aten.softmax': torch.ops.aten.softmax,
    # 'torch.ops.aten.sort': torch.ops.aten.sort,
    # 'torch.ops.aten.stack': torch.ops.aten.stack,
    # 'torch.ops.aten.sub': torch.ops.aten.sub,
    # 'torch.ops.aten.sum': torch.ops.aten.sum,
    # 'torch.ops.aten.to': torch.ops.aten.to,
    # 'torch.ops.aten.zero_': torch.ops.aten.zero_,
    # 'torch.ops.aten.zeros': torch.ops.aten.zeros,
    # 'torch.ops.aten.zeros_like': torch.ops.aten.zeros_like,
}

# PYTORCH_OPERATORS = BENCHMARK_OPERATORS
# PYTORCH_OPERATORS = V2_OPERATORS

if os.environ.get("FLAGBENCH_USE_DYNAMIC_IMPL_INFO", "0") == "1":
    dynamic_impl_info = DynamicImplInfo()
    IMPL_INFO = dynamic_impl_info
    
if __name__ == "__main__":
    op_name_list = list(PYTORCH_OPERATORS.keys())
    dynamic_impl_info = DynamicImplInfo()
    namespace = "aten"
    for op_name in op_name_list:
        impl_info = dynamic_impl_info.get(namespace, op_name.split(".")[-1])
        print(f"{op_name}: {impl_info}")