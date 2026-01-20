from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "snrm2", has_backward=Autograd.disable, namespace="baseline")
def snrm2(n, x, incx, result):
    """CuPy cuBLAS baseline for snrm2: computes Euclidean norm of vector x"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Pre-allocate output buffer and call cuBLAS
    result_cp = cp.empty(1, dtype=x_cp.dtype)
    cublas.nrm2(x_cp, out=result_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out