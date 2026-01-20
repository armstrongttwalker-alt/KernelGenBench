from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "cscal", has_backward=Autograd.disable, namespace="baseline")
def cscal(n, alpha, x, incx):
    """CuPy cuBLAS baseline for cscal: x = alpha * x (in-place complex64 scaling)"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    alpha_cp = cp.complex64(alpha)
    
    # Call cuBLAS via CuPy
    cublas.scal(alpha_cp, x_cp)
    
    # Convert back and return
    ref_out = from_dlpack(x_cp.toDlpack())
    return ref_out