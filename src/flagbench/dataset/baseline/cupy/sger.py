from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "sger", has_backward=Autograd.disable, namespace="baseline")
def sger(m, n, alpha, x, incx, y, incy, A, lda):
    """CuPy cuBLAS baseline for sger: A = alpha * x @ y.T + A"""
    # Convert to CuPy arrays
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    A_cp = cp.from_dlpack(to_dlpack(A))
    
    # Call cuBLAS ger operation - ger is in-place, returns None
    cublas.ger(alpha, x_cp, y_cp, A_cp)
    
    # Convert back to PyTorch (A_cp is modified in-place)
    ref_out = from_dlpack(A_cp.toDlpack())
    return ref_out