from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "zgeru", has_backward=Autograd.disable, namespace="baseline")
def zgeru(m, n, alpha, x, incx, y, incy, A, lda):
    """CuPy cuBLAS baseline for zgeru: A = alpha * x @ y.T + A (complex128)"""
    # Convert to CuPy arrays
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    A_cp = cp.from_dlpack(to_dlpack(A))
    
    # Call cuBLAS geru (in-place)
    cublas.geru(alpha, x_cp, y_cp, A_cp)
    
    # Convert back and return
    ref_out = from_dlpack(A_cp.toDlpack())
    return ref_out