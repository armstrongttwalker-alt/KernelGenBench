import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zscal(n, alpha, x, incx):
    """CuPy cuBLAS baseline for zscal: x = alpha * x (in-place scaling of complex128 vector)"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Call cuBLAS via CuPy (in-place operation)
    cublas.scal(alpha, x_cp)
    
    # Convert back and return
    ref_out = from_dlpack(x_cp.toDlpack())
    return ref_out