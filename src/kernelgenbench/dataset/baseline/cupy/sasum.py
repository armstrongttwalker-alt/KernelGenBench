import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def sasum(n, x, incx, result):
    """CuPy cuBLAS baseline for sasum: sum of absolute values of vector"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Pre-allocate output buffer and call cuBLAS
    result_cp = cp.empty(1, dtype=x_cp.dtype)
    cublas.asum(x_cp, out=result_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out