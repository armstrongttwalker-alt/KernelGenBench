import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def saxpy(n, alpha, x, incx, y, incy):
    """CuPy cuBLAS baseline for saxpy: y = alpha * x + y"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Call cuBLAS axpy (modifies y in-place)
    cublas.axpy(alpha, x_cp, y_cp)
    
    # Convert back and return
    ref_out = from_dlpack(y_cp.toDlpack())
    return ref_out