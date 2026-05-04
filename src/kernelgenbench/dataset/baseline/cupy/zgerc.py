import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zgerc(m, n, alpha, x, incx, y, incy, A, lda):
    """CuPy cuBLAS baseline for zgerc: A = alpha * x @ y^H + A (complex rank-1 update)"""
    # Convert to CuPy arrays
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    A_cp = cp.from_dlpack(to_dlpack(A))
    
    # Call cuBLAS gerc (in-place)
    cublas.gerc(alpha, x_cp, y_cp, A_cp)
    
    # Convert back and return
    ref_out = from_dlpack(A_cp.toDlpack())
    return ref_out