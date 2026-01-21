from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def sdot(n, x, incx, y, incy, result):
    """CuPy cuBLAS baseline for sdot: result = x.T * y"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Pre-allocate output buffer
    result_cp = cp.empty(1, dtype=x_cp.dtype)
    
    # Call cuBLAS dot
    cublas.dot(x_cp, y_cp, result_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out