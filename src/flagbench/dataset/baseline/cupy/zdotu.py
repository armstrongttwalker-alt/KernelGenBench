from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zdotu(n, x, incx, y, incy, result):
    """CuPy cuBLAS baseline for zdotu: complex dot product without conjugation"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Pre-allocate output buffer
    result_cp = cp.empty(1, dtype=cp.complex128)
    
    # Call cuBLAS dotu
    cublas.dotu(x_cp, y_cp, out=result_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out