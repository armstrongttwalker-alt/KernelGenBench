from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zaxpy(n, alpha, x, incx, y, incy):
    """CuPy cuBLAS baseline for zaxpy: y = alpha * x + y (complex128)"""
    # Convert to CuPy arrays
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Call cuBLAS axpy (modifies y in-place)
    cublas.axpy(alpha, x_cp, y_cp)
    
    # Return the modified y
    ref_out = from_dlpack(y_cp.toDlpack())
    return ref_out