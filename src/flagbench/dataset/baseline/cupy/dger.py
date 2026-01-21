from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def dger(m, n, alpha, x, incx, y, incy, A, lda):
    """CuPy cuBLAS baseline for dger: A = alpha * x @ y.T + A"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    A_cp = cp.from_dlpack(to_dlpack(A))
    
    # Call cuBLAS via CuPy - ger is in-place, returns None
    cublas.ger(alpha, x_cp, y_cp, A_cp)
    
    # Convert back and return (A_cp is modified in-place)
    ref_out = from_dlpack(A_cp.toDlpack())
    return ref_out