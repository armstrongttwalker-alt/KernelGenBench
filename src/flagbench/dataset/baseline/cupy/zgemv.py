from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """CuPy cuBLAS baseline for zgemv: y = alpha * op(A) * x + beta * y"""
    # Convert to CuPy arrays
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Call cuBLAS gemv (modifies y_cp in-place)
    cublas.gemv(trans, alpha, A_cp, x_cp, beta, y_cp)
    
    # Convert back and return
    ref_out = from_dlpack(y_cp.toDlpack())
    return ref_out