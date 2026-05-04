import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def dsbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """CuPy cuBLAS baseline for dsbmv: y = alpha*A*x + beta*y where A is symmetric band matrix"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Map uplo to CuPy format - use integer
    if isinstance(uplo, str):
        uplo_val = 0 if uplo.upper() == 'U' else 1
    else:
        uplo_val = uplo
    
    # Call cuBLAS via CuPy
    cublas.sbmv(uplo_val, alpha, A_cp, x_cp, beta, y_cp)
    
    # Convert back and return
    ref_out = from_dlpack(y_cp.toDlpack())
    return ref_out