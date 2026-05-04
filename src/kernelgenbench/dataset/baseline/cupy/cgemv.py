import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """CuPy cuBLAS baseline for cgemv: y = alpha * op(A) * x + beta * y"""
    # Convert to CuPy arrays
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    y_cp = cp.from_dlpack(to_dlpack(y))
    
    # Convert alpha and beta to complex64 scalars (handle both Tensor and scalar)
    alpha_val = alpha if isinstance(alpha, (int, float, complex)) else alpha.item()
    beta_val = beta if isinstance(beta, (int, float, complex)) else beta.item()
    alpha_cp = cp.complex64(alpha_val)
    beta_cp = cp.complex64(beta_val)
    
    # Call cuBLAS gemv (modifies y_cp in-place)
    cublas.gemv(trans, alpha_cp, A_cp, x_cp, beta_cp, y_cp)
    
    # Convert back and return
    ref_out = from_dlpack(y_cp.toDlpack())
    return ref_out