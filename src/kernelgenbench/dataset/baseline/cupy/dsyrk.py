import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def dsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """CuPy cuBLAS baseline for dsyrk: C = alpha*A@A.T + beta*C (symmetric rank-k update)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Map cuBLAS parameters to CuPY's syrk
    trans_char = 'T' if trans else 'N'
    lower = (uplo == 1)  # Assuming 1 is CUBLAS_FILL_MODE_LOWER
    
    # Call cuBLAS via CuPy
    C_cp = cublas.syrk(trans_char, A_cp, out=C_cp, alpha=alpha, beta=beta, lower=lower)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out