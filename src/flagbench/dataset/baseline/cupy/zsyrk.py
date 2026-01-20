from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "zsyrk", has_backward=Autograd.disable, namespace="baseline")
def zsyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """CuPy cuBLAS baseline for zsyrk: C = alpha*A@A.T + beta*C (complex128)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Map cuBLAS parameters to CuPY's syrk
    lower = (uplo == 1)  # 1 for CUBLAS_FILL_MODE_LOWER, 0 for UPPER
    trans_char = 'T' if trans else 'N'
    
    # Call cuBLAS via CuPy
    C_cp = cublas.syrk(trans_char, A_cp, out=C_cp, alpha=alpha, beta=beta, lower=lower)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out