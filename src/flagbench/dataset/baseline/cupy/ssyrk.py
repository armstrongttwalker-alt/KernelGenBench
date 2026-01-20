from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "ssyrk", has_backward=Autograd.disable, namespace="baseline")
def ssyrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """CuPy cuBLAS baseline for ssyrk: C = alpha*A@A.T + beta*C"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Map cuBLAS parameters to CuPy's syrk
    trans_char = 'T' if trans else 'N'
    lower = True if uplo == 1 else False
    
    # Call cuBLAS via CuPy
    C_cp = cublas.syrk(trans_char, A_cp, out=C_cp, alpha=alpha, beta=beta, lower=lower)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out