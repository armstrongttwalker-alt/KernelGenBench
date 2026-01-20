from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "dgeam", has_backward=Autograd.disable, namespace="baseline")
def dgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """CuPy cuBLAS baseline for dgeam: C = alpha * op(A) + beta * op(B)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Call cuBLAS via CuPy
    C_cp = cublas.geam(transa, transb, alpha, A_cp, beta, B_cp, out=C_cp)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out