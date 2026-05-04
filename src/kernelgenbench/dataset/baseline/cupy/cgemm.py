import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def cgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """CuPy cuBLAS baseline for cgemm: C = alpha * op(A) @ op(B) + beta * C"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Call cuBLAS via CuPy
    C_cp = cublas.gemm(transa, transb, A_cp, B_cp, out=C_cp, alpha=alpha, beta=beta)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out