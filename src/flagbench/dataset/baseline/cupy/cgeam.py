from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def cgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """CuPy cuBLAS baseline for cgeam: C = alpha * op(A) + beta * op(B)"""
    # Convert to CuPy arrays
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    # Convert scalars to complex64
    alpha_cp = cp.complex64(alpha)
    beta_cp = cp.complex64(beta)
    
    # Call cuBLAS geam
    C_cp = cublas.geam(transa, transb, alpha_cp, A_cp, beta_cp, B_cp)
    
    # Convert back to PyTorch
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out