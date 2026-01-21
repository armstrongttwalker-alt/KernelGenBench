from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """CuPy cuBLAS baseline for zgemm: C = alpha * op(A) @ op(B) + beta * C"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    # Convert scalar parameters
    alpha_cp = complex(alpha.item()) if isinstance(alpha, torch.Tensor) else complex(alpha)
    beta_cp = complex(beta.item()) if isinstance(beta, torch.Tensor) else complex(beta)
    
    # Call cuBLAS via CuPy
    if beta_cp == 0.0:
        C_cp = cublas.gemm(transa, transb, A_cp, B_cp, alpha=alpha_cp, beta=beta_cp)
    else:
        C_cp = cp.from_dlpack(to_dlpack(C))
        C_cp = cublas.gemm(transa, transb, A_cp, B_cp, out=C_cp, alpha=alpha_cp, beta=beta_cp)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out