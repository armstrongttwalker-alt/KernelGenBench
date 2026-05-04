import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def hgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """CuPy cuBLAS baseline for hgemm: C = alpha * op(A) @ op(B) + beta * C"""
    # Convert to CuPy arrays
    A_cp = cp.asarray(A, dtype=cp.float16)
    B_cp = cp.asarray(B, dtype=cp.float16)
    C_cp = cp.asarray(C, dtype=cp.float16)
    
    # Handle transpose flags
    # 'N' or 0 = no transpose, 'T' or 1 = transpose, 'C' or 2 = conjugate transpose
    if transa in ['N', 'n', 0]:
        A_op = A_cp
    elif transa in ['T', 't', 1]:
        A_op = A_cp.T
    else:  # 'C' or conjugate
        A_op = A_cp.T.conj()
    
    if transb in ['N', 'n', 0]:
        B_op = B_cp
    elif transb in ['T', 't', 1]:
        B_op = B_cp.T
    else:  # 'C' or conjugate
        B_op = B_cp.T.conj()
    
    # Compute: C = alpha * op(A) @ op(B) + beta * C
    result = alpha * A_op @ B_op + beta * C_cp
    
    # Convert back and return
    ref_out = torch.from_numpy(result.get()).to(device='cuda', dtype=torch.float16)
    return ref_out