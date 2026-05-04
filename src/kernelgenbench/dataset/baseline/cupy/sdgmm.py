import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def sdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """CuPy cuBLAS baseline for sdgmm: C = A diag(x) (left/right multiplication)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Determine side mode ('L' for left, 'R' for right)
    side = 'L' if mode == 0 else 'R'
    
    # Call cuBLAS dgmm
    result_cp = cublas.dgmm(side, A_cp, x_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out