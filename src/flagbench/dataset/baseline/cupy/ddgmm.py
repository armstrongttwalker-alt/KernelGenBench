from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "ddgmm", has_backward=Autograd.disable, namespace="baseline")
def ddgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """CuPy cuBLAS baseline for ddgmm: C = A diag(x) (left) or diag(x) A (right)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Determine side mode ('L' or 'R')
    side = 'L' if mode == 0 else 'R'
    
    # Call cuBLAS via CuPy
    result_cp = cublas.dgmm(side, A_cp, x_cp)
    
    # Convert back and return
    ref_out = from_dlpack(result_cp.toDlpack())
    return ref_out