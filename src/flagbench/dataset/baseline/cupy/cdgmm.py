from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

@register("CUDA", "cdgmm", has_backward=Autograd.disable, namespace="baseline")
def cdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """CuPy cuBLAS baseline for cdgmm: C = A diag(x) (left) or diag(x) A (right)"""
    # Convert to CuPy
    A_cp = cp.from_dlpack(to_dlpack(A))
    x_cp = cp.from_dlpack(to_dlpack(x))
    C_cp = cp.from_dlpack(to_dlpack(C))
    
    # Map cublasSideMode_t to CuPy side parameter
    side = 'L' if mode == 0 else 'R'
    
    # Call cuBLAS dgmm via CuPy
    C_cp = cublas.dgmm(side, A_cp, x_cp, out=C_cp, incx=incx)
    
    # Convert back and return
    ref_out = from_dlpack(C_cp.toDlpack())
    return ref_out