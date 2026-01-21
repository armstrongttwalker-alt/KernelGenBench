from flagbench.dataset import Autograd
import torch
import cupy as cp
from cupy import cublas
from torch.utils.dlpack import to_dlpack, from_dlpack

def sscal(n, alpha, x, incx):
    """CuPy cuBLAS baseline for sscal: x = alpha * x (in-place)"""
    # Convert to CuPy
    x_cp = cp.from_dlpack(to_dlpack(x))
    
    # Call cuBLAS via CuPy
    cublas.scal(alpha, x_cp)
    
    # Convert back and return
    ref_out = from_dlpack(x_cp.toDlpack())
    return ref_out