"""
cuBLAS Baseline Functions using CuPy

This module contains 47 cuBLAS baseline implementations that use CuPy's 
cuBLAS wrappers. All functions are automatically registered to the 
flagbench.baseline.* namespace.

Generated functions:
- BLAS Level 1: asum, axpy, dot, dotu, dotc, nrm2, scal (multiple dtypes)
- BLAS Level 2: gemv, ger, geru, gerc, sbmv (multiple dtypes)
- BLAS Level 3: gemm, syrk (multiple dtypes)
- Extensions: geam, dgmm (multiple dtypes)
"""

# Import all baseline functions
from .sasum import sasum
from .dasum import dasum
from .saxpy import saxpy
from .daxpy import daxpy
from .caxpy import caxpy
from .zaxpy import zaxpy
from .sdot import sdot
from .ddot import ddot
from .cdotu import cdotu
from .cdotc import cdotc
from .zdotu import zdotu
from .zdotc import zdotc
from .snrm2 import snrm2
from .dnrm2 import dnrm2
from .sscal import sscal
from .dscal import dscal
from .cscal import cscal
from .zscal import zscal
from .sgemv import sgemv
from .dgemv import dgemv
from .cgemv import cgemv
from .zgemv import zgemv
from .sger import sger
from .dger import dger
from .cgeru import cgeru
from .cgerc import cgerc
from .zgeru import zgeru
from .zgerc import zgerc
from .ssbmv import ssbmv
from .dsbmv import dsbmv
from .sgemm import sgemm
from .dgemm import dgemm
from .cgemm import cgemm
from .zgemm import zgemm
from .hgemm import hgemm
from .ssyrk import ssyrk
from .dsyrk import dsyrk
from .csyrk import csyrk
from .zsyrk import zsyrk
from .sgeam import sgeam
from .dgeam import dgeam
from .cgeam import cgeam
from .zgeam import zgeam
from .sdgmm import sdgmm
from .ddgmm import ddgmm
from .cdgmm import cdgmm
from .zdgmm import zdgmm

__all__ = [
    # BLAS Level 1 - asum
    'sasum', 'dasum',
    
    # BLAS Level 1 - axpy
    'saxpy', 'daxpy', 'caxpy', 'zaxpy',
    
    # BLAS Level 1 - dot
    'sdot', 'ddot', 'cdotu', 'cdotc', 'zdotu', 'zdotc',
    
    # BLAS Level 1 - nrm2
    'snrm2', 'dnrm2',
    
    # BLAS Level 1 - scal
    'sscal', 'dscal', 'cscal', 'zscal',
    
    # BLAS Level 2 - gemv
    'sgemv', 'dgemv', 'cgemv', 'zgemv',
    
    # BLAS Level 2 - ger
    'sger', 'dger', 'cgeru', 'cgerc', 'zgeru', 'zgerc',
    
    # BLAS Level 2 - sbmv
    'ssbmv', 'dsbmv',
    
    # BLAS Level 3 - gemm
    'sgemm', 'dgemm', 'cgemm', 'zgemm', 'hgemm',
    
    # BLAS Level 3 - syrk
    'ssyrk', 'dsyrk', 'csyrk', 'zsyrk',
    
    # Extensions - geam
    'sgeam', 'dgeam', 'cgeam', 'zgeam',
    
    # Extensions - dgmm
    'sdgmm', 'ddgmm', 'cdgmm', 'zdgmm',
]
