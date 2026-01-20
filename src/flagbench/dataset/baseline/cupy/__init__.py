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

# Import all baseline modules to trigger registration
from . import (
    # BLAS Level 1 - asum
    sasum, dasum,
    
    # BLAS Level 1 - axpy
    saxpy, daxpy, caxpy, zaxpy,
    
    # BLAS Level 1 - dot
    sdot, ddot, cdotu, cdotc, zdotu, zdotc,
    
    # BLAS Level 1 - nrm2
    snrm2, dnrm2,
    
    # BLAS Level 1 - scal
    sscal, dscal, cscal, zscal,
    
    # BLAS Level 2 - gemv
    sgemv, dgemv, cgemv, zgemv,
    
    # BLAS Level 2 - ger
    sger, dger, cgeru, cgerc, zgeru, zgerc,
    
    # BLAS Level 2 - sbmv
    ssbmv, dsbmv,
    
    # BLAS Level 3 - gemm
    sgemm, dgemm, cgemm, zgemm, hgemm,
    
    # BLAS Level 3 - syrk
    ssyrk, dsyrk, csyrk, zsyrk,
    
    # Extensions - geam
    sgeam, dgeam, cgeam, zgeam,
    
    # Extensions - dgmm
    sdgmm, ddgmm, cdgmm, zdgmm,
)

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
