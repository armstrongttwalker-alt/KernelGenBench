"""
cuBLAS Operations Test Functions

10 cuBLAS operators (excluding CgemmStridedBatched):
- 4 GEMM: S/D/H/Z gemmStridedBatched
- 4 GEMV: S/D/C/Z gemvStridedBatched
- SAXPY_v2, SSCAL_v2
"""

from .test_cublasSgemmStridedBatched import test_accuracy_cublasSgemmStridedBatched
from .test_cublasDgemmStridedBatched import test_accuracy_cublasDgemmStridedBatched
from .test_cublasHgemmStridedBatched import test_accuracy_cublasHgemmStridedBatched
from .test_cublasZgemmStridedBatched import test_accuracy_cublasZgemmStridedBatched
from .test_cublasSgemvStridedBatched import test_accuracy_cublasSgemvStridedBatched
from .test_cublasDgemvStridedBatched import test_accuracy_cublasDgemvStridedBatched
from .test_cublasCgemvStridedBatched import test_accuracy_cublasCgemvStridedBatched
from .test_cublasZgemvStridedBatched import test_accuracy_cublasZgemvStridedBatched
from .test_cublasSaxpy_v2 import test_accuracy_cublasSaxpy_v2
from .test_cublasSscal_v2 import test_accuracy_cublasSscal_v2

__all__ = [
    'test_accuracy_cublasSgemmStridedBatched',
    'test_accuracy_cublasDgemmStridedBatched',
    'test_accuracy_cublasHgemmStridedBatched',
    'test_accuracy_cublasZgemmStridedBatched',
    'test_accuracy_cublasSgemvStridedBatched',
    'test_accuracy_cublasDgemvStridedBatched',
    'test_accuracy_cublasCgemvStridedBatched',
    'test_accuracy_cublasZgemvStridedBatched',
    'test_accuracy_cublasSaxpy_v2',
    'test_accuracy_cublasSscal_v2',
]
