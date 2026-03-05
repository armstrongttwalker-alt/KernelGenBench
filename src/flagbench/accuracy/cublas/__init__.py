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

from .test_cublasCcopy_v2 import test_accuracy_cublasCcopy_v2
from .test_cublasCdotu_v2 import test_accuracy_cublasCdotu_v2
from .test_cublasCgemmStridedBatched import test_accuracy_cublasCgemmStridedBatched
from .test_cublasCgemmStridedBatched_64 import test_accuracy_cublasCgemmStridedBatched_64
from .test_cublasCgemvBatched_64 import test_accuracy_cublasCgemvBatched_64
from .test_cublasCgemv_v2 import test_accuracy_cublasCgemv_v2
from .test_cublasCgeru_v2 import test_accuracy_cublasCgeru_v2
from .test_cublasCsymm_v2 import test_accuracy_cublasCsymm_v2
from .test_cublasCsymv_v2 import test_accuracy_cublasCsymv_v2
from .test_cublasDasum_v2 import test_accuracy_cublasDasum_v2
from .test_cublasDaxpy_v2 import test_accuracy_cublasDaxpy_v2
from .test_cublasDcopy_v2 import test_accuracy_cublasDcopy_v2
from .test_cublasDdgmm import test_accuracy_cublasDdgmm
from .test_cublasDgemmBatched import test_accuracy_cublasDgemmBatched
from .test_cublasDgemmStridedBatched_64 import test_accuracy_cublasDgemmStridedBatched_64
from .test_cublasDgemm_v2 import test_accuracy_cublasDgemm_v2
from .test_cublasDgemv_v2 import test_accuracy_cublasDgemv_v2
from .test_cublasDsbmv_v2 import test_accuracy_cublasDsbmv_v2
from .test_cublasDscal_v2 import test_accuracy_cublasDscal_v2
from .test_cublasDsyr2_v2 import test_accuracy_cublasDsyr2_v2
from .test_cublasDtrsmBatched import test_accuracy_cublasDtrsmBatched
from .test_cublasHgemm import test_accuracy_cublasHgemm
from .test_cublasHgemmBatched import test_accuracy_cublasHgemmBatched
from .test_cublasSdgmm import test_accuracy_cublasSdgmm
from .test_cublasSgeam import test_accuracy_cublasSgeam
from .test_cublasSgemmBatched_64 import test_accuracy_cublasSgemmBatched_64
from .test_cublasSsyrk_v2 import test_accuracy_cublasSsyrk_v2
from .test_cublasStbmv_v2 import test_accuracy_cublasStbmv_v2
from .test_cublasStrmm_v2 import test_accuracy_cublasStrmm_v2
from .test_cublasStrsmBatched import test_accuracy_cublasStrsmBatched
from .test_cublasStrsm_v2 import test_accuracy_cublasStrsm_v2
from .test_cublasStrsv_v2 import test_accuracy_cublasStrsv_v2
from .test_cublasZdotc_v2 import test_accuracy_cublasZdotc_v2
from .test_cublasZgemmBatched import test_accuracy_cublasZgemmBatched
from .test_cublasZgemm_v2 import test_accuracy_cublasZgemm_v2
from .test_cublasZgemvBatched import test_accuracy_cublasZgemvBatched
from .test_cublasZgemv_v2 import test_accuracy_cublasZgemv_v2
from .test_cublasZgerc_v2 import test_accuracy_cublasZgerc_v2
from .test_cublasZswap_v2 import test_accuracy_cublasZswap_v2
from .test_cublasZtrsmBatched import test_accuracy_cublasZtrsmBatched

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
    'test_accuracy_cublasCcopy_v2',
    'test_accuracy_cublasCdotu_v2',
    'test_accuracy_cublasCgemmStridedBatched',
    'test_accuracy_cublasCgemmStridedBatched_64',
    'test_accuracy_cublasCgemvBatched_64',
    'test_accuracy_cublasCgemv_v2',
    'test_accuracy_cublasCgeru_v2',
    'test_accuracy_cublasCsymm_v2',
    'test_accuracy_cublasCsymv_v2',
    'test_accuracy_cublasDasum_v2',
    'test_accuracy_cublasDaxpy_v2',
    'test_accuracy_cublasDcopy_v2',
    'test_accuracy_cublasDdgmm',
    'test_accuracy_cublasDgemmBatched',
    'test_accuracy_cublasDgemmStridedBatched_64',
    'test_accuracy_cublasDgemm_v2',
    'test_accuracy_cublasDgemv_v2',
    'test_accuracy_cublasDsbmv_v2',
    'test_accuracy_cublasDscal_v2',
    'test_accuracy_cublasDsyr2_v2',
    'test_accuracy_cublasDtrsmBatched',
    'test_accuracy_cublasHgemm',
    'test_accuracy_cublasHgemmBatched',
    'test_accuracy_cublasSdgmm',
    'test_accuracy_cublasSgeam',
    'test_accuracy_cublasSgemmBatched_64',
    'test_accuracy_cublasSsyrk_v2',
    'test_accuracy_cublasStbmv_v2',
    'test_accuracy_cublasStrmm_v2',
    'test_accuracy_cublasStrsmBatched',
    'test_accuracy_cublasStrsm_v2',
    'test_accuracy_cublasStrsv_v2',
    'test_accuracy_cublasZdotc_v2',
    'test_accuracy_cublasZgemmBatched',
    'test_accuracy_cublasZgemm_v2',
    'test_accuracy_cublasZgemvBatched',
    'test_accuracy_cublasZgemv_v2',
    'test_accuracy_cublasZgerc_v2',
    'test_accuracy_cublasZswap_v2',
    'test_accuracy_cublasZtrsmBatched',
]
