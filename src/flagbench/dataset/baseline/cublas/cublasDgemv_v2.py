import torch
import ctypes
from sandbox.config import DEVICE as device
from flagbench import register

# Load cuBLAS library
cublas = ctypes.CDLL("libcublas.so.12")

# cuBLAS types
cublasHandle_t = ctypes.c_void_p
cublasOperation_t = ctypes.c_int

# cublasOperation_t values
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

# Define cublasDgemv_v2 function signature
cublas.cublasDgemv_v2.argtypes = [
    cublasHandle_t,      # handle
    cublasOperation_t,   # trans
    ctypes.c_int,        # m
    ctypes.c_int,        # n
    ctypes.POINTER(ctypes.c_double),  # alpha
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.c_int,        # lda
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.c_int,        # incx
    ctypes.POINTER(ctypes.c_double),  # beta
    ctypes.POINTER(ctypes.c_double),  # y
    ctypes.c_int,        # incy
]
cublas.cublasDgemv_v2.restype = ctypes.c_int

# cuBLAS handle (global)
_cublas_handle = None

def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = cublasHandle_t()
        cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    return _cublas_handle

@register("cublasDgemv_v2", "cublasDgemv_v2", False, namespace="baseline")
def cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = _get_cublas_handle()

    # Convert trans
    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]

    # Convert scalars
    alpha_c = ctypes.c_double(float(alpha))
    beta_c = ctypes.c_double(float(beta))

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_double))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_double))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_double))

    # Call cuBLAS
    status = cublas.cublasDgemv_v2(
        handle, trans, m, n,
        ctypes.byref(alpha_c),
        A_ptr, lda,
        x_ptr, incx,
        ctypes.byref(beta_c),
        y_ptr, incy
    )

    if status != 0:
        raise RuntimeError(f"cublasDgemv_v2 failed with status {status}")

    return y
