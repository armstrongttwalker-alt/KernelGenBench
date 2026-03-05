import torch
import ctypes
from sandbox.config import DEVICE as device
from flagbench import register

# Load cuBLAS library
cublas = ctypes.CDLL("libcublas.so.12")

# cuBLAS types
cublasHandle_t = ctypes.c_void_p
cublasOperation_t = ctypes.c_int
cuComplex = ctypes.c_float * 2

# cublasOperation_t values
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

# Define cublasCgemv_v2 function signature
cublas.cublasCgemv_v2.argtypes = [
    cublasHandle_t,      # handle
    cublasOperation_t,   # trans
    ctypes.c_int,        # m
    ctypes.c_int,        # n
    ctypes.POINTER(cuComplex),  # alpha
    ctypes.POINTER(cuComplex),  # A
    ctypes.c_int,        # lda
    ctypes.POINTER(cuComplex),  # x
    ctypes.c_int,        # incx
    ctypes.POINTER(cuComplex),  # beta
    ctypes.POINTER(cuComplex),  # y
    ctypes.c_int,        # incy
]
cublas.cublasCgemv_v2.restype = ctypes.c_int

# cuBLAS handle (global)
_cublas_handle = None

def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = cublasHandle_t()
        cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    return _cublas_handle

def _torch_complex_to_cucomplex(val):
    """Convert torch complex scalar to cuComplex"""
    c = cuComplex()
    c[0] = val.real
    c[1] = val.imag
    return c

@register("cublasCgemv_v2", "cublasCgemv_v2", False, namespace="baseline")
def cublasCgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = _get_cublas_handle()

    # Convert trans
    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]

    # Convert scalars
    alpha_c = _torch_complex_to_cucomplex(alpha)
    beta_c = _torch_complex_to_cucomplex(beta)

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(cuComplex))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(cuComplex))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(cuComplex))

    # Call cuBLAS
    status = cublas.cublasCgemv_v2(
        handle, trans, m, n,
        ctypes.byref(alpha_c),
        A_ptr, lda,
        x_ptr, incx,
        ctypes.byref(beta_c),
        y_ptr, incy
    )

    if status != 0:
        raise RuntimeError(f"cublasCgemv_v2 failed with status {status}")

    return y
