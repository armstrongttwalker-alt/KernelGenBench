import torch
import ctypes
from sandbox.config import DEVICE as device
from flagbench import register

cublas = ctypes.CDLL("libcublas.so.12")
cublasHandle_t = ctypes.c_void_p
cublasOperation_t = ctypes.c_int
cuDoubleComplex = ctypes.c_double * 2

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

cublas.cublasZgemv_v2.argtypes = [
    cublasHandle_t, cublasOperation_t, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(cuDoubleComplex), ctypes.POINTER(cuDoubleComplex), ctypes.c_int,
    ctypes.POINTER(cuDoubleComplex), ctypes.c_int,
    ctypes.POINTER(cuDoubleComplex), ctypes.POINTER(cuDoubleComplex), ctypes.c_int
]
cublas.cublasZgemv_v2.restype = ctypes.c_int

_cublas_handle = None

def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = cublasHandle_t()
        cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    return _cublas_handle

def _complex128_to_cudoublecomplex(val):
    c = cuDoubleComplex()
    c[0] = val.real
    c[1] = val.imag
    return c

@register("cublasZgemv_v2", "cublasZgemv_v2", False, namespace="baseline")
def cublasZgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = _get_cublas_handle()

    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]

    alpha_c = _complex128_to_cudoublecomplex(alpha)
    beta_c = _complex128_to_cudoublecomplex(beta)

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(cuDoubleComplex))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(cuDoubleComplex))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(cuDoubleComplex))

    status = cublas.cublasZgemv_v2(
        handle, trans, m, n,
        ctypes.byref(alpha_c), A_ptr, lda,
        x_ptr, incx, ctypes.byref(beta_c),
        y_ptr, incy
    )

    if status != 0:
        raise RuntimeError(f"cublasZgemv_v2 failed with status {status}")

    return y
