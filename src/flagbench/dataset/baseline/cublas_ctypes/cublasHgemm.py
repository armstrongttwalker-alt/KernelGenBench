import torch
import ctypes
from sandbox.config import DEVICE as device
from flagbench import register

cublas = ctypes.CDLL("libcublas.so.12")
cublasHandle_t = ctypes.c_void_p
cublasOperation_t = ctypes.c_int
half = ctypes.c_uint16

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

cublas.cublasHgemm.argtypes = [
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(half), ctypes.POINTER(half), ctypes.c_int,
    ctypes.POINTER(half), ctypes.c_int,
    ctypes.POINTER(half), ctypes.POINTER(half), ctypes.c_int
]
cublas.cublasHgemm.restype = ctypes.c_int

_cublas_handle = None

def _get_cublas_handle():
    global _cublas_handle
    if _cublas_handle is None:
        _cublas_handle = cublasHandle_t()
        cublas.cublasCreate_v2(ctypes.byref(_cublas_handle))
    return _cublas_handle

def _float16_to_half(val):
    h = half()
    h.value = torch.tensor(val, dtype=torch.float16).view(torch.int16).item()
    return h

@register("cublasHgemm", "cublasHgemm", False, namespace="baseline")
def cublasHgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    handle = _get_cublas_handle()

    if isinstance(transa, str):
        transa = CUBLAS_OP_N if transa == 'N' else CUBLAS_OP_T
    if isinstance(transb, str):
        transb = CUBLAS_OP_N if transb == 'N' else CUBLAS_OP_T

    alpha_h = _float16_to_half(alpha)
    beta_h = _float16_to_half(beta)

    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(half))
    B_ptr = ctypes.cast(B.data_ptr(), ctypes.POINTER(half))
    C_ptr = ctypes.cast(C.data_ptr(), ctypes.POINTER(half))

    status = cublas.cublasHgemm(
        handle, transa, transb, m, n, k,
        ctypes.byref(alpha_h), A_ptr, lda,
        B_ptr, ldb, ctypes.byref(beta_h),
        C_ptr, ldc
    )

    if status != 0:
        raise RuntimeError(f"cublasHgemm failed with status {status}")

    return C
