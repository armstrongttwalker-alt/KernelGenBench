import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()

        # Create handle
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        # Setup SetPointerMode function (once)
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int

        # Set to device mode (once)
        _cublas_set_pointer_mode(_cublas_handle, 1)

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasStrmm_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,             # handle
            ctypes.c_int,                # side
            ctypes.c_int,                # uplo
            ctypes.c_int,                # trans
            ctypes.c_int,                # diag
            ctypes.c_int,                # m
            ctypes.c_int,                # n
            ctypes.POINTER(ctypes.c_float),  # alpha (device pointer)
            ctypes.c_void_p,             # A (device pointer)
            ctypes.c_int,                # lda
            ctypes.c_void_p,             # B (device pointer)
            ctypes.c_int,                # ldb
            ctypes.c_void_p,             # C (device pointer)
            ctypes.c_int                 # ldc
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasStrmm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasStrmm_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        A_ptr,
        lda,
        B_ptr,
        ldb,
        C_ptr,
        ldc
    )

    return C

if __name__ == "__main__":
    torch.manual_seed(1234)
    m, n = 4, 3
    alpha = 1.5

    A = torch.tril(torch.randn(m, m, dtype=torch.float32, device='cuda'))
    A.fill_diagonal_(2.0)
    B = torch.randn(m, n, dtype=torch.float32, device='cuda')

    A_cm = A.t().contiguous()
    B_cm = B.t().contiguous()
    C_cm = torch.empty(n, m, dtype=torch.float32, device='cuda')
    alpha_t = torch.tensor([alpha], dtype=torch.float32, device='cuda')

    # side=LEFT(0), uplo=LOWER(0), trans=N(0), diag=NON_UNIT(0)
    result = cublasStrmm_v2(0, 0, 0, 0, m, n, alpha, A_cm, m, B_cm, m, C_cm, m)
    out = result.t().contiguous()
    expected = alpha * (A @ B)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    print("✓ cublasStrmm_v2 test passed")