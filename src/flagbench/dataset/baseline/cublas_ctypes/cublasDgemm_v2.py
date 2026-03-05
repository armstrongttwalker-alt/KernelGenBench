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
        status = _cublas_set_pointer_mode(_cublas_handle, 1)
        if status != 0:
            raise RuntimeError(f"cublasSetPointerMode_v2 failed with status {status}")

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasDgemm_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,            # handle
            ctypes.c_int,               # transa
            ctypes.c_int,               # transb
            ctypes.c_int,               # m
            ctypes.c_int,               # n
            ctypes.c_int,               # k
            ctypes.POINTER(ctypes.c_double),  # alpha
            ctypes.POINTER(ctypes.c_double),  # A
            ctypes.c_int,               # lda
            ctypes.POINTER(ctypes.c_double),  # B
            ctypes.c_int,               # ldb
            ctypes.POINTER(ctypes.c_double),  # beta
            ctypes.POINTER(ctypes.c_double),  # C
            ctypes.c_int                # ldc
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemm_v2(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasDgemm_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Allow string shorthand for operations
    if isinstance(transa, str):
        if transa == 'N':
            transa = 0
        elif transa == 'T':
            transa = 1
        elif transa == 'C':
            transa = 2
    if isinstance(transb, str):
        if transb == 'N':
            transb = 0
        elif transb == 'T':
            transb = 1
        elif transb == 'C':
            transb = 2

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double)), ctypes.c_int(lda),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_double)), ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_double)), ctypes.c_int(ldc)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemm_v2 failed with status {status}")
    return C

if __name__ == "__main__":
    torch.manual_seed(0)
    m, n, k = 5, 4, 3
    alpha = 1.25
    beta = -0.75

    A = torch.randn(m, k, dtype=torch.float64, device='cuda')
    B = torch.randn(k, n, dtype=torch.float64, device='cuda')
    C = torch.randn(m, n, dtype=torch.float64, device='cuda')
    A0, B0, C0 = A.clone(), B.clone(), C.clone()

    # Column-major trick: C=alpha*A@B+beta*C => cublas(N,N, n,m,k, alpha, B,n, A,k, beta, C,n)
    out = cublasDgemm_v2('N', 'N', n, m, k, alpha, B, n, A, k, beta, C, n)
    expected = alpha * (A0 @ B0) + beta * C0
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemm_v2 test passed")