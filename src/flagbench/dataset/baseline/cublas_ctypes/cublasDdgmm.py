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
        _cublas_func = libcublas.cublasDdgmm
        _cublas_func.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,     # mode (cublasSideMode_t)
            ctypes.c_int,     # m
            ctypes.c_int,     # n
            ctypes.c_void_p,  # A (const double*)
            ctypes.c_int,     # lda
            ctypes.c_void_p,  # x (const double*)
            ctypes.c_int,     # incx
            ctypes.c_void_p,  # C (double*)
            ctypes.c_int      # ldc
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasDdgmm'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Call cuBLAS C API
    func(handle, mode, m, n, A_ptr, lda, x_ptr, incx, C_ptr, ldc)

    return C

if __name__ == "__main__":
    # Constants for cublasSideMode_t
    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    # Test parameters
    m, n = 3, 4

    # Create test data on GPU (double precision)
    A = torch.randn(m, n, dtype=torch.float64, device='cuda')
    x_left = torch.randn(m, dtype=torch.float64, device='cuda')
    x_right = torch.randn(n, dtype=torch.float64, device='cuda')

    # Prepare column-major representations (as row-major of transposes)
    A_cm = A.t().contiguous()           # shape (n, m), row-major equals column-major of A (m, n)
    A_cm_in = A_cm.clone()
    x_left_in = x_left.clone()
    x_right_in = x_right.clone()

    # LEFT mode test: C = diag(x_left) * A
    C_cm_left = torch.empty_like(A_cm_in)
    res_left = cublasDdgmm(CUBLAS_SIDE_LEFT, m, n, A_cm_in, m, x_left_in, 1, C_cm_left, m)
    assert res_left is not None
    expected_left = (x_left.view(m, 1) * A)  # row scaling
    expected_left_cm = expected_left.t().contiguous()
    torch.testing.assert_close(res_left, expected_left_cm, rtol=1e-5, atol=1e-5)

    # RIGHT mode test: C = A * diag(x_right)
    C_cm_right = torch.empty_like(A_cm_in)
    res_right = cublasDdgmm(CUBLAS_SIDE_RIGHT, m, n, A_cm_in, m, x_right_in, 1, C_cm_right, m)
    assert res_right is not None
    expected_right = (A * x_right.view(1, n))  # column scaling
    expected_right_cm = expected_right.t().contiguous()
    torch.testing.assert_close(res_right, expected_right_cm, rtol=1e-5, atol=1e-5)

    print("✓ cublasDdgmm test passed")