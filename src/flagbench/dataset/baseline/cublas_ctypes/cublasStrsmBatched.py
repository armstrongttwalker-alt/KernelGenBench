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
        _cublas_func = libcublas.cublasStrsmBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # side
            ctypes.c_int,                    # uplo
            ctypes.c_int,                    # trans
            ctypes.c_int,                    # diag
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_float),  # alpha (device)
            ctypes.POINTER(ctypes.c_void_p), # Aarray (device pointer array)
            ctypes.c_int,                    # lda
            ctypes.POINTER(ctypes.c_void_p), # Barray (device pointer array)
            ctypes.c_int,                    # ldb
            ctypes.c_int                     # batchCount
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

def cublasStrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount):
    '''ctypes cuBLAS C API baseline for cublasStrsmBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Device pointer arrays (int64 tensor on GPU)
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())

    # Get cached scalar GPU tensor for alpha
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(ldb),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasStrsmBatched failed with status {status}")
    return Barray

if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float32
    device = 'cuda'

    batchCount = 2
    m = 4
    n = 3
    alpha = 0.7

    side = 0   # LEFT
    uplo = 1   # UPPER
    trans = 0  # N
    diag = 0   # NON_UNIT

    As = []
    Bs_cm = []
    Bs_orig = []
    for i in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals).to(device=device, dtype=dtype)
        B_rm = torch.randn(m, n, dtype=dtype, device=device)

        As.append(A)
        Bs_orig.append(B_rm.clone())
        Bs_cm.append(B_rm.t().contiguous())

    Aarray = torch.tensor([A.data_ptr() for A in As], dtype=torch.int64, device=device)
    Barray = torch.tensor([B.data_ptr() for B in Bs_cm], dtype=torch.int64, device=device)

    lda = m
    ldb = m

    result = cublasStrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount)
    assert result is not None

    for i in range(batchCount):
        inv_diag = 1.0 / torch.diag(As[i])
        result_rm = Bs_cm[i].t().contiguous()
        expected = alpha * (inv_diag.view(-1, 1) * Bs_orig[i])
        torch.testing.assert_close(result_rm, expected, rtol=1e-4, atol=1e-4)

    print("✓ cublasStrsmBatched test passed")