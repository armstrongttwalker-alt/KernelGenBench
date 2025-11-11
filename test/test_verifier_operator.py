from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source


triton_addmm_code = """@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    print("In custom addmm!!!!!!!!!!!!!!!")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    # assert broadcastable_to(
    #     bias.shape, (mat1.shape[0], mat2.shape[1])
    # ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    # mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch.cuda.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            32,
            32,
            32,
        )
    return out"""
# class VerifyConfig:
#     run_name: str
#     test_type: str = "accuracy" # accuracy, performance, both
#     run_dir: str = os.path.join(REPO_TOP_DIR, "runs")
#     store_type: str = "local"
#     strict_check: bool = False
#     seed: int = 42
#     sample_id: int = 0
#     save_log: bool = True

triton_arange_code = """import math
@triton.jit
def arange_func(y_ptr, start, end, step, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    y_ptr += pid * BLOCK_SIZE
    step_offset = pid * BLOCK_SIZE * step

    cols = tl.arange(0, BLOCK_SIZE)
    arange_val = cols * step + step_offset + start
    mask = cols + pid * BLOCK_SIZE
    tl.store(y_ptr + cols, arange_val, mask=mask < size)

def arange_start(
    start, end, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
):
    print("In custom arange_start!!!!!!!!!!!!!!!")
    if dtype is torch.int64:
        sgn = (step > 0) - (step < 0)
        size = (end - start + step - sgn) // step
    else:
        size = math.ceil((end - start) / step)

    BLOCK_SIZE = 128
    grid = triton.cdiv(size, BLOCK_SIZE)

    if dtype is None:
        dtype = torch.int64

    if pin_memory is None:
        pin_memory = False

    result = torch.empty((size,), device=device, dtype=dtype, pin_memory=pin_memory)
    arange_func[grid,](result, start, end, step, size, BLOCK_SIZE)
    return result

def arange_start_step(
    start, end, step, *, dtype=None, layout=None, device=None, pin_memory=None
):
    print("In custom arange_start_step!!!!!!!!!!!!!!!")
    return arange_start(
        start, end, step, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )

def arange(end, *, dtype=None, layout=None, device=None, pin_memory=None):
    print("In custom arange!!!!!!!!!!!!!!!")
    return arange_start(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )"""

config = VerifyConfig(
    run_name="test_verifier",
    test_type="both",
    run_dir="/root/tmp/runs",
    store_type="local",
    strict_check=True,
    seed=42,
    sample_id=0,
    save_log=True,
)

verifier = Verifier(config)

def test_verifier_operator():
    result = verifier.only_verify(
        name_source_map=[
            VerifyRequest(
                source=[Source(
                    source=triton_arange_code,
                    function_name="arange"
                )]
            )
        ], 
        test_type="accuracy"        # accuracy, performance, both
    )[-1][0]
    print("Verification Result:", result)


if __name__ == "__main__":
    test_verifier_operator()