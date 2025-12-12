from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
import os
os.environ["FLAGBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"

triton_square_code = "import torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\ndef square_kernel(\n    x_ptr,            # *Pointer* to input tensor\n    out_ptr,          # *Pointer* to output tensor\n    n_elements,       # Total number of elements\n    BLOCK_SIZE: tl.constexpr,  # Number of elements per program\n):\n    pid = tl.program_id(axis=0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    x = tl.load(x_ptr + offsets, mask=mask)\n    y = x * x\n    tl.store(out_ptr + offsets, y, mask=mask)\n\n\ndef _launch_square_kernel(x: torch.Tensor, out: torch.Tensor):\n    n_elements = out.numel()\n    if n_elements == 0:\n        return\n    assert x.is_cuda and out.is_cuda, \"Inputs must be CUDA tensors\"\n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    square_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)\n\n\ndef square(x: torch.Tensor):\n    \"\"\"\n    Wrapper for ATen operator: ('square', <Autograd.disable: False>)\n    \"\"\"\n    assert x.is_cuda, \"Input must be a CUDA tensor\"\n    x_contig = x.contiguous()\n    out = torch.empty_like(x_contig)\n    _launch_square_kernel(x_contig, out)\n    return out\n\n\ndef square_out(x: torch.Tensor, out: torch.Tensor):\n    \"\"\"\n    Wrapper for ATen operator: ('square.out', <Autograd.disable: False>)\n    \"\"\"\n    assert x.is_cuda and out.is_cuda, \"Inputs must be CUDA tensors\"\n    assert x.shape == out.shape, \"Input and out must have the same shape\"\n    assert x.dtype == out.dtype, \"Input and out must have the same dtype\"\n\n    if x.is_contiguous() and out.is_contiguous():\n        _launch_square_kernel(x, out)\n    else:\n        x_contig = x.contiguous()\n        if out.is_contiguous():\n            _launch_square_kernel(x_contig, out)\n        else:\n            tmp = torch.empty_like(x_contig)\n            _launch_square_kernel(x_contig, tmp)\n            out.copy_(tmp)\n    return out"

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
                    source=triton_square_code,
                    function_name="square"
                )]
            )
        ], 
        test_type="accuracy"        # accuracy, performance, both
    )[-1][0]
    print("Verification Result:", result)


if __name__ == "__main__":
    test_verifier_operator()