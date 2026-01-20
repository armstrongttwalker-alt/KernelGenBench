import torch
import triton
import triton.language as tl

@triton.jit
def prelu_kernel(
    x_ptr,              # input tensor
    w_ptr,              # weight tensor (scalar or 1D)
    y_ptr,              # output tensor
    sizes_ptr,          # int32[MAX_DIMS] sizes of x
    x_strides_ptr,      # int64[MAX_DIMS] strides of x (in elements)
    y_strides_ptr,      # int64[MAX_DIMS] strides of y (in elements)
    w_is_scalar,        # int32 flag: 1 if scalar weight, 0 otherwise
    w_stride0,          # int64 stride for weight[0] if 1D
    w_size0,            # int32 size of weight if 1D (C)
    n_elements,         # total number of elements in x/y
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Cast to 64-bit for safe indexing math
    rem = offs.to(tl.int64)

    x_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    y_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    ch_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Compute multi-dimensional index from linear index and translate to memory offsets
    # Iterate from last dim to first
    for rd in range(MAX_DIMS):
        i = MAX_DIMS - 1 - rd
        size_i = tl.load(sizes_ptr + i).to(tl.int64)
        # ensure size_i >= 1 (should be ensured by host but avoid div by zero)
        size_i = tl.maximum(size_i, 1)
        idx_i = rem % size_i
        rem = rem // size_i

        xi_stride = tl.load(x_strides_ptr + i)
        yi_stride = tl.load(y_strides_ptr + i)
        x_off += idx_i * xi_stride
        y_off += idx_i * yi_stride

        if i == 1:
            ch_idx = idx_i

    # Load input
    x_val = tl.load(x_ptr + x_off, mask=mask, other=0)

    # Compute weight value per element
    w_scalar = tl.load(w_ptr)  # safe to load once
    # For per-channel: guarded masked load (mask prevents OOB usage)
    wch_mask = mask & (ch_idx < tl.load(w_size0 + 0, eviction_policy='evict_last') if False else mask)  # placeholder to satisfy syntax
    # Implement per-channel load safely without relying on the above trick
    # We still compute w_channel with a mask that ensures we don't read out of bounds.
    w_len = tl.load((w_size0 + 0) if False else w_size0, eviction_policy='evict_last') if False else w_size0  # dummy to keep constants in register
    w_len = w_len  # no-op; w_len remains compile-time arg
    w_mask = mask  # since weight length equals sizes[1], ch_idx is always valid when not scalar
    w_channel = tl.load(w_ptr + ch_idx * w_stride0, mask=w_mask, other=w_scalar)

    # Select appropriate weight based on scalar flag
    use_scalar = w_is_scalar != 0
    w_val = tl.where(use_scalar, w_scalar, w_channel)

    # PReLU: y = x if x>0 else w*x
    y_val = tl.where(x_val > 0, x_val, w_val * x_val)

    # Store output
    tl.store(y_ptr + y_off, y_val, mask=mask)

def non_torch_prelu(self: torch.Tensor, weight: torch.Tensor):
    # Validate device and dtypes
    assert self.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert self.device == weight.device, "Inputs must be on the same device"
    # Match PyTorch semantics: weight must be scalar or 1D of size C (self.size(1))
    if weight.numel() == 1 or weight.dim() == 0:
        w_is_scalar = 1
    else:
        if weight.dim() != 1:
            raise ValueError("prelu: weight must be a scalar or 1D tensor of shape (C,)")
        if self.dim() < 2:
            raise ValueError("prelu: per-channel weight requires input with at least 2 dims (N, C, ...)")
        if weight.numel() != self.size(1):
            raise ValueError(f"prelu: weight.numel() ({weight.numel()}) must equal input.size(1) ({self.size(1)})")
        w_is_scalar = 0

    # Ensure weight dtype matches input dtype for kernel arithmetic
    if weight.dtype != self.dtype:
        weight = weight.to(self.dtype)

    # Prepare output (preserve memory format)
    out = torch.empty_like(self)

    # Early return for empty tensors
    n_elements = self.numel()
    if n_elements == 0:
        return out

    # Kernel meta
    BLOCK_SIZE = 1024
    MAX_DIMS = 8
    if self.dim() > MAX_DIMS:
        raise ValueError(f"prelu: input with dim={self.dim()} exceeds MAX_DIMS={MAX_DIMS}")

    # Build sizes and strides arrays (padded to MAX_DIMS)
    sizes = [1] * MAX_DIMS
    x_strides = [0] * MAX_DIMS
    y_strides = [0] * MAX_DIMS
    for i in range(self.dim()):
        sizes[i] = self.size(i)
        x_strides[i] = self.stride(i)
        y_strides[i] = out.stride(i)

    # Create device tensors for sizes/strides
    sizes_t = torch.tensor(sizes, dtype=torch.int32, device=self.device)
    x_strides_t = torch.tensor(x_strides, dtype=torch.int64, device=self.device)
    y_strides_t = torch.tensor(y_strides, dtype=torch.int64, device=self.device)

    # Weight stride and size
    if w_is_scalar == 1:
        w_stride0 = 0
        w_size0 = torch.tensor(1, dtype=torch.int32, device=self.device)
    else:
        w_stride0 = weight.stride(0)
        w_size0 = torch.tensor(weight.size(0), dtype=torch.int32, device=self.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    prelu_kernel[grid](
        self,
        weight,
        out,
        sizes_t,
        x_strides_t,
        y_strides_t,
        w_is_scalar,
        w_stride0,
        w_size0,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_DIMS=MAX_DIMS,
    )
    return out
