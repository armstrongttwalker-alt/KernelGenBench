"""
CublasPromptBuilder - cuBLAS 框架的 Prompt 构造器
"""

from typing import TYPE_CHECKING
from .prompt_builder import PromptBuilder
from flagbench.framework.generate_args import BaseGenerateArgs, CublasGenerateArgs

if TYPE_CHECKING:
    from sandbox.utils.accuracy_utils import VerifyResult


class CublasPromptBuilder(PromptBuilder):
    """cuBLAS 框架的 Prompt 构造器"""

    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        if not isinstance(gen_args, CublasGenerateArgs):
            raise TypeError(f"CublasPromptBuilder requires CublasGenerateArgs, got {type(gen_args)}")

        info: CublasGenerateArgs = gen_args

        prompt = "You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function that implements the same functionality as a cuBLAS baseline function.\\n"

        prompt += "\\nHere is an example of a cuBLAS baseline function and its corresponding Triton kernel implementation:\\n"
        prompt += "cuBLAS baseline function (SAXPY):\\n"
        prompt += """```python
def saxpy(n, alpha, x, incx, y, incy):
    '''SAXPY: y = alpha * x + y'''
    # cuBLAS C API call via ctypes
    cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
```
""".strip() + "\\n"

        prompt += "\\nTriton kernel implementation:\\n"
        prompt += """```python
import triton
import triton.language as tl

@triton.jit
def saxpy_kernel(n, alpha, x_ptr, incx, y_ptr, incy, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx
    y_idx = offsets * incy

    x = tl.load(x_ptr + x_idx, mask=mask)
    y = tl.load(y_ptr + y_idx, mask=mask)

    result = alpha * x + y
    tl.store(y_ptr + y_idx, result, mask=mask)

def saxpy(n, alpha, x, incx, y, incy):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    saxpy_kernel[grid](n, alpha, x, incx, y, incy, BLOCK_SIZE=1024)
    return y
```
""".strip() + "\\n"

        prompt += f"\\n\\nNow, please generate a Triton kernel for the following cuBLAS baseline function:\\n"
        prompt += f"Operation: {info.cublas_kernel_name}\\n"
        prompt += f"Type: {info.blas_operation_type}\\n"
        prompt += f"Description: {info.func_desc}\\n\\n"
        prompt += f"Baseline function:\\n```python\\n{info.baseline_code}\\n```\\n"

        if info.impl_info:
            prompt += f"\\nOperator Interface Information:\\n"
            if isinstance(info.impl_info, dict):
                if "name" in info.impl_info:
                    prompt += f"Function Name: {info.impl_info['name']}\\n"
                if "operation" in info.impl_info:
                    prompt += f"Operation: {info.impl_info['operation']}\\n"
                if "dtype" in info.impl_info:
                    prompt += f"Data Type: {info.impl_info['dtype']}\\n"
                if "args" in info.impl_info:
                    prompt += f"Arguments: {len(info.impl_info['args'])} parameters\\n"

        prompt += "\\nRequirements:\\n"
        prompt += "1. Implement the Triton kernel with proper memory coalescing\\n"
        prompt += "2. Use appropriate block sizes for GPU parallelization\\n"
        prompt += "3. Handle edge cases and boundary conditions\\n"
        prompt += "4. Ensure numerical stability\\n"
        prompt += "5. Return only the Triton kernel code without explanations\\n"

        return prompt

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        if not isinstance(gen_args, CublasGenerateArgs):
            raise TypeError(f"CublasPromptBuilder requires CublasGenerateArgs, got {type(gen_args)}")

        info: CublasGenerateArgs = gen_args
        has_history = info.history is not None and len(info.history) > 0

        prompt = "The previous Triton kernel implementation failed verification. Please analyze the error and generate a corrected version.\n\n"

        prompt += f"cuBLAS baseline function:\n```python\n{info.baseline_code}\n```\n\n"

        if has_history:
            # 多轮历史：逐轮展示 kernel + 报错
            history = info.history or []
            prompt += f"Below are ALL previous attempts ({len(history)} round(s)) and their error messages. Learn from each failure:\n"
            for entry in history:
                round_num = entry.get("round", "?")
                code = entry.get("code") or ""
                traceback = entry.get("traceback") or "(no error recorded)"
                params = entry.get("params")
                prompt += f"\n--- Attempt Round {round_num} ---\n"
                prompt += f"Kernel code:\n```python\n{code}\n```\n"
                prompt += f"Error message:\n{traceback}\n"
                if params:
                    prompt += f"Test parameters:\n{params}\n"
        else:
            # 单轮兼容逻辑
            if info.old_code:
                prompt += f"Previous failed implementation:\n```python\n{info.old_code}\n```\n\n"
            if info.check_result:
                prompt += f"Error information:\n{info.check_result.traceback}\n\n"
                if info.check_result.params:
                    prompt += f"Test parameters:\n{info.check_result.params}\n\n"

        prompt += "Please fix the issues and provide a corrected Triton kernel implementation.\n"

        return prompt

    def build_optimization(self, gen_args: BaseGenerateArgs) -> str:
        # 优化场景：暂时复用build_fix逻辑
        return self.build_fix(gen_args)
