from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code

from .sampler.generate_samples import (
    TritonKernelGenerateArgs,
)


class TritonKernelGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    # @print_prompt
    def generate_prompt(self, info: TritonKernelGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.code.strip() == info.old_code.strip():
                console.print("Generating prompt for Triton kernel fix...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for Triton kernel optimization...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for Triton kernel optimization...")
            return self.generate_prompt_for_optimization(info)
        console.print("Generating prompt for new Triton kernel...")
        return self.generate_prompt_for_new(info)

    def generate_prompt_for_optimization(self, info: TritonKernelGenerateArgs):
        # Implement the logic to generate the prompt for the optimization of the triton kernel function
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to optimize the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the same functionality as user provided description: {info.func_desc}\n"
        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        prompt += f"Make sure to consider the input and output args of the function as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function that launches the kernel.\n"
        prompt += f"The wrapper function name must be exactly the same as the provided function name: {info.op_name}\n"
        
        # Add Wiki reference implementations if available
        if info.wiki_reference:
            prompt += f"\n## Reference Implementations\n"
            prompt += f"Here are some reference implementations from similar operators that you can learn from:\n"
            try:
                if isinstance(info.wiki_reference, list):
                    for idx, ref in enumerate(info.wiki_reference[:3], 1):
                        if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                            prompt += f"\nReference {idx} (from {ref['link']}):\n"
                            prompt += "```python\n"
                            prompt += f"{ref['code']}\n"
                            prompt += "```\n"
                elif isinstance(info.wiki_reference, dict) and 'data' in info.wiki_reference:
                    refs = info.wiki_reference['data']
                    if isinstance(refs, list):
                        for idx, ref in enumerate(refs[:3], 1):
                            if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                                prompt += f"\nReference {idx} (from {ref['link']}):\n"
                                prompt += "```python\n"
                                prompt += f"{ref['code']}\n"
                                prompt += "```\n"
            except Exception as e:
                pass
        
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        
        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"

        return prompt


    def generate_prompt_for_fix(self, info: TritonKernelGenerateArgs):
        # Implement the logic to generate the prompt for fixing the Triton kernel
        # Use VerifyResult information when available
        check_result = info.check_result
        op_name = check_result.op_name if check_result and check_result.op_name else info.op_name
        
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to fix the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        
        prompt += f"The Triton kernel function that needs to be fixed is:\n"
        prompt += "```python\n"
        prompt += f"{check_result.code}\n"
        prompt += "```\n"
        
        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        
        prompt += f"The error message from the previous test run is:\n"
        prompt += f"{check_result.traceback}\n"
        
        if check_result.params:
            prompt += f"\nThe parameters of the test function are as follows:\n"
            prompt += f"{check_result.params}\n"
        
        # Add impl_info guidance if available and contains multiple operators
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"\nIMPORTANT: This PyTorch API is internally implemented using multiple ATen C++ operators. "
            prompt += f"You need to provide separate Python wrapper functions for each ATen operator interface, but they can share the same Triton kernel(s).\n"
            prompt += f"The ATen operators involved are:\n"
            for op in info.impl_info:
                prompt += f"  - {op}\n"
            prompt += f"\nYour implementation should include:\n"
            prompt += f"1. One or more @triton.jit decorated kernel function(s) that implement the core computation logic\n"
            prompt += f"2. Multiple Python wrapper functions, each corresponding to one ATen operator interface\n"
            prompt += f"3. For syntax considerations, replace '.' with '_' in the operator names when defining the Python wrapper function names\n"
            prompt += f"\nThe Python wrapper functions should have signatures like:\n"
            for op, _ in info.impl_info:
                py_op_name = op.replace(".", "_")
                prompt += f"  - def {py_op_name}(...): # This wrapper calls the shared kernel(s)\n"
            prompt += f"\nThese wrapper functions can internally call the same Triton kernel(s) with appropriate parameters.\n"
        else:
            prompt += f"\nThe Triton kernel function name should be {op_name}.\n"
        
        prompt += f"\nThe input and output args of the function are as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function(s) that launch the kernel.\n"
        
        # Modify this part based on whether multiple operators are involved
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"You must provide wrapper functions for all the ATen operators mentioned above.\n"
        else:
            prompt += f"The wrapper function name must be exactly the same as the provided function name: {op_name}\n"
        
        # Add Wiki reference implementations if available
        if info.wiki_reference:
            prompt += f"\n## Reference Implementations\n"
            prompt += f"Here are some reference implementations from similar operators that you can learn from:\n"
            try:
                if isinstance(info.wiki_reference, list):
                    for idx, ref in enumerate(info.wiki_reference[:3], 1):
                        if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                            prompt += f"\nReference {idx} (from {ref['link']}):\n"
                            prompt += "```python\n"
                            prompt += f"{ref['code']}\n"
                            prompt += "```\n"
                elif isinstance(info.wiki_reference, dict) and 'data' in info.wiki_reference:
                    refs = info.wiki_reference['data']
                    if isinstance(refs, list):
                        for idx, ref in enumerate(refs[:3], 1):
                            if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                                prompt += f"\nReference {idx} (from {ref['link']}):\n"
                                prompt += "```python\n"
                                prompt += f"{ref['code']}\n"
                                prompt += "```\n"
            except Exception as e:
                pass
        
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"

        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch function, only focus on fixing the Triton kernel logic based on the error message provided above.\n"

        return prompt
    

    def generate_prompt_for_new(self, info: TritonKernelGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to generate a Triton kernel function.\n"
        prompt += f"Here is an example of a PyTorch function and its corresponding Triton kernel implementation:\n"
        prompt += f"PyTorch function:\n"
        prompt += """def add(A, B):
        return torch.add(A, B)
    """.strip() + "\n"
        prompt += f"Triton kernel implementation:\n"
        prompt += """import triton
    import triton.language as tl


    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                y_ptr,  # *Pointer* to second input vector.
                output_ptr,  # *Pointer* to output vector.
                n_elements,  # Size of the vector.
                BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                # NOTE: `constexpr` so it can be used as a shape value.
                ):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
    """.strip() + "\n"
        prompt += f"You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel should implement the same functionality as the following PyTorch function:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        
        # Add impl_info guidance if available and contains multiple operators
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"\nIMPORTANT: This PyTorch API is internally implemented using multiple ATen C++ operators. "
            prompt += f"You need to provide separate Python wrapper functions for each ATen operator interface, but they can share the same Triton kernel(s).\n"
            prompt += f"The ATen operators involved are:\n"
            for op in info.impl_info:
                prompt += f"  - {op}\n"
            prompt += f"\nYour implementation should include:\n"
            prompt += f"1. One or more @triton.jit decorated kernel function(s) that implement the core computation logic\n"
            prompt += f"2. Multiple Python wrapper functions, each corresponding to one ATen operator interface\n"
            prompt += f"3. For syntax considerations, replace '.' with '_' in the operator names when defining the Python wrapper function names\n"
            prompt += f"\nThe Python wrapper functions should have signatures like:\n"
            for op, _ in info.impl_info:
                py_op_name = op.replace(".", "_")
                prompt += f"  - def {py_op_name}(...): # This wrapper calls the shared kernel(s)\n"
            prompt += f"\nThese wrapper functions can internally call the same Triton kernel(s) with appropriate parameters.\n"
        else:
            prompt += f"The Triton kernel function name should be {info.op_name}.\n"
        
        prompt += f"\nThe input and output args of the function are as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The generated code should include both the Triton kernel definition (with @triton.jit) and the Python wrapper function(s) that launch the kernel.\n"
        
        # Modify this part based on whether multiple operators are involved
        if info.impl_info is not None and len(info.impl_info) > 1:
            prompt += f"You must provide wrapper functions for all the ATen operators mentioned above.\n"
        else:
            prompt += f"The wrapper function name must be exactly the same as the provided function name: {info.op_name}\n"
        
        # Add Wiki reference implementations if available
        if info.wiki_reference:
            prompt += f"\n## Reference Implementations\n"
            prompt += f"Here are some reference implementations from similar operators that you can learn from:\n"
            try:
                if isinstance(info.wiki_reference, list):
                    for idx, ref in enumerate(info.wiki_reference[:3], 1):
                        if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                            prompt += f"\nReference {idx} (from {ref['link']}):\n"
                            prompt += "```python\n"
                            prompt += f"{ref['code']}\n"
                            prompt += "```\n"
                elif isinstance(info.wiki_reference, dict) and 'data' in info.wiki_reference:
                    refs = info.wiki_reference['data']
                    if isinstance(refs, list):
                        for idx, ref in enumerate(refs[:3], 1):
                            if isinstance(ref, dict) and 'link' in ref and 'code' in ref:
                                prompt += f"\nReference {idx} (from {ref['link']}):\n"
                                prompt += "```python\n"
                                prompt += f"{ref['code']}\n"
                                prompt += "```\n"
            except Exception as e:
                pass
        
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        
        prompt += f"You must generate the valid Triton code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        
        return prompt

    def _init_data(self, kwargs):
        config = kwargs.dict()
        if "from_mcp" in config:
            self.from_mcp = config.pop("from_mcp")
        if "check_result" in config and isinstance(config["check_result"], dict):
            from sandbox.utils.accuracy_utils import VerifyResult
            config["check_result"] = VerifyResult(**config["check_result"])
        if self.generation_config.use_ai_advice is True:
            console.print("AI advice generation is enabled for Triton kernel generator.")
            self.ai_advice = TritonKernelAdviceGenerator(self.generation_config)
            config["user_advice"] = self.ai_advice(TritonKernelGenerateArgs(**config))
        # config["op_name"] = config.pop("triton_kernel_name", None)
        config = TritonKernelGenerateArgs(**config)
        self.kernel_name = config.triton_kernel_name
        return config
    
    def post_process(self, results: list) -> list:
        codes = super().post_process(results)
        names = [r[1].op_name for r in results]
        sample_id = [r[1].sample_id for r in results]
        processed_results = []
        for res in codes:
            extracted_code = extract_first_code(res, ["python", "cpp"])
            if extracted_code is not None:
                processed_results.append(extracted_code.strip())
            else:
                console.print("Code extraction failed, using raw output.")
                processed_results.append(res.strip())
        
        # Apply decouple_bench if from_mcp is True
        if self.from_mcp:
            for i in range(len(processed_results)):
                processed_results[i] = self.decouple_bench(processed_results[i], self.kernel_name)
            return processed_results
        else:
            return [[code, name, sample_id] for code, name, sample_id in zip(processed_results, names, sample_id)]

    def decouple_bench(self, code: str, kernel_name: str = None) -> str:
        """
        Convert Triton kernel function name for MCP compatibility.
        - Replace def {kernel_name}( with def {kernel_name}_triton(
        """
        import re

        kernel_name = kernel_name or self.kernel_name
        assert kernel_name is not None, "kernel_name must be provided either in argument or as self.kernel_name"
        
        # Replace the wrapper function name: def {kernel_name}( with def {kernel_name}_triton(
        # Use word boundary to ensure exact match of function name
        pattern = rf'\bdef\s+{re.escape(kernel_name)}\s*\('
        replacement = f'def {kernel_name}_triton('
        code = re.sub(pattern, replacement, code)
        
        return code


class TritonKernelAdviceGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: TritonKernelGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.code.strip() == info.old_code.strip():
                console.print("Generating prompt for Triton kernel fix advice...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for Triton kernel optimization advice...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for Triton kernel optimization advice...")
            return self.generate_prompt_for_optimization(info)
        console.print("No advice generated for new Triton kernel.")
        return ""
    
    def generate_prompt_for_fix(self, info: TritonKernelGenerateArgs):
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to provide advice on how to fix the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be fixed is\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n"
        prompt += f"The Triton kernel got the following error when running the test function: {info.check_result.traceback}\n"
        prompt += f"The parameters of the test function are as follows:\n"
        prompt += f"{info.check_result.params}\n"
        prompt += f"Please provide concise and specific advice on what needs to be fixed in the Triton kernel function to resolve the error. Focus on the most likely issues based on the error message and the provided parameters.\n"
        prompt += f"Do not consider the logical correctness of the torch function, the test function.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the fixed code, only provide advice on what needs to be changed or improved in the existing code.\n"

        return prompt
    
    def generate_prompt_for_optimization(self, info: TritonKernelGenerateArgs):
        prompt = f"You are a skilled GPU programmer proficient in Triton. Your task is to provide advice on how to optimize the following Triton kernel function that implements the same functionality as a given PyTorch function. You must strictly adhere to the following specifications:\n"
        prompt += f"The Triton kernel function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the same functionality as user provided description: {info.func_desc}\n"
        prompt += f"And reference PyTorch function is as follows:\n"
        prompt += "```python\n"
        prompt += f"{info.torch_kernel_code}\n"
        prompt += "```\n"
        prompt += f"Please provide concise and specific advice on what needs to be optimized in the Triton kernel function to fix correctness or improve performance. Focus on the most likely issues based on the provided function description and the reference PyTorch function.\n"
        prompt += f"Do not consider the logical correctness of the torch function, the test function.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the optimized code, only provide advice on what needs to be changed or improved in the existing code.\n"
        
        return prompt
    
    def post_process(self, results: list) -> list:
        results = super().post_process(results)
        print("post process advice results:", results)
         # only return the first result
        return results[0]