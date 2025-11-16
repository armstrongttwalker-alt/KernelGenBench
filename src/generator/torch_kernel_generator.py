from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code

from .sampler.generate_samples import (
    TorchKernelGenerateArgs,
)


class TorchKernelGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: TorchKernelGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.code.strip() == info.old_code.strip():
                console.print("Generating prompt for Torch kernel fix...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for Torch kernel optimization...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for Torch kernel optimization...")
            return self.generate_prompt_for_optimization(info)
        console.print("Generating prompt for new Torch kernel...")
        return self.generate_prompt_for_new(info)
    
    def generate_prompt_for_optimization(self, info: TorchKernelGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch. Your task is to optimize the following PyTorch function that implements a specific functionality. The function is a ground truth implementation for a Triton kernel function.\n"
        prompt += f"The requested functionality is: {info.func_desc}\n"
        prompt += f"The PyTorch function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The PyTorch function must follow exactly the same format as the original:\n"
        prompt += f"def {{op_name}}(XXX):\n"
        prompt += f"    ...\n"
        prompt += f"You must generate the valid PyTorch code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the func_desc, only focus on optimizing the PyTorch function logic.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt

    def generate_prompt_for_fix(self, info: TorchKernelGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch. Your task is to fix the following PyTorch function that implements a specific functionality. The function is a ground truth implementation for a Triton kernel function.\n"
        prompt += f"The requested functionality is: {info.func_desc}\n"
        prompt += f"The PyTorch function that needs to be fixed is\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n"
        prompt += f"The error message from the previous test run is: {info.check_result.traceback}\n"
        prompt += f"The PyTorch function must follow exactly the same format as the original:\n"
        prompt += f"def {{op_name}}(XXX):\n"
        prompt += f"    ...\n"
        prompt += f"You must generate the valid PyTorch code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the func_desc, only focus on fixing the PyTorch function logic.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt

    def generate_prompt_for_new(self, info: TorchKernelGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch. Your task is to implement a PyTorch function as ground truth for a triton kernel request. Here are the triton kernel request details:\n"
        prompt += f"Functionality: {info.func_desc}\n"
        prompt += f"Triton Kernel Function Name: {info.op_name}\n"
        prompt += f"The input and output args of the function are as follows:\n"
        prompt += f"Input Args: {info.input_args}\n"
        prompt += f"Output Args: {info.output_args}\n"
        prompt += f"The PyTorch function name must be exactly the same as the triton kernel function name: {info.op_name}\n"
        prompt += f"If the requested functionality is as same as an existing PyTorch function, you should implement it using that PyTorch function. If the requested functionality is a combination of multiple existing PyTorch functions, you should implement it using those PyTorch functions. You must strictly follow the input and output args format provided above.\n"
        prompt += f"You must generate the valid PyTorch code directly without any explanations or additional text, and ensure no testing or benchmarking code is included. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt

    def _init_data(self, kwargs):
        config = kwargs.dict()
        if "check_result" in config and isinstance(config["check_result"], dict):
            from bench.sandbox.test.run_test import VerifyResult
            config["check_result"] = VerifyResult(**config["check_result"])
        # config["op_name"] = config.pop("torch_kernel_name", None)
        return TorchKernelGenerateArgs(**config)
    
    def post_process(self, results: list) -> list:
        results = super().post_process(results)
        processed_results = []
        for res in results:
            extracted_code = extract_first_code(res, ["python", "cpp"])
            if extracted_code is not None:
                processed_results.append(extracted_code.strip())
            else:
                console.print("Code extraction failed, using raw output.")
                processed_results.append(res.strip())
        return processed_results
    

class TorchKernelAdviceGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: TorchKernelGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.code.strip() == info.old_code.strip():
                return self.generate_prompt_for_fix(info)
            else:
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            return self.generate_prompt_for_optimization(info)
        return ""
    
    def generate_prompt_for_fix(self, info: TorchKernelGenerateArgs):
        prompt = f"You are a skilled software engineer proficient in PyTorch. Your task is to provide advice on how to fix the following PyTorch function that implements a specific functionality. You must strictly adhere to the following specifications:\n"
        prompt += f"The PyTorch function that needs to be fixed is\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the following functionality: {info.func_desc}\n"
        prompt += f"The error message from the previous test run is: {info.check_result.traceback}\n"
        prompt += f"The parameters of the test function are as follows:\n"
        prompt += f"{info.check_result.params}\n"
        prompt += f"Please provide concise and specific advice on what needs to be fixed in the PyTorch function to resolve the error. Focus on the most likely issues based on the error message and the provided parameters.\n"
        prompt += f"Do not consider the logical correctness of the func_desc, only focus on fixing the PyTorch function logic.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the fixed code, only provide advice on what needs to be changed or improved in the existing code.\n"

        return prompt
    
    def generate_prompt_for_optimization(self, info: TorchKernelGenerateArgs):
        prompt = f"You are a skilled software engineer proficient in PyTorch. Your task is to provide advice on how to optimize the following PyTorch function that implements a specific functionality. You must strictly adhere to the following specifications:\n"
        prompt += f"The PyTorch function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The function should implement the following functionality: {info.func_desc}\n"
        prompt += f"Please provide concise and specific advice on what needs to be optimized in the PyTorch function to fix correctness or improve performance. Focus on the most likely issues based on the provided function description.\n"
        prompt += f"Do not consider the logical correctness of the func_desc, only focus on optimizing the PyTorch function logic.\n"
        prompt += f"Make sure keep your advice concise and to the point, do not provide lengthy explanations.\n"
        prompt += f"Do not provide the optimized code, only provide advice on what needs to be changed or improved in the existing code.\n"
        
        return prompt