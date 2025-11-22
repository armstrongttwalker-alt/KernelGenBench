from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code

from .sampler.generate_samples import (
    TestFuncGenerateArgs,
)

class TestFuncGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: TestFuncGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.test_func.strip() == info.old_code.strip():
                console.print("Generating prompt for test function fix...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for test function optimization...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for test function optimization...")
            return self.generate_prompt_for_optimization(info)
        console.print("Generating prompt for new test function...")
        return self.generate_prompt_for_new(info)

    def generate_prompt_for_optimization(self, info: TestFuncGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to optimize the following test function that compares the outputs of a PyTorch kernel and its \
            Triton implementation."
        prompt += "You must strictly adhere to the following specifications:\n"
        prompt += f"The test function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The test function should compare the following two kernels:\n"
        prompt += f"PyTorch kernel:\nbench.{info.torch_kernel_name}\n"
        prompt += f"Triton kernel:\nbench.triton.{info.triton_kernel_name}\n"
        prompt += f"The test function must follow exactly the same format as the original:\n"
        prompt += f"@label(\"{{torch_kernel_name}}\")\n"
        prompt += f"@parametrize(XXX)\n"
        prompt += f"def {{op_name}}(XXX):\n"
        prompt += f"    # initialize the input data\n"
        prompt += f"    XXXXX\n"
        prompt += f"    # if necessary, cast the input\n"
        prompt += f"    ref_input = to_reference(input, True)\n"
        prompt += f"    ref_out = bench.{{torch_kernel_name}}(ref_input)\n"
        prompt += f"    res_out = bench.triton.{{triton_kernel_name}}(input)\n"
        prompt += f"    assert_close(res_out, ref_out, dtype)\n"
        prompt += f"You must use @parametrize to cover different input shapes and data types, do not import anything else like parametrize, the import logic will be handled automatically.\n"
        prompt += f"Do not consider the import source of to_reference function, it will be added automatically.\n"
        prompt += f"You must use bench.{{torch_kernel_name}} to call the torch kernel and bench.triton.{{triton_kernel_name}} to call the triton kernel rather than call them directly.\n"
        prompt += f"You must generate the valid test function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
        prompt += f"You should use assert_close to compare the outputs of the two kernels as the last step of the test function.\n"
        prompt += f"Use the output of torch kernel as the ground truth reference rather than manually create the result yourself.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch and triton kernels, only focus on optimizing the test function logic.\n"
        prompt += f"If you encounter tensor dtype mismatch issues (e.g., halftensor vs tensor), you can use to_reference(input, True) to cast the input to tf32 to resolve the problem. Note that to_reference only accepts tensor inputs, not tuple or list.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt

    def generate_prompt_for_fix(self, info: TestFuncGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to fix the following test function that compares the outputs of a PyTorch kernel and its \
            Triton implementation."
        prompt += "You must strictly adhere to the following specifications:\n"
        prompt += f"The test function that needs to be fixed is\n"
        prompt += "```python\n"
        prompt += f"{info.check_result.code}\n"
        prompt += "```\n"
        prompt += f"The test function should compare the following two kernels:\n"
        prompt += f"PyTorch kernel:\nbench.{info.torch_kernel_name}\n"
        prompt += f"Triton kernel:\nbench.triton.{info.triton_kernel_name}\n"
        prompt += f"The error message from the previous test run is: \n{info.check_result.traceback}\n"
        prompt += f"The test function must follow exactly the same format as the original:\n"
        prompt += "```python\n"
        prompt += f"@label(\"{{torch_kernel_name}}\")\n"
        prompt += f"@parametrize(\"a, b\", [(1, 2), (3, 4)])\n"
        prompt += f"@parametrize(\"dtype\", [torch.float16, torch.float32, torch.bfloat16])\n"
        prompt += f"def {{op_name}}(XXX):\n"
        prompt += f"    # initialize the input data\n"
        prompt += f"    XXXXX\n"
        prompt += f"    # if necessary, cast the input\n"
        prompt += f"    ref_input = to_reference(input, True)\n"
        prompt += f"    ref_out = bench.{{torch_kernel_name}}(ref_input)\n"
        prompt += f"    res_out = bench.triton.{{triton_kernel_name}}(input)\n"
        prompt += f"    assert_close(res_out, ref_out, dtype)\n"
        prompt += f"```\n"
        prompt += f"You must use @parametrize to cover different input shapes and data types, do not import anything else like parametrize, the import logic will be handled automatically.\n"
        prompt += f"Do not consider the import source of to_reference function, it will be added automatically.\n"
        prompt += f"You must use bench.{{torch_kernel_name}} to call the torch kernel and bench.triton.{{triton_kernel_name}} to call the triton kernel rather than call them directly.\n"
        prompt += f"You must generate the valid test function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
        prompt += f"You should use assert_close to compare the outputs of the two kernels as the last step of the test function.\n"
        prompt += f"Use the output of torch kernel as the ground truth reference rather than manually create the result yourself.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch and triton kernels, only focus on fixing the test function logic.\n"
        prompt += f"If you encounter tensor dtype mismatch issues (e.g., halftensor vs tensor), you can use to_reference(input, True) to cast the input to tf32 to resolve the problem. Note that to_reference only accepts tensor inputs, not tuple or list.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt


    # def generate_prompt_for_new(self, info: TestFuncGenerateArgs):
    #     # Implement the logic to generate the prompt for the test function
    #     prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to generate a test function that compares the outputs of a PyTorch kernel and its \
    #         Triton implementation. You must strictly adhere to the following specifications:\n"
    #     prompt += f"The test function name should be {info.op_name}.\n"
    #     prompt += f"The test function should compare the following two kernels:\n"
    #     prompt += f"PyTorch kernel:\n```python\n{info.torch_kernel_code}\n```\n"
    #     prompt += f"Triton kernel:\n```python\n{info.triton_kernel_code}\n```\n"
    #     prompt += f"The test function must follow exactly this format:\n"
    #     prompt += f"@label(\"{{torch_kernel_name}}\")\n"
    #     prompt += f"@parametrize(\"M, N\", [(1, 32), (160, 1024), (5333, 497)])\n"
    #     prompt += f"@parametrize(\"dtype\", [torch.float16, torch.float32, torch.bfloat16])\n"
    #     prompt += f"def {{op_name}}(M, N, dtype):\n"
    #     prompt += f"    # initialize the input data\n"
    #     prompt += f"    XXXXX\n"
    #     prompt += f"    # if necessary, cast the input"
    #     prompt += f"    ref_input = to_reference(input, True)\n"
    #     prompt += f"    ref_out = bench.{{torch_kernel_name}}(ref_input)\n"
    #     prompt += f"    res_out = bench.triton.{{triton_kernel_name}}(input)\n\n"
    #     prompt += f"    assert_close(res_out, ref_out, dtype, reduce_dim=M)\n"
    #     prompt += f"You must use @parametrize to cover different input shapes and data types, do not import anything else like parametrize, the import logic will be handled automatically.\n"
    #     prompt += f"Do not consider the import source of to_reference function, it will be added automatically.\n"
    #     prompt += f"You must use bench.{{torch_kernel_name}} to call the torch kernel and bench.triton.{{triton_kernel_name}} to call the triton kernel rather than call them directly.\n"
    #     prompt += f"You must generate the valid test function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
    #     prompt += f"You should use assert_close to compare the outputs of the two kernels as the last step of the test function.\n"
    #     prompt += f"The func assert_close has been imported for you to compare the outputs of the two kernels.\n"
    #     prompt += f"It has the following input args: assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1)\n"
    #     prompt += f"Use the output of torch kernel as the ground truth reference rather than manually create the result yourself.\n"
    #     prompt += f"You must use ```python ... ``` to format the code block.\n"
    #     prompt += f"use easy and common sense values for M, N in parametrize rather than extreme values.\n"
    #     prompt += f"do not use too many parametrize decorators, 2 or 3 are enough.\n"
    #     prompt += f"If you encounter tensor dtype mismatch issues (e.g., halftensor vs tensor), you can use to_reference(input, True) to cast the input to tf32 to resolve the problem. Note that to_reference only accepts tensor inputs, not tuple or list.\n"
    #     if info.user_advice:
    #         prompt += f"And the following user advice should be considered: {info.user_advice}\n"
    #     return prompt
    def generate_prompt_for_new(self, info: TestFuncGenerateArgs):
        # for test
        return "hello"

    def _init_data(self, kwargs):
        config = kwargs.dict()
        if "from_mcp" in config:
            self.from_mcp = config.pop("from_mcp")
        # config["op_name"] = config.pop("test_func_name", None)
        if "check_result" in config and isinstance(config["check_result"], dict):
            from sandbox.test.run_test import VerifyResult
            config["check_result"] = VerifyResult(**config["check_result"])
        config = TestFuncGenerateArgs(**config)
        self.kernel_name = config.kernel_name
        return config
    
    def _post_process(self, results: list) -> list:
        codes = super().post_process(results)
        names = [r[1].op_name for r in results]
        sample_id = [r[1].sample_id for r in results]
        processed_results = []
        for res in codes:
            console.rule("[bold blue]Raw Output from LLM")
            console.print(res)
            console.rule("[bold blue]End of Raw Output")
            extracted_code = extract_first_code(res, ["python", "cpp"])
            console.rule("[bold blue]Extracted Code Block")
            console.print(extracted_code)
            console.rule("[bold blue]End of Extracted Code Block")
            if extracted_code is not None:
                processed_results.append(extracted_code)
            else:
                console.print("Code extraction failed, using raw output.")
                processed_results.append(res)
        return [[code, name, sample_id] for code, name, sample_id in zip(processed_results, names, sample_id)]

    def post_process(self, results: list) -> list:
        results = self._post_process(results)
        if not self.from_mcp:
            test_func_prefix = """
import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch
""".strip() + "\n\n"
        else:
            test_func_prefix = "import torch\n\nimport pytest\n\n"
        for i in range(len(results)):
            results[i][0] = self.decouple_bench(results[i][0], self.kernel_name) if self.from_mcp else results[i][0]
            results[i][0] = test_func_prefix + results[i][0]
            results[i][0] = results[i][0].strip()
        return results

    def decouple_bench(self, code: str, kernel_name: str = None) -> str:
        """
        Convert custom decorators to pytest standard decorators.
        - Replace @label("XXX") with @pytest.mark.XXX
        - Replace @parametrize with @pytest.mark.parametrize
        - Replace bench.triton.{kernel_name} with {kernel_name}_triton
        - Replace bench.{kernel_name} with {kernel_name}
        """
        import re

        kernel_name = kernel_name or self.kernel_name
        assert kernel_name is not None, "kernel_name must be provided either in argument or as self.kernel_name"
        
        # Replace @label("XXX") with @pytest.mark.XXX
        # Pattern: @label("something") or @label('something')
        code = re.sub(
            r'@label\(["\']([^"\']+)["\']\)',
            r'@pytest.mark.\1',
            code
        )
        
        # Replace @parametrize with @pytest.mark.parametrize
        code = code.replace('@parametrize', '@pytest.mark.parametrize')
        
        # Replace bench.triton.{kernel_name} with {kernel_name}_triton
        # This must be done before replacing bench.{kernel_name} to avoid conflicts
        code = code.replace(f'bench.triton.{kernel_name}', f'{kernel_name}_triton')
        
        # Replace bench.{kernel_name} with {kernel_name}
        code = code.replace(f'bench.{kernel_name}', kernel_name)

        to_ref_func = '''def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float32)
    return ref_inp
'''
        if 'to_reference' in code:
            code = to_ref_func + '\n' + code
        
        return code

