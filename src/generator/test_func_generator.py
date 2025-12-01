from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code

from .sampler.generate_samples import (
    TestFuncGenerateArgs,
)

class TestFuncGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    # @print_prompt
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
        # console.print("Generating prompt for new test function...")
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
        prompt = "You are a test function generation expert proficient in PyTorch and Triton. Your task is to create a test python script that includes one or more test functions comparing the outputs of a PyTorch function and its Triton implementation. You must strictly adhere to the following specifications:\n\n"
        
        # 基本信息
        prompt += f"## Operator Information\n"
        prompt += f"- Operator name: {info.kernel_name}\n"
        prompt += f"- PyTorch API call: torch.ops.aten.{info.kernel_name}(...)\n"
        prompt += f"- Triton implementation call: Use the same API within `flagbench.use_gems()` context\n\n"
        
        # 算子的schema信息
        if info.operators:
            prompt += f"## Available Operator Schemas\n"
            prompt += f"The operator `{info.kernel_name}` has the following variants:\n"
            prompt += "```json\n"
            import json
            prompt += json.dumps(info.operators, indent=2)
            prompt += "\n```\n"
            prompt += "You MUST test ALL these variants. Each schema represents a different overload.\n\n"
        
        # 测试函数要求
        prompt += "## Test Function Requirements\n\n"
        prompt += "### 1. Function Structure\n"
        prompt += "Each test function must:\n"
        prompt += f"- Be decorated with `@label(f\"{info.kernel_name}\")`\n"
        prompt += "- Use `@parametrize()` decorator for test cases (similar to pytest.mark.parametrize)\n"
        prompt += "- Construct appropriate input tensors/scalars based on the schema\n"
        prompt += "- Call both PyTorch API and Triton implementation\n"
        prompt += "- Compare results using `assert_close()`\n\n"
        
        prompt += "### 2. API Calling Convention\n"
        prompt += "```python\n"
        prompt += f"# PyTorch reference implementation\n"
        prompt += f"ref_out = torch.ops.{info.ops_namespace}.{info.kernel_name}(...)\n\n"
        prompt += f"# Triton implementation\n"
        prompt += f"with flagbench.use_gems(REGISTERED_OPS):\n"
        prompt += f"    act_out = torch.ops.{info.ops_namespace}.{info.kernel_name}(...)\n\n"
        prompt += f"# Compare results\n"
        prompt += f"assert_close(act_out, ref_out)\n"
        prompt += "```\n\n"
        
        prompt += "### 3. Test Coverage Strategy\n"
        prompt += "- For schemas with the SAME number and types of arguments: Use `@parametrize` to test multiple cases in one function\n"
        prompt += "- For schemas with DIFFERENT argument counts or types: Create separate test functions\n"
        prompt += f"- When creating multiple test functions, ensure ALL use `@label(\"{info.kernel_name}\")`\n"
        prompt += "- Test various input shapes: small (e.g., (2, 3)), medium (e.g., (128, 256)), and reasonable large sizes\n"
        prompt += "- Test common dtypes: torch.float32, torch.float16, torch.bfloat16 (when applicable)\n\n"
        
        prompt += "### 4. Example Structure\n"
        prompt += "```python\n"
        prompt += "@label(\"example_op\")\n"
        prompt += '@parametrize("shape", [(2, 3), (128, 256), (512, 512)])\n'
        prompt += '@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])\n'
        prompt += "def test_example_op_tensor(shape, dtype):\n"
        prompt += "    input_tensor = torch.randn(shape, dtype=dtype, device='cuda')\n"
        prompt += "    other_tensor = torch.randn(shape, dtype=dtype, device='cuda')\n"
        prompt += "    \n"
        prompt += "    # Clone inputs for reference and triton implementations\n"
        prompt += "    ref_input = input_tensor.clone()\n"
        prompt += "    ref_other = other_tensor.clone()\n"
        prompt += "    \n"
        prompt += "    ref_out = torch.ops.aten.example_op(ref_input, ref_other)\n"
        prompt += "    \n"
        prompt += "    with flagbench.use_gems(REGISTERED_OPS):\n"
        prompt += "        act_out = torch.ops.aten.example_op(input_tensor, other_tensor)\n"
        prompt += "    \n"
        prompt += "    assert_close(act_out, ref_out, dtype=dtype)\n"
        prompt += "```\n\n"
        
        prompt += "## Important Constraints\n"
        prompt += "- DO NOT import `label`, `parametrize`, `assert_close`, or `flagbench` - these will be auto-imported\n"
        prompt += "- DO NOT include any explanations, comments (除了必要的代码注释), or unrelated code\n"
        prompt += "- Only output the test function code, ready to run\n"
        prompt += "- Wrap your output in ```python ``` code blocks\n"
        prompt += "- Use sensible, realistic values for test parameters (avoid extreme edge cases unless necessary)\n"
        prompt += "- All tensors should be on 'cuda' device\n"
        prompt += "- The `assert_close` function signature: `assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1)` - use it to compare outputs\n"
        prompt += f"- Function names should be descriptive, e.g., `test_{info.kernel_name}_tensor`, `test_{info.kernel_name}_scalar`\n\n"
        
        if info.user_advice:
            prompt += f"## Additional User Guidance\n{info.user_advice}\n\n"
        
        prompt += "Now generate the complete test functions based on the operator schemas provided above.\n"
        
        return prompt

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
            console.print(res, markup=False)
            console.rule("[bold blue]End of Raw Output")
            extracted_code = extract_first_code(res, ["python", "cpp"])
            console.rule("[bold blue]Extracted Code Block")
            console.print(extracted_code, markup=False)
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
import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
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

