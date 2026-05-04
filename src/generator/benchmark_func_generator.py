from .generator import BaseGenerator, print_prompt, console
from generator.sampler.utils import extract_first_code

from .sampler.generate_samples import (
    BenchmarkFuncGenerateArgs,
)

class BenchmarkFuncGenerator(BaseGenerator):
    def __init__(self, generation_config):
        super().__init__(generation_config)

    @print_prompt
    def generate_prompt(self, info: BenchmarkFuncGenerateArgs):
        if info.check_result is not None and info.check_result.success is False:
            if info.check_result.test_func.strip() == info.old_code.strip():
                console.print("Generating prompt for benchmark function fix...")
                return self.generate_prompt_for_fix(info)
            else:
                console.print("Generating prompt for benchmark function optimization...")
                return self.generate_prompt_for_optimization(info)
        if info.old_code is not None and len(info.old_code.strip()) > 0:
            console.print("Generating prompt for benchmark function optimization...")
            return self.generate_prompt_for_optimization(info)
        console.print("Generating prompt for new benchmark function...")
        return self.generate_prompt_for_new(info)
    
    def generate_prompt_for_optimization(self, info: BenchmarkFuncGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to optimize the following benchmark function that compares the performance of a PyTorch kernel and its Triton implementation. You must strictly adhere to the following specifications:\n"
        prompt += f"The benchmark function that needs to be optimized is\n"
        prompt += "```python\n"
        prompt += f"{info.old_code}\n"
        prompt += "```\n"
        prompt += f"The benchmark function should compare the following two kernels:\n"
        prompt += f"PyTorch kernel:\nbench.XXX\n"
        prompt += f"Triton kernel:\nbench.triton.XXX\n"
        prompt += f"You will be provided with a checked accuracy test function: \n```python\n{info.test_func_code}\n```\n"
        prompt += f"The benchmark function must follow exactly the same format as the original:\n"
        prompt += f"@label(\"{{torch_kernel_name}}_benchmark\")\n"
        prompt += f"@parametrize(XXX)\n"
        prompt += f"def {{op_name}}_benchmark(XXX):\n"
        prompt += f"    ...\n"
        prompt += f"Using @label and @parametrize decorators to cover all parameter combinations as in the test function rather than @triton.testing.perf_report or anything else.\n"
        prompt += f"The @label name should be the original operator name appended with '_benchmark'.\n"
        prompt += f"Do not include warm up cause triton.testing.do_bench already includes warm up.\n"
        prompt += f"Do not print anything in the benchmark function.\n"
        prompt += f"You must generate the valid benchmark function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch and triton kernels, only focus on optimizing the benchmark function logic.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt
    
    def generate_prompt_for_fix(self, info: BenchmarkFuncGenerateArgs):
        # Implement the logic to generate the prompt for the test function
        prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to fix the following benchmark function that compares the performance of a PyTorch kernel and its Triton implementation. You must strictly adhere to the following specifications:\n"
        prompt += f"The benchmark function that needs to be fixed is\n```python\n{info.check_result.code}\n```\n"
        prompt += f"The benchmark function should compare the following two kernels:\n"
        prompt += f"The error message from the previous test run is: {info.check_result.traceback}\n"
        prompt += f"You will be provided with a checked accuracy test function: \n```python\n{info.test_func_code}\n```\n"
        prompt += f"The benchmark function must follow exactly the same format as the original:\n"
        prompt += f"@label(\"{{torch_kernel_name}}_benchmark\")\n"
        prompt += f"@parametrize(XXX)\n"
        prompt += f"def {{op_name}}_benchmark(XXX):\n"
        prompt += f"    ...\n"
        prompt += f"Using @label and @parametrize decorators to cover all parameter combinations as in the test function rather than @triton.testing.perf_report or anything else.\n"
        prompt += f"The @label name should be the original operator name appended with '_benchmark'.\n"
        prompt += f"Do not include warm up cause triton.testing.do_bench already includes warm up.\n"
        prompt += f"Do not print anything in the benchmark function.\n"
        prompt += f"You must generate the valid benchmark function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
        prompt += f"You must use ```python ... ``` to format the code block.\n"
        prompt += f"Do not consider the logical correctness of the torch and triton kernels, only focus on fixing the benchmark function logic.\n"
        if info.user_advice:
            prompt += f"And the following user advice should be considered: {info.user_advice}\n"
        return prompt

#     def generate_prompt_for_new(self, info: BenchmarkFuncGenerateArgs):
#         prompt = f"You are a skilled software engineer proficient in PyTorch and Triton. Your task is to generate a benchmark function that compares the performance of a PyTorch kernel and its Triton implementation. You must strictly adhere to the following specifications:\n"
#         prompt += f"You will be provided with a checked accuracy test function, the PyTorch kernel code, and the Triton kernel code.\n"
#         prompt += "Here is an example of given a test function and how to write a benchmark function:\n"
#         prompt += "test function:\n"
#         prompt += "```python\n"
#         prompt += """@label("add")
# @parametrize("A_shape, B_shape", [((32,), (32,)), ((32, 32), (32, 32))])
# @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
# def test_add(A_shape, B_shape, dtype):
#     a = torch.randn(A_shape, dtype=dtype, device=device)
#     b = torch.randn(B_shape, dtype=dtype, device=device)
#     ref_out = bench.add(ref_a, ref_b)
#     res_out = bench.triton.add(a, b)
#     gems_assert_close(res_out, ref_out, dtype)
# ```
# """.strip() + "\n"
#         prompt += "The corresponding benchmark function you should write is:\n"
#         prompt += "```python\n"
#         prompt += """@label("add_benchmark")
# @parametrize("A_shape, B_shape", [((32,), (32,)), ((32, 32), (32, 32))])
# @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
# def add_benchmark(A_shape, B_shape, dtype):
#     import torch.utils.benchmark as benchmark
#     from sandbox.utils.accuracy_utils import CustomBenchmarkResult

#     quantiles = [0.5, 0.2, 0.8]

#     # input data
#     a = torch.randn(A_shape, dtype=dtype, device=device)
#     b = torch.randn(B_shape, dtype=dtype, device=device)

#     ms_triton, _, _ = triton.testing.do_bench(lambda: bench.triton.add(a, b), rep=100, quantiles=quantiles)
#     ms_torch, _, _ = triton.testing.do_bench(lambda: bench.add(a, b), rep=100, quantiles=quantiles)
    
#     speedup = ms_torch / ms_triton
#     result = CustomBenchmarkResult(
#         ref_time=ms_torch,
#         res_time=ms_triton, 
#         speedup=speedup, 
#     )
#     return result
# ```
# """
#         prompt += "\n"
#         prompt += f"Now please write the benchmark function for the following test function:\n"
#         prompt += "```python\n"
#         prompt += f"{info.test_func_code}\n"
#         prompt += "```\n"
#         prompt += f"Using @label and @parametrize decorators to cover all parameter combinations as in the test function rather than @triton.testing.perf_report or anything else.\n"
#         prompt += f"The @label name should be the original operator name appended with '_benchmark'.\n"
#         prompt += f"Do not include warm up cause triton.testing.do_bench already includes warm up.\n"
#         prompt += f"Do not print anything in the benchmark function.\n"
#         prompt += f"You must generate the valid benchmark function code directly without any explanations or additional text. The code must be complete and ready to run.\n"
#         prompt += f"You must use ```python ... ``` to format the code block.\n"
#         prompt += f"Like the example above, you must use @label to label the benchmark function with the original operator name appended with '_benchmark'. You must use @parametrize to cover all parameter combinations as in the test function. You must use torch.utils.benchmark to measure the execution time of both PyTorch and Triton kernels, and record the results in a CustomBenchmarkResult object.\n"
#         if info.user_advice:
#             prompt += f"And the following user advice should be considered: {info.user_advice}\n"
#         return prompt

    def generate_prompt_for_new(self, info: BenchmarkFuncGenerateArgs):
        # Role definition
        prompt = "You are a performance testing expert for PyTorch and Triton operators. "
        prompt += "Your task is to create a performance benchmark function based on a verified correctness test.\n\n"
        
        # Operator Information
        namespace = info.ops_namespace if info.ops_namespace else "aten"
        prompt += "## Operator Information\n"
        prompt += f"- Operator name: {info.kernel_name}\n"
        prompt += f"- Target API: torch.ops.{namespace}.{info.kernel_name}\n"
        
        if info.test_func_code:
             prompt += f"\n## Reference Correctness Test (Logic Verified)\n"
             prompt += "Use this code to understand the input arguments and shapes required:\n"
             prompt += "```python\n"
             prompt += info.test_func_code
             prompt += "\n```\n\n"

        # Benchmark Framework Definition
        prompt += "## Benchmarking Framework\n"
        prompt += "You must use the `GenericBenchmark` class. Here is its definition:\n"
        prompt += "```python\n"
        prompt += "class GenericBenchmark(Benchmark):\n"
        prompt += "    def __init__(self, *args, input_fn, **kwargs):\n"
        prompt += "        super().__init__(*args, **kwargs)\n"
        prompt += "        self.input_fn = input_fn\n\n"
        prompt += "    def set_more_shapes(self):\n"
        prompt += "        more_shapes_1d = [(2**28,)]\n"
        prompt += "        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 16)]\n"
        prompt += "        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 16)]\n"
        prompt += "        return more_shapes_1d + more_shapes_2d + more_shapes_3d\n\n"
        prompt += "    def get_input_iter(self, cur_dtype) -> Generator:\n"
        prompt += "        for shape in self.shapes:\n"
        prompt += "            yield from self.input_fn(shape, cur_dtype, self.device)\n"
        prompt += "```\n\n"

        # Target Format Example
        prompt += "## Target Format Example\n"
        prompt += "Your output must strictly follow this structure:\n"
        prompt += "```python\n"
        prompt += f'@label("{info.kernel_name}")\n'
        prompt += f"def {info.test_perf_func_name}():\n"
        prompt += "     # Define input generation logic matching the operator arguments\n"
        prompt += f"    def {info.kernel_name}_input_fn(shape, dtype, device):\n"
        prompt += "        # Generate and yield inputs as required by the operator\n"
        prompt += "        inp1 = torch.randn(shape, dtype=dtype, device=device)\n"
        prompt += "        inp2 = torch.randn(shape, dtype=dtype, device=device)\n"
        prompt += "        yield inp1, inp2\n\n"
        prompt += "        # You can also yield kwargs if needed\n"
        prompt += "        # yield inp1, inp2, {'arg_name': arg_value}\n\n"
        prompt += "    # Initialize benchmark\n"
        prompt += "    bench = GenericBenchmark(\n"
        prompt += f"        input_fn={info.kernel_name}_input_fn,\n"
        prompt += f'        op_name="{info.kernel_name}",\n'
        prompt += f"        torch_op=torch.ops.{namespace}.{info.kernel_name},\n"
        prompt += "        dtypes=[torch.float32, torch.float16],\n"
        prompt += "    )\n\n"
        prompt += "    return bench.run()\n"
        prompt += "```\n\n"

        # Requirements
        prompt += "## Requirements\n"
        prompt += f"1. Create a function named `{info.test_perf_func_name}` (usually `test_perf_{info.kernel_name}`).\n"
        prompt += f"2. Decorate it with `@label(\"{info.kernel_name}\")`.\n"
        prompt += "3. Define an `input_fn` inner function that yields inputs required by the operator.\n"
        prompt += "   - Construct input tensors using `torch.randn` or other appropriate torch functions.\n"
        prompt += "   - Ensure tensors are created with the correct `dtype` and `device` passed to `input_fn`.\n"
        prompt += "4. Instantiate `GenericBenchmark`.\n"
        prompt += f"   - `op_name`: \"{info.kernel_name}\"\n"
        prompt += f"   - `torch_op`: torch.ops.{namespace}.{info.kernel_name}\n"
        prompt += "    - `dtypes`: Explicitly define the list of dtypes to benchmark (e.g., `[torch.float32, torch.float16]`).\n"
        prompt += "5. Shape Customization:\n"
        prompt += "   - If the default shapes in `GenericBenchmark` are suitable, use `GenericBenchmark` directly as shown in the example.\n"
        prompt += "   - If the operator requires specific shapes (e.g., must be 3D, or power of 2) or logic, define a subclass inheriting from `GenericBenchmark` INSIDE the test function.\n"
        prompt += "   - In the subclass, override `set_more_shapes` to return your custom shape list, or `get_input_iter` if needed.\n"
        prompt += "6. Return `bench.run()`.\n"
        prompt += "7. Output ONLY the python code block.\n"
        
        if info.user_advice:
            prompt += f"\n## Additional User Guidance\n{info.user_advice}\n\n"

        return prompt
    
    def _init_data(self, kwargs):
        config = kwargs.dict()
        self.kernel_name = ""
        if "from_mcp" in config:
            self.from_mcp = config.pop("from_mcp")
            # self.kernel_name = config.pop("triton_kernel_name")
        # config["op_name"] = config.pop("test_func_name", None)
        if "check_result" in config and isinstance(config["check_result"], dict):
            from bench.sandbox.test.run_test import VerifyResult
            config["check_result"] = VerifyResult(**config["check_result"])
        config = BenchmarkFuncGenerateArgs(**config)
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
import kernelgenbench
from kernelgenbench.perfermance.performance_utils import GenericBenchmark
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch
""".strip() + "\n\n"
        else:
            test_func_prefix = "import torch\n\nimport pytest\n\nimport triton\n\n"
        for i in range(len(results)):
            results[i][0] = self.decouple_bench(results[i][0], self.kernel_name) if self.from_mcp else results[i][0]
            results[i][0] = test_func_prefix + results[i][0]
            results[i][0] = results[i][0].strip()
        return results

    def decouple_bench(self, code: str, kernel_name: str = "") -> str:
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
        
        return code

