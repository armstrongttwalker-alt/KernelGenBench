import json
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import torch
from logging import getLogger
from datetime import datetime

from generator.sampler.generate_samples import (
    TritonKernelGenerateArgs,
    GenerationConfig,
    InputArg,
    OutputArg,
)

logger = getLogger(__name__)


def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _placeholder(*args, **kwargs):
    raise NotImplementedError("This is a placeholder function.")


def get_torch_api_signature(torch_op_name: str, torch_op_func) -> Dict[str, Any]:
    """
    Extract signature information from a PyTorch API.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        Dictionary containing input_args, output_args, and func_desc
    """
    # Try to get the docstring for description
    func_desc = f"PyTorch operator: {torch_op_name}"
    if hasattr(torch_op_func, '__doc__') and torch_op_func.__doc__:
        # Extract first line of docstring as description
        doc_lines = torch_op_func.__doc__.strip()
        if doc_lines:
            func_desc = doc_lines
    
    # For now, we'll use generic input/output args since we don't have
    # detailed parameter information without runtime inspection
    # In a real implementation, you might want to use inspect module or
    # parse the docstring to extract parameter information
    
    # TODO : Improve argument extraction logic
    result = get_function_signature(torch_op_func)
    input_args = result["input_args"]
    # output_args = result["output_args"]
    
    # input_args = [InputArg(arg_name="args", arg_type="Any", arg_desc="Input arguments")]
    # output_args = [OutputArg(arg_type="Any", arg_desc="Output result")]
    
    return {
        "input_args": input_args,
        "output_args": None,
        "func_desc": func_desc,
    }


def create_triton_generate_args(torch_op_name: str, torch_op_func_or_namespace: Callable | str, impl_info) -> TritonKernelGenerateArgs:
    """
    Create TritonKernelGenerateArgs for a given PyTorch operator.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        TritonKernelGenerateArgs instance
    """
    # Extract the function name from the full path
    # e.g., 'torch.add' -> 'add', 'torch.nn.functional.gelu' -> 'gelu'
    kernel_name = torch_op_name.split('.')[-1]
    
    # Get signature information
    if isinstance(torch_op_func_or_namespace, str) and len(torch_op_func_or_namespace) > 0:
        # check torch.ops has the attribute
        if hasattr(torch.ops, torch_op_func_or_namespace):
            # torch_op_func actually is the namespace
            torch_op_name = f"{torch_op_func_or_namespace}::{kernel_name}"
            torch_op_namespace = getattr(torch.ops, torch_op_func_or_namespace)
            torch_op_func = getattr(torch_op_namespace, kernel_name)
            torch_op_func_name = f"{torch_op_namespace.__name__}.{kernel_name}"
    else:
        torch_op_func = torch_op_func_or_namespace
        torch_op_func_name = torch_op_name
    sig_info = get_torch_api_signature(torch_op_name, torch_op_func)
    
    # Create a simple torch kernel code snippet as reference
    torch_kernel_code = f"""
# Reference PyTorch implementation for {torch_op_name}
import torch

{kernel_name} = {torch_op_func_name}
""".strip()
    
    return TritonKernelGenerateArgs(
        triton_kernel_name=torch_op_name,
        func_desc=sig_info["func_desc"],
        torch_kernel_code=torch_kernel_code,
        input_args=sig_info["input_args"],
        # output_args=sig_info["output_args"],
        impl_info=impl_info,
        # func_type="other",  # Could be refined based on operator analysis
        from_mcp=False,
    )


def load_api_to_process_from_test_func_path(test_func_result_path: Path, get_success: bool = True) -> dict[str, str]:
    # check the path is dir or file
    if test_func_result_path.is_dir():
        dirs = [x for x in test_func_result_path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        api_info = {}
        for d in dirs:
            api_info.update(load_api_to_process_from_test_func_result(d, get_success))
        return api_info
    else:
        return load_api_to_process_from_test_func_result(test_func_result_path, get_success)

def load_api_to_process_from_test_func_result(test_func_result_path: Path, get_success: bool = True) -> dict[str, str]:
    """Load API information from a test function result file.

    Args:
        test_func_result_path: Path to the test function result file.
    Returns:
        A list of API information dictionaries.
    """
    test_func_result_path = test_func_result_path / "result.json" if test_func_result_path.name != "result.json" else test_func_result_path
    if not test_func_result_path.exists():
        logger.warning(f"Test function result path {test_func_result_path} does not exist.")
        return {}
    with open(test_func_result_path, "r") as f:
        test_func_results = json.load(f)

    api_info = {}
    for result in test_func_results:
        api_name = result["op_name"]
        ok = result["success"] == get_success
        if ok:
            if "::" in api_name:
                namespace, name = api_name.split("::", 1)
                api_info[name] = namespace
            else:
                logger.warning(f"API name {api_name} does not contain namespace info, defaulting to 'aten'")
                api_info[api_name] = "aten"
    return api_info


def load_right_test_function_from_test_func_dir(path: Path, get_success: bool = True) -> Dict[str, str]:
    # check the path is dir or file
    if path.is_dir():
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        test_funcs = {}
        for d in dirs:
            test_funcs.update(load_right_test_function_from_result_path(d / "result.json", get_success))
        return test_funcs
    else:
        return load_right_test_function_from_result_path(path, get_success)


def load_right_test_function_from_result_path(path: Path, get_success: bool = True) -> Dict[str, str]:
    path = path / "result.json" if path.name != "result.json" else path
    if not path.exists():
        logger.warning(f"Test function result path {path} does not exist.")
        return {}
    with open(path, "r") as f:
        eval_result = json.load(f)
    test_funcs = {}
    for item in eval_result:
        if item["success"] == get_success:
            test_funcs[item["op_name"]] = item["test_func"]
    return test_funcs


def convert_accuracy_to_performance_test(test_funcs: Dict[str, str]) -> Dict[str, str]:
    """Convert accuracy test functions to performance test functions.

    This function transforms accuracy test functions into performance test functions
    by applying predefined conversion rules.

    Args:
        test_funcs: Dictionary mapping operator names to accuracy test function code.
                   Format: {op_name: test_func_code}

    Returns:
        Dictionary mapping operator names to performance test function code.
        Format: {op_name: perf_test_func_code}
    """
    result = {}

    for op_name, test_func_code in test_funcs.items():
        try:
            converted_code = _convert_single_test_func(test_func_code)
            result[op_name] = converted_code
        except Exception as e:
            logger.error(f"Failed to convert test function for {op_name}: {e}")
            # Keep original code if conversion fails
            result[op_name] = test_func_code

    return result


def _convert_single_test_func(code: str) -> str:
    """Convert a single accuracy test function to performance test function.

    The code may contain multiple function definitions.
    """
    # Step 1: Modify @label decorators
    # code = re.sub(
    #     r'@label\(["\']([^"\']+)["\']\)',
    #     r'@label("\1_benchmark")',
    #     code
    # )

    # Step 2: Modify function names (test_xxx -> xxx_benchmark)
    # Match function definitions and extract the label name for better naming
    def replace_func_name(match):
        func_name = match.group(1)
        # Remove 'test_' prefix if exists
        if func_name.startswith('test_'):
            func_name = func_name[5:]  # Remove 'test_'
        # Remove common suffixes like '_tensor', '_out' temporarily
        # We'll add '_benchmark' before these suffixes
        for suffix in ['_tensor', '_out', '_scalar']:
            if func_name.endswith(suffix):
                base_name = func_name[:-len(suffix)]
                return f'def {base_name}_benchmark{suffix}('
        return f'def {func_name}_benchmark('

    code = re.sub(
        r'def\s+(test_\w+)\s*\(',
        replace_func_name,
        code
    )

    # Step 3: Add imports and initialization at the beginning of each function
    # Find all function definitions and add imports after the function signature
    lines = code.split('\n')
    new_lines = []
    in_function = False
    added_imports = False

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Detect function definition
        if re.match(r'def\s+\w+_benchmark', line):
            in_function = True
            added_imports = False

        # Add imports after function signature (after the line with ':')
        if in_function and not added_imports and line.strip().endswith(':'):
            # Add imports and quantiles
            imports = '''    import torch.utils.benchmark as benchmark
    from sandbox.utils.accuracy_utils import CustomBenchmarkResult
    import triton

    quantiles = [0.5, 0.2, 0.8]
'''
            new_lines.append(imports)
            added_imports = True
            in_function = False

    code = '\n'.join(new_lines)

    # Step 4: Transform test logic
    code = _transform_test_logic(code)

    return code


def _transform_test_logic(code: str) -> str:
    """Transform the test logic from accuracy checking to performance measurement."""
    lines = code.split('\n')
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Transform reference implementation calls (with assignment)
        # Pattern: ref_out = torch.ops.aten.xxx(...) or ref_out = torch.ops.aten.xxx.out(...)
        ref_call_match = re.search(r'(\s*)(\w+)\s*=\s*(torch\.ops\.\w+\.\w+(?:\.\w+)?)\((.*)\)', line)
        if ref_call_match and ref_call_match.group(2).startswith('ref_'):
            indent = ref_call_match.group(1)

            # Collect multi-line arguments if needed
            full_line = line
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while paren_count > 0 and j < len(lines):
                full_line += '\n' + lines[j]
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1

            # Extract the complete function call
            call_match = re.search(r'(\w+)\s*=\s*(torch\.ops\.\w+\.\w+(?:\.\w+)?\(.*?\))\s*$', full_line, re.DOTALL)
            if call_match:
                complete_call = call_match.group(2)

                # Generate performance test for PyTorch reference
                new_lines.append(f'{indent}# PyTorch reference implementation')
                new_lines.append(f'{indent}ms_torch, _, _ = triton.testing.do_bench(')
                new_lines.append(f'{indent}    lambda: {complete_call},')
                new_lines.append(f'{indent}    rep=100,')
                new_lines.append(f'{indent}    quantiles=quantiles')
                new_lines.append(f'{indent})')
                i = j
                continue
            else:
                # If extraction failed, keep the line and continue
                new_lines.append(line)
                i += 1
                continue

        # Transform reference implementation calls (without assignment)
        # Pattern: torch.ops.aten.xxx(...) where arguments contain ref_ variables
        ref_no_assign_match = re.search(r'(\s*)(torch\.ops\.\w+\.\w+(?:\.\w+)?)\((.*)\)', line)
        if ref_no_assign_match and 'ref_' in line and '=' not in line.split('torch.ops')[0]:
            indent = ref_no_assign_match.group(1)

            # Collect multi-line arguments if needed
            full_line = line
            paren_count = line.count('(') - line.count(')')
            j = i + 1
            while paren_count > 0 and j < len(lines):
                full_line += '\n' + lines[j]
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1

            # Extract the complete function call
            call_match = re.search(r'(torch\.ops\.\w+\.\w+(?:\.\w+)?\(.*?\))', full_line, re.DOTALL)
            if call_match:
                complete_call = call_match.group(1)

                # Generate performance test for PyTorch reference
                new_lines.append(f'{indent}# PyTorch reference implementation')
                new_lines.append(f'{indent}ms_torch, _, _ = triton.testing.do_bench(')
                new_lines.append(f'{indent}    lambda: {complete_call},')
                new_lines.append(f'{indent}    rep=100,')
                new_lines.append(f'{indent}    quantiles=quantiles')
                new_lines.append(f'{indent})')
                i = j
                continue
            else:
                # If extraction failed, keep the line and continue
                new_lines.append(line)
                i += 1
                continue

        # Transform Triton implementation calls with flagbench.use_gems
        # Pattern: with flagbench.use_gems(REGISTERED_OPS):
        #              act_out = torch.ops.aten.xxx(...)
        if 'with flagbench.use_gems' in line:
            indent = re.match(r'(\s*)', line).group(1)
            new_lines.append('')
            new_lines.append(f'{indent}# Triton implementation')
            new_lines.append(line)  # Keep the 'with' statement

            # Process the block inside 'with'
            i += 1
            while i < len(lines):
                inner_line = lines[i]
                # Check if we're still inside the with block
                if inner_line.strip() and not inner_line.startswith(indent + '    '):
                    break

                # Transform the actual call
                # Support .out suffix: torch.ops.aten.xxx.out(...)
                act_call_match = re.search(r'(\s*)(\w+)\s*=\s*(torch\.ops\.\w+\.\w+(?:\.\w+)?)\((.*)\)', inner_line)
                if act_call_match and act_call_match.group(2).startswith('act_'):
                    inner_indent = act_call_match.group(1)

                    # Collect multi-line arguments
                    full_inner_line = inner_line
                    paren_count = inner_line.count('(') - inner_line.count(')')
                    j = i + 1
                    while paren_count > 0 and j < len(lines):
                        full_inner_line += '\n' + lines[j]
                        paren_count += lines[j].count('(') - lines[j].count(')')
                        j += 1

                    # Extract complete call
                    call_match = re.search(r'\w+\s*=\s*(torch\.ops\.\w+\.\w+(?:\.\w+)?\(.*?\))\s*$', full_inner_line, re.DOTALL)
                    if call_match:
                        complete_call = call_match.group(1)

                        new_lines.append(f'{inner_indent}ms_triton, _, _ = triton.testing.do_bench(')
                        new_lines.append(f'{inner_indent}    lambda: {complete_call},')
                        new_lines.append(f'{inner_indent}    rep=100,')
                        new_lines.append(f'{inner_indent}    quantiles=quantiles')
                        new_lines.append(f'{inner_indent})')
                        i = j
                        break
                    else:
                        # If extraction failed, keep the line and continue
                        new_lines.append(inner_line)
                        i += 1
                else:
                    # Check for calls without assignment (e.g., torch.ops.aten.xxx.out(...))
                    act_no_assign_match = re.search(r'(\s*)(torch\.ops\.\w+\.\w+(?:\.\w+)?)\((.*)\)', inner_line)
                    if act_no_assign_match and '=' not in inner_line.split('torch.ops')[0]:
                        inner_indent = act_no_assign_match.group(1)

                        # Collect multi-line arguments
                        full_inner_line = inner_line
                        paren_count = inner_line.count('(') - inner_line.count(')')
                        j = i + 1
                        while paren_count > 0 and j < len(lines):
                            full_inner_line += '\n' + lines[j]
                            paren_count += lines[j].count('(') - lines[j].count(')')
                            j += 1

                        # Extract complete call
                        call_match = re.search(r'(torch\.ops\.\w+\.\w+(?:\.\w+)?\(.*?\))', full_inner_line, re.DOTALL)
                        if call_match:
                            complete_call = call_match.group(1)

                            new_lines.append(f'{inner_indent}ms_triton, _, _ = triton.testing.do_bench(')
                            new_lines.append(f'{inner_indent}    lambda: {complete_call},')
                            new_lines.append(f'{inner_indent}    rep=100,')
                            new_lines.append(f'{inner_indent}    quantiles=quantiles')
                            new_lines.append(f'{inner_indent})')
                            i = j
                            break
                        else:
                            # If extraction failed, keep the line and continue
                            new_lines.append(inner_line)
                            i += 1
                    else:
                        # Keep all other lines (including clone operations)
                        new_lines.append(inner_line)
                        i += 1
            continue

        # Remove assert_close calls and add return statement
        # But skip import statements
        if ('assert_close' in line or 'gems_assert_close' in line) and \
           not line.strip().startswith('import') and \
           not line.strip().startswith('from'):
            # Find the indentation of the assert line
            indent = re.match(r'(\s*)', line).group(1)

            # Skip this line and check if it spans multiple lines
            paren_count = line.count('(') - line.count(')')
            while paren_count > 0 and i + 1 < len(lines):
                i += 1
                paren_count += lines[i].count('(') - lines[i].count(')')
            i += 1

            # After removing assert, add return statement
            new_lines.append('')
            new_lines.append(f'{indent}# Calculate speedup and return result')
            new_lines.append(f'{indent}speedup = ms_torch / ms_triton')
            new_lines.append(f'{indent}result = CustomBenchmarkResult(')
            new_lines.append(f'{indent}    ref_time=ms_torch,')
            new_lines.append(f'{indent}    res_time=ms_triton,')
            new_lines.append(f'{indent}    speedup=speedup,')
            new_lines.append(f'{indent})')
            new_lines.append(f'{indent}return result')
            continue

        new_lines.append(line)
        i += 1

    return '\n'.join(new_lines)


def load_right_kernel_code_from_acc_verify_dir(path: Path, get_success: bool = True) -> Dict[str, str]:
    # check the path is dir or file
    # breakpoint()
    if path.is_dir():
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.startswith("log_")]
        kernel_code = {}
        for d in dirs:
            kernel_code.update(load_right_kernel_code_from_acc_verify_result_path(d / "result.json", get_success))
        return kernel_code
    else:
        return load_right_kernel_code_from_acc_verify_result_path(path, get_success)


def load_right_kernel_code_from_acc_verify_result_path(path: Path, get_success: bool = True) -> Dict[str, str]:
    path = path / "result.json" if path.name != "result.json" else path
    if not path.exists():
        logger.warning(f"Test function result path {path} does not exist.")
        return {}
    with open(path, "r") as f:
        eval_result = json.load(f)
    kernel_code = {}
    for item in eval_result:
        if item["success"] == get_success:
            kernel_code[item["op_name"]] = item["code"]
    return kernel_code


def get_function_signature(func: Callable) -> Dict[str, Any]:
    """
    获取函数的入参和出参信息。
    
    支持两种类型：
    1. 普通 Python 函数：使用 inspect 模块
    2. torch.ops 算子：使用 _schema 属性
    
    Args:
        func: 可调用对象
        
    Returns:
        包含入参和出参信息的字典，格式：
        {
            "input_args": [InputArg, ...],
            "output_args": [OutputArg, ...],
            "signature": "完整的函数签名字符串"
        }
    """
    result = {
        "input_args": [],
        "output_args": [],
        "signature": ""
    }
    
    # 情况1: torch.ops 算子，使用 _schemas 属性
    if hasattr(func, '_schemas'):
        schema = func._schemas
        result["signature"] = str(schema)
        
        input_output_info = {k: str(v) for k, v in schema.items()}
        return {"input_args": input_output_info}
        # # 解析 schema
        # # 格式示例: "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        # schema_str = str(schema)
        
        # # 提取输入参数部分
        # input_match = re.search(r'\((.*?)\)', schema_str)
        # if input_match:
        #     params_str = input_match.group(1)
            
        #     # 分割参数（处理 * 分隔符）
        #     params_parts = params_str.split('*')
        #     all_params = []
            
        #     for part in params_parts:
        #         for param in part.split(','):
        #             param = param.strip()
        #             if param and param != '*':
        #                 all_params.append(param)
            
        #     # 解析每个参数
        #     for param in all_params:
        #         param = param.strip()
        #         if not param:
        #             continue
                
        #         # 解析参数：格式 "Type name" 或 "Type name=default"
        #         has_default = '=' in param
        #         if has_default:
        #             param_no_default, default_value = param.split('=', 1)
        #         else:
        #             param_no_default = param
        #             default_value = None
                
        #         parts = param_no_default.strip().split()
        #         if len(parts) >= 2:
        #             param_type = parts[0]
        #             param_name = parts[1]
                    
        #             # 处理类型中的修饰符 (如 Tensor(a!))
        #             param_type = re.sub(r'\(.*?\)', '', param_type)
                    
        #             # 创建 InputArg 对象
        #             input_arg = InputArg(
        #                 arg_name=param_name,
        #                 arg_type=param_type,
        #                 arg_value=None,
        #                 arg_default=default_value.strip() if default_value else None,
        #                 arg_desc=""
        #             )
        #             result["input_args"].append(input_arg)
        
        # # 提取输出参数
        # output_match = re.search(r'->\s*(.+)$', schema_str)
        # if output_match:
        #     output_type = output_match.group(1).strip()
            
        #     # 处理多个返回值 (如 "(Tensor, Tensor)")
        #     if output_type.startswith('(') and output_type.endswith(')'):
        #         output_types = [t.strip() for t in output_type[1:-1].split(',')]
        #         for out_type in output_types:
        #             # 清理类型修饰符
        #             out_type = re.sub(r'\(.*?\)', '', out_type)
        #             output_arg = OutputArg(
        #                 arg_type=out_type,
        #                 arg_value=None,
        #                 arg_desc=""
        #             )
        #             result["output_args"].append(output_arg)
        #     else:
        #         # 单个返回值
        #         output_type = re.sub(r'\(.*?\)', '', output_type)
        #         output_arg = OutputArg(
        #             arg_type=output_type,
        #             arg_value=None,
        #             arg_desc=""
        #         )
        #         result["output_args"].append(output_arg)
    
    # 情况2: 普通 Python 函数，使用 inspect
    else:
        try:
            sig = inspect.signature(func)
            result["signature"] = str(sig)
            
            # 解析输入参数
            for param_name, param in sig.parameters.items():
                # 获取类型注解
                param_type = (
                    str(param.annotation) 
                    if param.annotation != inspect.Parameter.empty 
                    else "Any"
                )
                
                # 清理类型字符串（移除 <class '...'> 包装）
                if param_type.startswith("<class '") and param_type.endswith("'>"):
                    param_type = param_type[8:-2]
                
                # 获取默认值
                param_default = (
                    param.default 
                    if param.default != inspect.Parameter.empty 
                    else None
                )
                
                # 创建 InputArg 对象
                input_arg = InputArg(
                    arg_name=param_name,
                    arg_type=param_type,
                    arg_value=None,
                    arg_default=param_default,
                    arg_desc=""
                )
                result["input_args"].append(input_arg)
            
            # 解析返回类型
            if sig.return_annotation != inspect.Signature.empty:
                return_type = str(sig.return_annotation)
                
                # 清理类型字符串
                if return_type.startswith("<class '") and return_type.endswith("'>"):
                    return_type = return_type[8:-2]
                
                # 处理多返回值类型 (如 Tuple[int, str])
                # 简单处理：如果是 Tuple/tuple，就作为单个返回类型
                output_arg = OutputArg(
                    arg_type=return_type,
                    arg_value=None,
                    arg_desc=""
                )
                result["output_args"].append(output_arg)
            else:
                # 没有类型注解时
                output_arg = OutputArg(
                    arg_type="Any",
                    arg_value=None,
                    arg_desc=""
                )
                result["output_args"].append(output_arg)
                
        except (ValueError, TypeError) as e:
            # 无法获取签名时，返回空结果
            result["signature"] = f"<Unable to get signature: {e}>"
    
    return result


def get_function_signature_simple(func: Callable) -> tuple[List[InputArg], List[OutputArg]]:
    """
    简化版本：直接返回 InputArg 和 OutputArg 列表。
    
    Args:
        func: 可调用对象
        
    Returns:
        (InputArg 列表, OutputArg 列表)
    """
    sig_info = get_function_signature(func)
    return sig_info["input_args"], sig_info["output_args"]


def query_operator_wiki(operator_name: str) -> str:
    """Query DeepWiki for information about a given operator.

    Args:
        operator_name (str): The name of the operator to query.
    Returns:
        str: The information retrieved from DeepWiki about the operator.
    """
    import requests

    DEEPWIKI_API_URL = "http://120.92.108.161/chat/completions/stream"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "repo_url": "https://github.com/triton/triton",
        "type": "github",
        "messages": [{
            "role": "user",
            "content": (
                f"请仅返回一个 JSON 字符串，该字符串是一个列表，形如 "
                f'[{{\"link\": \"算子文件路径\", \"code\": \"代码\"}}, ...]。'
                f"每个对象必须包含 link（算子文件路径）和 code（算子代码，无省略、无额外注释）。"
                f"总结果最多返回 5 条，优先返回 python 和 triton 算子，如果没有的话可以返回其他算子（比如 .h 文件）。"
                f"不要返回任何解释、前后缀、Markdown 或自然语言。"
                f"代码内容禁止包含行号、'第几行'、'36-63' 等描述，更不要出现额外括号或非 JSON 符号。"
                f"相关任务要求是：用 triton 实现 {operator_name} 这个算子。"
            ), 
        }],
        "provider": "openai",
        "model": "gpt-4.1",
        "language": "zh",
    }
    response = requests.post(DEEPWIKI_API_URL, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data