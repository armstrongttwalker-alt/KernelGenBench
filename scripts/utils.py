import json
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import torch
from datetime import datetime


@dataclass
class InputArg:
    arg_name: str
    arg_type: str
    arg_value: Any = None
    arg_default: Any = None
    arg_desc: str = ""


@dataclass
class OutputArg:
    arg_type: str
    arg_value: Any = None
    arg_desc: str = ""


def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_api_to_process_from_test_func_result(test_func_result_path: Path) -> dict[str, str]:
    """Load API information from a test function result file.

    Args:
        test_func_result_path: Path to the test function result file.
    Returns:
        A list of API information dictionaries.
    """
    with open(test_func_result_path, "r") as f:
        test_func_results = json.load(f)

    api_info = {}
    for result in test_func_results:
        api_name = result["op_name"]
        success = result["success"]
        if success:
            if "::" in api_name:
                namespace, name = api_name.split("::", 1)
            api_info[name] = namespace
    return api_info


def load_right_test_function_from_result_path(path: Path) -> Dict[str, str]:
    with open(path, "r") as f:
        eval_result = json.load(f)
    test_funcs = {}
    for item in eval_result:
        if item["success"]:
            test_funcs[item["op_name"]] = item["test_func"]
    return test_funcs


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
        
        # 解析 schema
        # 格式示例: "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        schema_str = str(schema)
        
        # 提取输入参数部分
        input_match = re.search(r'\((.*?)\)', schema_str)
        if input_match:
            params_str = input_match.group(1)
            
            # 分割参数（处理 * 分隔符）
            params_parts = params_str.split('*')
            all_params = []
            
            for part in params_parts:
                for param in part.split(','):
                    param = param.strip()
                    if param and param != '*':
                        all_params.append(param)
            
            # 解析每个参数
            for param in all_params:
                param = param.strip()
                if not param:
                    continue
                
                # 解析参数：格式 "Type name" 或 "Type name=default"
                has_default = '=' in param
                if has_default:
                    param_no_default, default_value = param.split('=', 1)
                else:
                    param_no_default = param
                    default_value = None
                
                parts = param_no_default.strip().split()
                if len(parts) >= 2:
                    param_type = parts[0]
                    param_name = parts[1]
                    
                    # 处理类型中的修饰符 (如 Tensor(a!))
                    param_type = re.sub(r'\(.*?\)', '', param_type)
                    
                    # 创建 InputArg 对象
                    input_arg = InputArg(
                        arg_name=param_name,
                        arg_type=param_type,
                        arg_value=None,
                        arg_default=default_value.strip() if default_value else None,
                        arg_desc=""
                    )
                    result["input_args"].append(input_arg)
        
        # 提取输出参数
        output_match = re.search(r'->\s*(.+)$', schema_str)
        if output_match:
            output_type = output_match.group(1).strip()
            
            # 处理多个返回值 (如 "(Tensor, Tensor)")
            if output_type.startswith('(') and output_type.endswith(')'):
                output_types = [t.strip() for t in output_type[1:-1].split(',')]
                for out_type in output_types:
                    # 清理类型修饰符
                    out_type = re.sub(r'\(.*?\)', '', out_type)
                    output_arg = OutputArg(
                        arg_type=out_type,
                        arg_value=None,
                        arg_desc=""
                    )
                    result["output_args"].append(output_arg)
            else:
                # 单个返回值
                output_type = re.sub(r'\(.*?\)', '', output_type)
                output_arg = OutputArg(
                    arg_type=output_type,
                    arg_value=None,
                    arg_desc=""
                )
                result["output_args"].append(output_arg)
    
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