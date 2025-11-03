import ast
import importlib.util
import sys
from typing import List, Dict

import os
import types

def scan_registrations(file_path: str) -> List[Dict]:
    """
    扫描Python文件并提取所有@register装饰的函数信息
    返回格式：[{"name": 注册名, "func_name": 函数名, "args": 装饰器参数}, ...]
    """
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    registrations = []
    
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
            
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
                
            if getattr(decorator.func, 'id', None) == 'register':
                # 解析装饰器参数
                args = []
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    elif isinstance(arg, ast.Name):
                        args.append(arg.id)
                
                registrations.append({
                    "name": args[0] if args else node.name,
                    "func_name": node.name,
                    "args": args[1:] if len(args) > 1 else []
                })
    
    return registrations

def auto_register_module(module_path_or_source: str):
    # """
    # 自动注册模块中所有带@register装饰的函数
    # """
    # # 动态导入模块
    # spec = importlib.util.spec_from_file_location("temp_module", module_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules["temp_module"] = module
    # spec.loader.exec_module(module)
    """
    自动注册模块中所有带 @register 装饰的函数。
    支持传入模块的文件路径或源码字符串。
    """
    module_name = "temp_module"
    if os.path.isfile(module_path_or_source):
        # 从文件路径导入模块
        spec = importlib.util.spec_from_file_location(module_name, module_path_or_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    else:
        # 处理源码字符串：创建一个新的 module 对象并执行源码
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        exec(module_path_or_source, module.__dict__)
    
    # 扫描注册信息
    # registrations = scan_registrations(module_path)
    
    # 执行注册
    # for reg in registrations:
    #     func = getattr(module, reg['func_name'])
    #     register(reg['name'], *reg['args'])(func)
    #     print(f"Registered: {reg['name']} -> {reg['func_name']}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python file path to scan")
    args = parser.parse_args()
    
    auto_register_module(args.file)