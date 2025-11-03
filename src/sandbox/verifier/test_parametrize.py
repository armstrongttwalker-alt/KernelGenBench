# import itertools

# _parameter_registry = {}

# def parametrize(param_names, values):
#     def decorator(func):
#         name = func.__name__
#         if name not in _parameter_registry:
#             _parameter_registry[name] = {}

#         # 解析 param_names，支持 "a" 或 "a,b,c"
#         if isinstance(param_names, str):
#             param_names_list = [p.strip() for p in param_names.split(",")]
#         else:
#             raise ValueError("param_names must be a string")

#         # 如果只有一个参数名，直接存对应的 values
#         if len(param_names_list) == 1:
#             _parameter_registry[name][param_names_list[0]] = values
#         else:
#             # 多参数名，values 应该是可迭代的元组列表
#             # 这里将 values 按位置拆开，存成对应参数名的列表
#             transposed = list(zip(*values))  # 将 [(a1,b1,c1), (a2,b2,c2)] 转成 [(a1,a2), (b1,b2), (c1,c2)]
#             for i, param in enumerate(param_names_list):
#                 _parameter_registry[name][param] = list(transposed[i])

#         return func
#     return decorator

# def get_registry():
#     return _parameter_registry

# label_registry.py
_label_registry = {}

def label(name):
    """给测试函数加上一个标签"""
    def decorator(func):
        # breakpoint()
        _label_registry.setdefault(name, []).append((func, "default"))
        return func
    return decorator

def get_funcs_by_label(name):
    """根据标签获取所有函数"""
    return _label_registry.get(name, [])

def get_all_labels():
    return list(_label_registry.keys())


_parameter_registry = {}

class Param:
    def __init__(self, *values, marks: str | list[str] = None):
        self.values = [values]
        self.marks = marks if isinstance(marks, list) else [marks]

    def __repr__(self) -> str:
        return f"Param(values={self.values}, marks={self.marks})"

def parametrize(param_names, values):
    def decorator(func):
        name = func.__name__
        if name not in _parameter_registry:
            _parameter_registry[name] = {}

        if isinstance(param_names, str):
            param_names_list = [p.strip() for p in param_names.split(",")]
        else:
            raise ValueError("param_names must be a string")

        if isinstance(values[0], Param):
            for param in values:
                for mark in param.marks:
                    _parameter_registry[name].setdefault(mark, []).append(
                        {
                            "names": param_names_list,
                            "values": param.values
                        }
                    )
                    _label_registry.setdefault(mark, []).append((func, mark))
        else:
            # slot：一组参数名和对应的值
            # _parameter_registry[name].append({
            #     "names": param_names_list,
            #     "values": values
            # })
            _parameter_registry[name].setdefault("default", []).append({
                "names": param_names_list,
                "values": values
            })

        return func
    return decorator

def get_registry():
    return _parameter_registry

def get_params(name, mark="default"):
    return _parameter_registry.get(name, {}).get(mark, {})
