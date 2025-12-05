import json
import ijson
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import FunctionSchema
from torch._ops import _OpNamespace
from dataclasses import dataclass

@dataclass
class APIInfo:
    namespace: str
    api: str
    schemas: Dict[str, str | FunctionSchema]
    to_str: bool = True

    def __post_init__(self):
        if self.to_str:
            if self.schemas and isinstance(next(iter(self.schemas.values())), FunctionSchema):
                self.schemas = {k: str(v) for k, v in self.schemas.items()}

class TorchOpsLoader:
    def __init__(self, to_str: bool = True):
        self._cache = {}
        self.to_str = to_str

    def load_namespace(self, namespace: str) -> Dict[str, APIInfo]:
        assert namespace in dir(torch.ops), f"Namespace {namespace} not found in torch.ops"
        assert isinstance(getattr(torch.ops, namespace), _OpNamespace), f"{namespace} is not a valid OpNamespace"
        if namespace in self._cache:
            return self._cache[namespace]
        ns_module = getattr(torch.ops, namespace)
        ops_dict = {}
        for op_name in dir(ns_module):
            op = getattr(ns_module, op_name)
            if callable(op):
                # ops_dict[op_name] = {"schemas": op._schemas, "api": op.__module__ + "." + op.__name__}
                ops_dict[op_name] = APIInfo(
                    api=op.__module__ + "." + op.__name__, 
                    schemas=op._schemas, 
                    namespace=namespace, 
                    to_str=self.to_str
                )
        self._cache[namespace] = ops_dict
        return ops_dict
    
    def get_operator(self, namespace: str, op_name: str) -> APIInfo:
        if namespace == "":
            namespace = "aten"
        ns_data = self.load_namespace(namespace)
        assert op_name in ns_data, f"Operator {op_name} not found in namespace {namespace}"
        return ns_data[op_name]
    
    def list_namespaces(self) -> List[str]:
        return [ns for ns in dir(torch.ops) if isinstance(getattr(torch.ops, ns), _OpNamespace)]
    
    def load_all(self) -> Dict[str, Dict[str, APIInfo]]:
        all_data = {}
        for namespace in self.list_namespaces():
            all_data[namespace] = self.load_namespace(namespace)
        return all_data

class OperatorLoader:
    """专门用于加载 all_operators.json 的工具类"""
    
    def __init__(self, json_path: str = "src/flagbench/dataset/all_operators.json"):
        self.json_path = Path(json_path)
        self._cache = {}
    
    def load_namespace(self, namespace: str) -> Dict:
        """加载特定命名空间（如 'aten', 'cudnn'）"""
        if namespace in self._cache:
            return self._cache[namespace]
        
        with open(self.json_path, 'rb') as f:
            parser = ijson.kvitems(f, '')
            for key, value in parser:
                if key == namespace:
                    self._cache[key] = value
                    return value
        
        return {}
    
    def get_operator(self, namespace: str, op_name: str) -> List[Dict]:
        """获取特定算子的所有变体"""
        ns_data = self.load_namespace(namespace)
        return ns_data.get(op_name, [])
    
    def list_namespaces(self) -> List[str]:
        """列出所有命名空间"""
        with open(self.json_path, 'rb') as f:
            parser = ijson.kvitems(f, '')
            return [key for key, _ in parser]
    
    def load_all(self, merge: bool = False) -> Dict[str, Dict[str, List[Dict]]]:
        """加载完整文件（如果内存足够）"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        if merge:
            merged = {}
            for namespace, ops in data.items():
                merged.update(ops)
            return merged        
        return data


if __name__ == "__main__":
    loader = TorchOpsLoader()
    aten_ops = loader.load_namespace('aten')
    print(f"Loaded {len(aten_ops)} aten operators.")
    print(aten_ops['abs'])