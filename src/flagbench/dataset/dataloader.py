import json
import ijson
from pathlib import Path
from typing import Dict, List, Optional

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
    
    def load_all(self) -> Dict[str, Dict[str, List[Dict]]]:
        """加载完整文件（如果内存足够）"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data


if __name__ == "__main__":
    loader = OperatorLoader()
    
    # 方式1: 只加载需要的命名空间
    aten_ops = loader.load_namespace('aten')
    print(f"Loaded {len(aten_ops)} aten operators")
    
    # 方式2: 获取特定算子
    batch_norm_ops = loader.get_operator('aten', '_batch_norm_with_update')
    print(f"Found {len(batch_norm_ops)} variants")
    
    # 方式3: 列出所有命名空间
    namespaces = loader.list_namespaces()
    print(f"Available namespaces: {namespaces}")