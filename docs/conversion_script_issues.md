# 转换脚本问题分析与修复方案

## 概述

本文档记录了 `scripts/convert_flaggems_tests.py` 在转换 FlagGems 测试到 flagbench 格式时遇到的问题，以及建议的修复方案。

## 发现的问题

### 问题 1: 常量定义不完整（最严重）

**位置**: `TestConverter.generate_imports()` 方法，lines 295-323

**问题描述**:
- 脚本只包含硬编码的 7 个常量定义（DIMS_LIST, DIM_LIST, KEEPDIM_DIMS_SHAPE, KIND_KEEPDIM_DIMS_SHAPE, KEEPDIM_DIMS, KEEPDIM_DIM, EMPTY_SHAPES）
- 无法动态检测测试中使用的其他常量
- 导致转换后的测试文件缺少必要的常量定义

**遇到的缺失常量**:
- `MNK_SHAPES` - 用于矩阵乘法相关测试
- `CAT_SHAPES` - 用于 cat 操作测试
- `CUMSUM_SHAPES` - 用于 cumsum 测试
- 可能还有更多...

**当前代码**:
```python
constant_definitions = {
    'DIMS_LIST': 'DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]',
    'DIM_LIST': 'DIM_LIST = [1] if QUICK_MODE else [0, 1]',
    # ... 只有 7 个硬编码的常量
}
```

**问题根源**:
脚本没有从 FlagGems 源文件中动态提取模块级别的常量定义，只依赖硬编码的列表。

**错误示例**:
```
NameError: name 'MNK_SHAPES' is not defined
```

---

### 问题 2: 缺少辅助函数提取

**问题描述**:
- 脚本只提取测试函数（以 `test_` 开头的函数）
- 不提取辅助函数（helper functions）
- FlagGems 测试文件中定义的辅助函数会丢失

**遇到的缺失函数**:
- `gen_cat_shapes_dim()` - 生成 cat 操作的测试形状和维度组合
- 可能还有其他辅助函数...

**当前代码逻辑**:
```python
def extract_test_functions(self, tree: ast.Module, source_file: Path) -> List[TestFunction]:
    """Extract all test functions from AST."""
    test_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            # 只提取 test_ 开头的函数
            ...
```

**影响**:
转换后的测试文件会因为缺少辅助函数而无法运行。

**错误示例**:
```
NameError: name 'gen_cat_shapes_dim' is not defined
```

---

### 问题 3: flag_gems 转换不当

**位置**: `TestConverter._convert_code_body()` 方法，lines 225-240

**问题描述**:
脚本使用简单的字符串替换来转换 `flag_gems` 引用，导致不正确的转换。

**当前转换逻辑**:
```python
def _convert_code_body(self, code: str) -> str:
    # Replace flag_gems.device with device
    code = re.sub(r'\bflag_gems\.device\b', 'device', code)

    # Replace flag_gems.use_gems() with flagbench.use_gems(REGISTERED_OPS)
    code = re.sub(
        r'with\s+flag_gems\.use_gems\(\)\s*:',
        'with flagbench.use_gems(REGISTERED_OPS):',
        code
    )

    # Replace import flag_gems with import flagbench
    code = re.sub(r'\bimport\s+flag_gems\b', 'import flagbench', code)

    return code
```

**导致的问题**:

1. **vendor_name 转换错误**:
   - `flag_gems.vendor_name` → `flagbench.vendor_name`
   - flagbench 可能没有 `vendor_name` 属性，或行为不同

2. **vendor-specific 导入错误**:
   - `from flag_gems.runtime.backend._kunlunxin import ops as kl_ops`
   - 转换为 `from flagbench.runtime.backend._kunlunxin import ops as kl_ops`
   - flagbench 没有这个模块路径

3. **缺少特殊处理**:
   - 没有处理 vendor-specific 的代码逻辑
   - 没有处理 flag_gems 特有的 API

**错误示例**:
```python
# 转换后的错误代码
if flagbench.vendor_name == "mthreads":  # flagbench 可能没有 vendor_name
    ...

from flagbench.runtime.backend._kunlunxin import ops as kl_ops  # 模块不存在
```

---

### 问题 4: 缺少模块级代码提取

**问题描述**:
- 脚本只提取函数定义（`ast.FunctionDef`）
- 不提取模块级的常量、辅助函数、类定义等
- FlagGems 测试文件的完整上下文丢失

**影响**:
- 常量定义丢失
- 辅助函数丢失
- 可能的类定义丢失
- 模块级的初始化代码丢失

**当前提取逻辑**:
```python
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
        # 只处理测试函数
        ...
```

---

## 建议的修复方案

### 修复 1: 动态提取常量定义

**目标**: 自动检测并提取测试中使用的所有常量定义

**实现步骤**:

1. **解析源文件的 AST**:
   ```python
   def extract_module_constants(self, tree: ast.Module) -> Dict[str, str]:
       """Extract module-level constant definitions."""
       constants = {}
       for node in tree.body:
           if isinstance(node, ast.Assign):
               # 提取赋值语句（常量定义）
               for target in node.targets:
                   if isinstance(target, ast.Name):
                       const_name = target.id
                       # 获取常量的源代码
                       const_code = ast.unparse(node)
                       constants[const_name] = const_code
       return constants
   ```

2. **检测使用的常量**:
   ```python
   def detect_used_constants(self, test_functions: List[TestFunction],
                            available_constants: Dict[str, str]) -> List[str]:
       """Detect which constants are used in test functions."""
       used_constants = set()
       for func in test_functions:
           func_code = self.convert_test_function(func, ...)
           for const_name in available_constants.keys():
               if const_name in func_code:
                   used_constants.add(const_name)
       return list(used_constants)
   ```

3. **包含到转换后的文件**:
   - 按依赖顺序添加常量定义
   - 处理常量之间的依赖关系

**优先级**: 高

---

### 修复 2: 提取辅助函数

**目标**: 自动提取测试文件中的辅助函数

**实现步骤**:

1. **识别辅助函数**:
   ```python
   def extract_helper_functions(self, tree: ast.Module) -> List[ast.FunctionDef]:
       """Extract non-test helper functions."""
       helper_functions = []
       for node in tree.body:
           if isinstance(node, ast.FunctionDef):
               # 不是测试函数，且不是私有函数
               if not node.name.startswith('test_') and not node.name.startswith('_'):
                   helper_functions.append(node)
       return helper_functions
   ```

2. **提取函数源代码**:
   ```python
   def get_function_source(self, func_node: ast.FunctionDef, source: str) -> str:
       """Extract the complete source code of a function."""
       # 使用 ast.unparse 或从源代码中提取
       return ast.unparse(func_node)
   ```

3. **检测使用的辅助函数**:
   - 分析测试函数中调用的函数
   - 只包含被使用的辅助函数

**优先级**: 高

---

### 修复 3: 改进 flag_gems 转换逻辑

**目标**: 更智能地处理 flag_gems 相关代码

**实现步骤**:

1. **保留 vendor_name 引用**:
   ```python
   # 不转换 flag_gems.vendor_name
   # 或者转换为正确的 flagbench 等价物
   if 'flag_gems.vendor_name' in code:
       # 添加必要的导入
       # import flag_gems  # 保留用于 vendor_name
   ```

2. **处理 vendor-specific 导入**:
   ```python
   # 检测并注释掉 vendor-specific 的导入
   if 'from flag_gems.runtime.backend' in code:
       # 注释掉或跳过这些导入
       code = re.sub(
           r'from flag_gems\.runtime\.backend.*',
           r'# \g<0>  # Vendor-specific import, may need manual handling',
           code
       )
   ```

3. **添加转换规则配置**:
   ```python
   conversion_rules = {
       'flag_gems.use_gems()': 'flagbench.use_gems(REGISTERED_OPS)',
       'flag_gems.device': 'device',
       # 可配置的转换规则
   }
   ```

**优先级**: 中

---

### 修复 4: 完整的模块级代码提取

**目标**: 提取测试文件的完整上下文

**实现步骤**:

1. **提取所有模块级定义**:
   ```python
   def extract_module_context(self, tree: ast.Module) -> Dict[str, Any]:
       """Extract all module-level definitions."""
       context = {
           'constants': {},
           'functions': [],
           'classes': [],
           'imports': []
       }

       for node in tree.body:
           if isinstance(node, ast.Assign):
               # 常量定义
               ...
           elif isinstance(node, ast.FunctionDef):
               # 函数定义
               ...
           elif isinstance(node, ast.ClassDef):
               # 类定义
               ...
           elif isinstance(node, (ast.Import, ast.ImportFrom)):
               # 导入语句
               ...

       return context
   ```

2. **依赖分析**:
   - 分析测试函数的依赖
   - 只包含必要的模块级代码
   - 处理依赖顺序

3. **智能过滤**:
   - 过滤掉不需要的定义
   - 保留测试相关的代码

**优先级**: 中

---

## 实施建议

### 短期修复（快速解决当前问题）

1. **扩展 constant_definitions 字典**:
   - 手动添加常见的缺失常量（MNK_SHAPES, CAT_SHAPES, CUMSUM_SHAPES 等）
   - 从 FlagGems 测试文件中收集常量定义

2. **添加辅助函数提取**:
   - 实现基本的辅助函数提取逻辑
   - 至少提取被测试函数直接调用的函数

### 长期改进（彻底解决问题）

1. **重构转换脚本架构**:
   - 使用 AST 分析而不是字符串替换
   - 实现完整的依赖分析
   - 支持可配置的转换规则

2. **添加验证步骤**:
   - 转换后自动验证语法
   - 检测缺失的定义
   - 生成转换报告

3. **改进错误处理**:
   - 更详细的错误信息
   - 提供修复建议
   - 支持部分转换（跳过有问题的测试）

---

## 测试验证

转换脚本修复后，应该能够：

1. ✅ 成功转换 Qwen Next 算子的 69 个测试
2. ✅ 生成的测试文件可以被 Python 导入（无语法错误）
3. ✅ 生成的测试文件包含所有必要的常量和辅助函数
4. ✅ 使用 `test/test_accuracy_ut.py` 验证转换后的测试可以运行

---

## 相关文件

- 转换脚本: `scripts/convert_flaggems_tests.py`
- 测试验证脚本: `test/test_accuracy_ut.py`
- FlagGems 测试目录: `FlagGems/tests/`
- 输出目录: `src/flagbench/accuracy/`

---

## 更新日志

- 2025-12-25: 初始文档，记录转换 Qwen Next 算子时发现的问题
