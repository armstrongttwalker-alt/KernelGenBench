# FlagBench TODO List

## 🔧 测试验证工具改进

### test_accuracy_ut.py 增强

#### 1. 添加 --clean 参数
**优先级**: Medium
**描述**: 添加参数来自动清空历史测试结果，避免历史失败信息干扰当前测试输出

**实现方案**:
```python
parser.add_argument("--clean", action="store_true",
                   help="Clean previous results before testing")

if args.clean:
    result_file = os.path.join(log_dir, "result.json")
    if os.path.exists(result_file):
        os.remove(result_file)
```

**使用场景**:
- 测试转换后的新文件时，不想看到历史失败信息
- 需要独立的测试结果，不与历史结果混合

**使用示例**:
```bash
python test/test_accuracy_ut.py \
    --name add \
    --test-file /tmp/test_converted.py \
    --clean
```

---

#### 2. 添加 --run-name 参数
**优先级**: Medium
**描述**: 允许用户指定自定义的 run_name，使每次测试使用独立的结果目录

**实现方案**:
```python
parser.add_argument("--run-name", type=str, default="test_verifier",
                   help="Custom run name for test results directory")

config = VerifyConfig(
    run_name=args.run_name,
    ...
)
```

**使用场景**:
- 需要保留多个测试结果进行对比
- 避免不同测试之间的结果混淆

**使用示例**:
```bash
python test/test_accuracy_ut.py \
    --name add \
    --test-file /tmp/test_converted.py \
    --run-name test_add_converted_v1
```

---

## 🔄 转换工具改进

### convert_flaggems_tests.py 增强

#### 3. 添加 --auto-register 参数
**优先级**: Low
**描述**: 转换完成后自动更新 `flagbench/__init__.py` 中的 `accuracy_modules` 列表

**实现方案**:
```python
parser.add_argument("--auto-register", action="store_true",
                   help="Automatically register converted module in flagbench/__init__.py")

if args.auto_register:
    update_module_list(output_file)
```

**注意事项**:
- 需要小心处理文件格式
- 可能有版本控制冲突
- 建议先实现并充分测试

---

#### 4. 添加 --verify 参数
**优先级**: Medium
**描述**: 转换完成后自动运行验证，形成一键式工作流

**实现方案**:
```python
parser.add_argument("--verify", action="store_true",
                   help="Automatically verify converted tests after conversion")

if args.verify:
    # 调用 test_accuracy_ut.py 进行验证
    subprocess.run([
        "python", "test/test_accuracy_ut.py",
        "--name", operator_name,
        "--test-file", str(output_file)
    ])
```

**使用示例**:
```bash
python scripts/convert_flaggems_tests.py \
    --operators add \
    --output /tmp/test_add.py \
    --verify  # 转换后自动验证
```

---

## 🐛 已知问题修复

### 5. 修复 test_accuracy_ut.py 中的 3 个测试失败
**优先级**: High
**描述**: 修复之前发现的 3 个测试失败

**失败的测试**:
1. **arange**: `TypeError: arange_start_step() missing 1 required positional argument: 'step'`
   - 位置: `test_special_ops.py:620`

2. **square**: `ValueError: Operator square not found in IMPL_INFO`
   - 位置: `test_v2_ops.py:793`

3. **addmm**: `ValueError: No new test function found`
   - 位置: `verifier.py:600`

**状态**: 待调查（与 mock triton code 无关）

---

## 📁 代码重构

### 6. 修复 perfermance 目录拼写错误
**优先级**: Low
**描述**: 将 `src/flagbench/perfermance/` 重命名为 `src/flagbench/performance/`

**影响范围**:
- `src/flagbench/__init__.py` (perf_modules 列表)
- `src/sandbox/verifier/verifier.py:20` (import 语句)
- `src/generator/benchmark_func_generator.py:258` (import 语句)
- `test/test_verifier_benchmark.py:8` (import 语句)
- `test/conftest.py` (可能的引用)

**注意事项**:
- 需要同时更新所有引用
- 建议使用测试覆盖来验证修改

---

## 🧪 测试覆盖改进

### 7. 添加转换工具的测试
**优先级**: Medium
**描述**: 为 `convert_flaggems_tests.py` 添加单元测试

**测试内容**:
- 装饰器转换是否正确
- 导入语句转换是否正确
- 代码体转换是否正确
- 文件生成是否正确

---

## 📚 文档改进

### 8. 添加转换工具使用文档
**优先级**: Medium
**描述**: 创建详细的使用文档，说明如何使用转换工具和验证工具

**内容包括**:
- 完整的工作流示例
- 常见问题解答
- 最佳实践
- 故障排除指南

---

## 🔍 环境变量支持

### 9. 添加 FLAGBENCH_EXTRA_MODULES 环境变量支持
**优先级**: Low
**描述**: 在 `Verifier.import_tests` 中添加环境变量支持，允许通过环境变量指定额外的模块

**实现方案**:
```python
def import_tests(self, mode: str = "accuracy"):
    # ... 现有代码 ...

    # 从环境变量读取额外的模块
    extra = os.environ.get("FLAGBENCH_EXTRA_MODULES", "")
    if extra:
        extra_modules = [m.strip() for m in extra.split(",") if m.strip()]
        modules.extend(extra_modules)
        logger.info(f"Added {len(extra_modules)} extra modules from FLAGBENCH_EXTRA_MODULES")

    for module in modules:
        self._import_module_or_path(module)
```

**使用示例**:
```bash
FLAGBENCH_EXTRA_MODULES="flagbench.accuracy.test_converted_ops" \
    python test/test_accuracy_ut.py --name add
```

---

## 📊 优先级说明

- **High**: 影响核心功能或阻塞工作流
- **Medium**: 改善用户体验或提高效率
- **Low**: 优化或增强功能

---

## 📝 备注

- 本文件记录了 FlagBench 项目的待办事项和改进建议
- 优先级可能根据实际需求调整
- 完成的任务请标记为 ~~删除线~~
- 新的待办事项请添加到相应的分类下


## 更新文档增加 flaggems 的安装