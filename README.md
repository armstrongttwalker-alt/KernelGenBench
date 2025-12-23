# FlagBench

FlagBench 是一个用于 Triton kernel 生成和验证的基准测试框架，支持自动生成测试用例、验证准确性和性能测试。

## 目录

- [安装](#安装)
- [项目结构](#项目结构)
- [Scripts 目录说明](#scripts-目录说明)
- [Test 目录说明](#test-目录说明)
- [使用指南](#使用指南)

## 安装

```bash
pip install -r requirements.txt
pip install .
```

## 项目结构

```
flag-bench/
├── src/                          # 源代码目录
│   ├── flagbench/               # 核心 benchmark 功能
│   ├── generator/               # 代码生成器
│   └── sandbox/                 # 沙箱验证器
├── scripts/                      # 工具脚本（详见下文）
├── test/                         # 测试文件（详见下文）
├── FlagGems/                     # FlagGems 子模块
├── output/                       # Triton kernel 生成输出
├── output_ut/                    # 单元测试生成输出
├── runs/                         # 验证运行结果
├── cache/                        # 缓存目录
└── README.md                     # 本文件
```

## Scripts 目录说明

### 核心脚本

#### 1. `generate_ut_and_verify.py`
**功能**: Pass@K 测试的完整流程，包括生成单元测试和验证

**用法**:
```bash
python scripts/generate_ut_and_verify.py \
    --name <operator_name> \
    --output-dir <output_directory> \
    --test-type accuracy \
    --max-rounds 10
```

**参数**:
- `--name`: 算子命名空间（默认: `aten`）
- `--output-dir`: 输出目录（默认: `output_ut/pass_at_k`）
- `--test-type`: 测试类型（`accuracy` 或 `performance`）
- `--max-rounds`: 最大轮数（默认: 10）

#### 2. `convert_flaggems_tests.py`
**功能**: 将 FlagGems 测试函数转换为 flagbench 格式

**用法**:
```bash
python scripts/convert_flaggems_tests.py \
    --operator <operator_name> \
    --output-dir <output_directory>
```

**特性**:
- 自动处理装饰器转换
- 智能导入语句管理
- 支持多行函数签名
- 自动添加必要的常量定义

#### 3. `generate_sample.py`
**功能**: 为已验证的准确性测试函数生成 Triton kernel 代码

**用法**:
```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 python scripts/generate_sample.py \
    --test-func-result-path <result_path>
```

#### 4. `generate_ut_sample.py`
**功能**: 生成准确性测试函数

**用法**:
```bash
python scripts/generate_ut_sample.py
```

### 评估脚本

#### 5. `eval_from_path_with_test_func.py`
**功能**: 使用测试函数验证生成的 Triton 代码

**用法**:
```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 \
python scripts/eval_from_path_with_test_func.py \
    --path <triton_code_dir> \
    --num-samples <k> \
    --device-count 8 \
    --timeout 300 \
    --test-func-path <test_func_path>
```

#### 6. `eval_from_path_with_perf_test_func.py`
**功能**: 使用性能测试函数评估生成的代码

#### 7. `eval_performance_from_acc_results.py`
**功能**: 从准确性结果评估性能

#### 8. `test_updated_accuracy_ut.py`
**功能**: 测试更新的准确性单元测试

**用法**:
```bash
python scripts/test_updated_accuracy_ut.py \
    --path <path_from_generation> \
    --device-count <gpu_counts>
```

### 其他工具

#### 9. `generate_kernel_and_verify.py`
**功能**: 生成 kernel 并进行验证的完整流程

#### 10. `generate_test_from_gems.py`
**功能**: 从 FlagGems 生成测试用例

#### 11. `utils.py`
**功能**: 提供通用工具函数

## Test 目录说明

### 单元测试

#### 1. `test_accuracy_ut.py`
**功能**: 准确性单元测试，支持单个或多个算子测试

**用法**:
```bash
# 测试单个算子
python test/test_accuracy_ut.py --name abs

# 测试多个算子（逗号分隔）
python test/test_accuracy_ut.py --name abs,mul,div

# 测试所有算子
python test/test_accuracy_ut.py --name all

# 指定设备数量和超时
python test/test_accuracy_ut.py --name abs --device-count 8 --timeout 300

# 使用自定义测试文件
python test/test_accuracy_ut.py --name abs --test-file path/to/test.py
```

**参数**:
- `--name`: 算子名称（支持逗号分隔的多个算子或 `all`）
- `--device-count`: GPU 数量（默认: 1）
- `--timeout`: 超时时间（秒，默认: 300）
- `--test-file`: 自定义测试文件路径（可选）

#### 2. `test_verifier_operator.py`
**功能**: 验证器算子测试

#### 3. `test_verifier_benchmark.py`
**功能**: 验证器基准性能测试

#### 4. `test_verifier_test_func.py`
**功能**: 验证器测试函数功能测试

#### 5. `test_fused_operator.py`
**功能**: 融合算子测试

### 结构测试

#### 6. `test_imports.py`
**功能**: 测试模块导入是否正常

#### 7. `test_module_structure.py`
**功能**: 测试模块结构完整性

## 使用指南

### 快速开始

#### 1. 生成并验证准确性测试函数

```bash
python scripts/generate_ut_sample.py
python scripts/test_updated_accuracy_ut.py --path <generated_path> --device-count 8
```

#### 2. 生成 Triton kernel 代码

```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 python scripts/generate_sample.py \
    --test-func-result-path <result_path_from_step1>
```

#### 3. 验证生成的 Triton 代码

```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 \
python scripts/eval_from_path_with_test_func.py \
    --path <triton_code_dir> \
    --num-samples 10 \
    --device-count 8 \
    --timeout 300 \
    --test-func-path <test_func_path>
```

### 完整示例

```bash
# 示例：评估生成的 Triton 代码
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 \
python scripts/eval_from_path_with_test_func.py \
    --path /share/project/tj/workspace/flag-bench/output/gpt-5-2025-08-07_num_samples_10_temp_1.6_max_tokens_16384_20251125-161139 \
    --num-samples 10 \
    --device-count 8 \
    --timeout 300 \
    --test-func-path /share/project/tj/workspace/flag-bench/cache/runs/ut_gpt-5-2025-08-07_num_samples_1_temp_0.0_max_tokens_16384_20251124-152104_accuracy_test_20251201-110330/log_0
```

### 从 FlagGems 转换测试

```bash
# 转换 FlagGems 测试到 flagbench 格式
python scripts/convert_flaggems_tests.py \
    --operator <operator_name> \
    --output-dir <output_directory>
```

### Pass@K 测试

```bash
# 运行 Pass@K 测试
python scripts/generate_ut_and_verify.py \
    --name aten \
    --test-type accuracy \
    --max-rounds 10
```

## 环境变量

- `FLAGBENCH_USE_DYNAMIC_IMPL_INFO`: 启用动态实现信息（设置为 `1`）
- `FLAGBENCH_SKIP_BOTH_TEST`: 跳过双重测试（设置为 `1`）
- `DISPATCH_TORCH_LIB`: 控制 Torch 库调度（设置为 `0`）
- `FLAGBENCH_UPCAST`: 控制类型提升（设置为 `0`）

## 输出目录

- `output/`: Triton kernel 生成结果
- `output_ut/`: 单元测试生成结果
- `runs/`: 验证运行日志和结果
- `cache/`: 缓存的中间结果

## 注意事项

1. 确保有足够的 GPU 资源用于测试
2. 某些测试可能需要较长时间，建议适当设置 `--timeout` 参数
3. 使用 `--device-count` 参数控制并行测试的 GPU 数量
4. 生成的结果会保存在对应的输出目录中，便于后续分析