# flagbench

## 安装

```bash
pip install requirements.txt
pip install .
```

## 使用

```bash
cd flag-bench
```

### accuracy test function generation

```bash
python scripts/generate_ut_sample.py
```

### accuracy test function verification

```bash
python scripts/test_updated_accuracy_ut.py --path {path from generation} --device-count {gpu counts}
```

### generate triton code for verified accuracy test function

```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 python scripts/generate_sample.py --test-func-result-path {result path from test_updated_accuracy_ut}
```

### verify generated triton code

```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 python eval_from_path_with_test_func.py --path {generated triton code dir} --num-samples {pass @ k} --device-count 8 --timeout 300 --test-func-path {result path from test_updated_accuracy_ut}
```

for example: 
```bash
FLAGBENCH_USE_DYNAMIC_IMPL_INFO=1 FLAGBENCH_SKIP_BOTH_TEST=1 python eval_from_path_with_test_func.py --path /share/project/tj/workspace/flag-bench/output/gpt-5-2025-08-07_num_samples_10_temp_1.6_max_tokens_16384_20251125-161139 --num-samples 10 --device-count 8 --timeout 300 --test-func-path /share/project/tj/workspace/flag-bench/cache/runs/ut_gpt-5-2025-08-07_num_samples_1_temp_0.0_max_tokens_16384_20251124-152104_accuracy_test_20251201-110330/log_0
```

目前已经生成好的 400+ accuracy test func 目录：/share/project/tj/workspace/flag-bench/cache/runs/ut_gpt-5-2025-08-07_num_samples_1_temp_0.0_max_tokens_16384_20251124-152104_accuracy_test_2025-11-25/log_0

