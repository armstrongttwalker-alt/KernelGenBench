# Runtime 集成方案

## 背景

当前项目需要支持多种硬件设备（CUDA、Ascend NPU、MUSA），不同设备有不同的：
- 环境变量（如 `CUDA_VISIBLE_DEVICES` vs `ASCEND_RT_VISIBLE_DEVICES`）
- torch 设备 API（如 `torch.cuda` vs `torch.npu` vs `torch.musa`）
- LLM Prompt 约束（设备特定的编程注意事项）

## 方案

直接使用 FlagGems 的 runtime 模块，不重复造轮子。创建轻量扩展模块补充缺失信息，在 PromptBuilder 层根据设备类型添加 Prompt 约束。

---

## 1. 使用 FlagGems Runtime

FlagGems 已经提供了完整的 runtime 支持（`FlagGems/src/flag_gems/runtime/`）：

```python
from flag_gems.runtime import device, torch_device_fn

# 设备名称
device.name  # 'cuda', 'npu', 'musa'

# 厂商名称
device.vendor_name  # 'nvidia', 'ascend', 'mthreads'

# torch 设备 API
torch_device_fn.synchronize()
torch_device_fn.device_count()
```

---

## 2. 创建 Runtime 扩展模块

FlagGems 的 `VendorInfoBase` 缺少 `visible_devices_env` 字段，创建轻量扩展模块补充：

**文件**: `src/runtime/__init__.py`

```python
"""
FlagGems runtime 扩展

补充 FlagGems runtime 中缺失的设备信息
"""
from flag_gems.runtime import device, torch_device_fn

# 设备可见性环境变量映射
VISIBLE_DEVICES_ENV = {
    'cuda': 'CUDA_VISIBLE_DEVICES',
    'npu': 'ASCEND_RT_VISIBLE_DEVICES',
    'musa': 'MUSA_VISIBLE_DEVICES',
}

def get_visible_devices_env() -> str:
    """获取当前设备的可见性环境变量名"""
    return VISIBLE_DEVICES_ENV.get(device.name, 'CUDA_VISIBLE_DEVICES')

# 重新导出 FlagGems runtime 的常用对象
__all__ = [
    'device',
    'torch_device_fn',
    'get_visible_devices_env',
    'VISIBLE_DEVICES_ENV',
]
```

---

## 3. 在 PromptBuilder 中添加设备约束

**文件**: `src/generator/prompt_builder.py` 或 `src/generator/torch_prompt_builder.py`

```python
from runtime import device

class PromptBuilder(ABC):
    def _get_device_constraints(self) -> str:
        """根据当前设备返回 Prompt 约束"""
        if device.name == 'npu':
            return """
## Device-Specific Requirements
It should be noted that the operator runs on Ascend NPU devices.
1. If `import torch` is used, it must be immediately followed by `import torch_npu`.
2. The device type is `npu`, e.g., `torch.device("npu:0")`, `torch.npu.synchronize()`.
3. "ub overflow" errors indicate UB overflow - split computation into smaller chunks.
"""
        elif device.name == 'musa':
            return """
## Device-Specific Requirements
It should be noted that the operator runs on MUSA devices.
1. If `import torch` is used, it must be immediately followed by `import torch_musa`.
2. The device type is `musa`, e.g., `torch.device("musa:0")`, `torch.musa.synchronize()`.
"""
        return ""  # CUDA 无需额外约束
```

在 `TorchPromptBuilder` 的 `build_new`, `build_fix` 等方法末尾调用：

```python
class TorchPromptBuilder(PromptBuilder):
    def build_new(self, gen_args: BaseGenerateArgs) -> str:
        # ... 现有代码 ...
        prompt += "You must use ```python ... ``` to format the code block.\n"

        # 追加设备约束
        prompt += self._get_device_constraints()

        return prompt

    def build_fix(self, gen_args: BaseGenerateArgs) -> str:
        # ... 现有代码 ...

        # 追加设备约束
        prompt += self._get_device_constraints()

        return prompt
```

---

## 4. 修改 verifier.py

使用 runtime 扩展模块设置设备可见性：

**文件**: `src/sandbox/verifier/verifier.py`

```python
from runtime import get_visible_devices_env

class Verifier:
    def _verify_with_one_device(self, task_collections, device_id, result_queue):
        # 使用 runtime 扩展获取正确的环境变量名
        os.environ[get_visible_devices_env()] = str(device_id)

        ctx = mp.get_context("spawn")
        # ... 后续代码不变
```

---

## 5. 简化 config.py

**文件**: `src/sandbox/config.py`

```python
from runtime import device

QUICK_MODE = False
TO_CPU = False
RECORD_LOG = "none"

# 设备名称从 runtime 获取
DEVICE = device.name
```

---

## 6. 修改 sampler/utils.py

添加 None 检查（从原分支移植的 bugfix）：

```python
def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    if output_string is None:
        return None
    # ... 后续代码
```

---

## 实施计划

### 阶段 1：创建 Runtime 扩展模块

- [ ] 创建 `src/runtime/__init__.py`
- [ ] 添加 `VISIBLE_DEVICES_ENV` 映射和 `get_visible_devices_env()` 函数
- [ ] 重新导出 FlagGems runtime 的常用对象

### 阶段 2：集成到现有代码

- [ ] 修改 `config.py`，从 runtime 获取设备名
- [ ] 修改 `verifier.py`，使用 `get_visible_devices_env()` 设置设备可见性
- [ ] 修改 `sampler/utils.py`，添加 None 检查

### 阶段 3：添加设备 Prompt 约束

- [ ] 在 `PromptBuilder` 基类添加 `_get_device_constraints()` 方法
- [ ] 在 `TorchPromptBuilder` 的 build 方法中调用设备约束

### 阶段 4：测试和验证

- [ ] 在 CUDA 环境测试
- [ ] 在 NPU 环境测试（如有条件）
- [ ] 在 MUSA 环境测试（如有条件）
- [ ] 验证向后兼容性

### 阶段 5：文档和清理

- [ ] 更新 CLAUDE.md 使用说明
- [ ] 清理原有分支中的硬编码逻辑
- [ ] 合并到 dev 分支

---

## 使用方式

FlagGems runtime 会自动检测设备，也可以通过环境变量 `GEMS_VENDOR` 指定：

```bash
# 自动检测设备
python scripts/generate_ut_and_verify.py ...

# 显式指定设备（使用 FlagGems 的环境变量）
GEMS_VENDOR=nvidia python scripts/generate_ut_and_verify.py ...
GEMS_VENDOR=ascend python scripts/generate_ut_and_verify.py ...
GEMS_VENDOR=mthreads python scripts/generate_ut_and_verify.py ...
```
