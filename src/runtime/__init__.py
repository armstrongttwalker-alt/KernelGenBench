"""
FlagGems runtime 扩展

补充 FlagGems runtime 中缺失的设备信息，提供统一的 runtime 接口。
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


__all__ = [
    'device',
    'torch_device_fn',
    'get_visible_devices_env',
    'VISIBLE_DEVICES_ENV',
]
