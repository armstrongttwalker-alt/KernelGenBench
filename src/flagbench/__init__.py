# from .run import TestRunner
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .dataset.kernel_list import PYTORCH_OPERATORS
from sandbox.register import Register, register, REGISTERED_OPS
from sandbox.verifier.test_parametrize import Param, parametrize, label

import os
DISPATCH_TORCH_LIB = os.environ.get("DISPATCH_TORCH_LIB", "1") == "1"
# __version__ = "0.1"
# device = runtime.device.name
# vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")

current_work_registrar = None

def enable(config, lib=aten_lib, unused=None, registrar=Register):
    global current_work_registrar
    if not DISPATCH_TORCH_LIB:
        current_work_registrar = None
        return
    current_work_registrar = registrar(
        config=config,
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
    )


class use_gems:
    def __init__(self, config, unused=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register
        self.config = config

    def __enter__(self):
        enable(config=self.config, lib=self.lib, unused=self.unused, registrar=self.registrar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar


def all_ops():
    return current_work_registrar.get_all_ops()

accuracy_modules = [
    "flagbench.accuracy.test_attention_ops",
    "flagbench.accuracy.test_binary_pointwise_ops",
    "flagbench.accuracy.test_blas_ops",
    "flagbench.accuracy.test_distribution_ops",
    "flagbench.accuracy.test_general_reduction_ops",
    "flagbench.accuracy.test_norm_ops",
    # "flagbench.accuracy.test_pointwise_type_promotion",
    "flagbench.accuracy.test_reduction_ops",
    "flagbench.accuracy.test_special_ops",
    "flagbench.accuracy.test_tensor_constructor_ops",
    "flagbench.accuracy.test_unary_pointwise_ops",
    "flagbench.accuracy.test_non_flaggems_ops",
    "flagbench.accuracy.test_v2_ops",
]
perf_modules = [
    "flagbench.perfermance.test_attention_perf",
    "flagbench.perfermance.test_binary_pointwise_perf",
    "flagbench.perfermance.test_blas_perf",
    "flagbench.perfermance.test_distribution_perf",
    # skip fused for now
    # "flagbench.perfermance.test_fused_perf",
    "flagbench.perfermance.test_generic_pointwise_perf",
    "flagbench.perfermance.test_norm_perf",
    "flagbench.perfermance.test_reduction_perf",
    "flagbench.perfermance.test_select_and_slice_perf",
    "flagbench.perfermance.test_special_perf",
    "flagbench.perfermance.test_tensor_concat_perf",
    "flagbench.perfermance.test_tensor_constructor_perf",
    "flagbench.perfermance.test_unary_pointwise_perf",
]

__all__ = [
    "enable",
    "use_gems",
    "register", 
]
