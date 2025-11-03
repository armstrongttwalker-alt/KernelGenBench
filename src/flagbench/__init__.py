# from .run import TestRunner
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .dataset.kernel_list import PYTORCH_OPERATORS
from sandbox.register import Register, register, REGISTERED_OPS
from sandbox.verifier.test_parametrize import Param, parametrize, label
# __version__ = "0.1"
# device = runtime.device.name
# vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")

current_work_registrar = None

def enable(config, lib=aten_lib, unused=None, registrar=Register):
    global current_work_registrar
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


__all__ = [
    "enable",
    "use_gems",
    "register", 
]
