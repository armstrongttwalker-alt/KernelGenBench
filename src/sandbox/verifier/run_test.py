
from dataclasses import asdict
import traceback
import itertools
import json
from dataclasses import dataclass
import importlib
from typing import Callable, List, Optional, Any, Dict, Union
import numpy as np  
import torch
from pydantic import BaseModel
from .test_parametrize import get_params
from flagbench.perfermance.attri_util import BenchmarkResult, CustomBenchmarkResult
from ..utils.accuracy_utils import VerifyResult
import logging
from rich.console import Console
from fastapi.encoders import jsonable_encoder


console = Console()
logger = logging.getLogger(__name__)



from .test_parametrize import get_funcs_by_label

def default_converter(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (np.generic,)):
        return o.item()
    if "torch" in str(type(o)):
        return str(o)
    if isinstance(o, Callable):
        return o.__name__
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pass

def run_tests(
    name, 
    json_path=None, 
    max_failures: Union[str, int] = "all", 
    seed=42, 
    strict_check=False
) -> VerifyResult:
    set_seed(seed)
    total = 0
    failed = 0
    ret = None
    funcs = get_funcs_by_label(name)
    print(funcs)
    results = []
    speedup = None
    for func, mark in funcs:
        func_name = func.__name__
        if func is None:
            logger.info(f"Fail Test function {func_name} not found")
            console.print(f"[red][bold]Fail[/bold][/red] Test function {func_name} not found")
            return VerifyResult(op_name=name, success=False, traceback=f"Test function {func_name} not found")
        params = get_params(func_name, mark)
        for combo in ([{}] if not params else expand_params(params)):
            total += 1
            success = True
            tb_str = None
            recorded_params = jsonable_encoder(combo, custom_encoder={torch.dtype: str})
            try:
                ret = func(**combo)
            except Exception as e:
                tb_str = traceback.format_exc()
                success = False
            if isinstance(ret, (BenchmarkResult, CustomBenchmarkResult)):
                if speedup is None:
                    speedup = []
                speed = asdict(ret) if isinstance(ret, BenchmarkResult) else ret.model_dump()
                speed["params"] = recorded_params
                speedup.append(
                    speed
                )
                results.append({
                    "params": combo,
                    "success": success,
                    "traceback": tb_str,
                    "speedup": speed,
                })
            else:
                results.append({
                    "params": combo,
                    "success": success,
                    "traceback": tb_str,
                })
            if not success:
                if strict_check or ("Tensor-likes are not close" not in tb_str):
                    failed += 1
                if max_failures != "all" and failed >= max_failures:
                    break
        # compute average speedup for all combos of this func and update it in speedup list, name params as avg
        if speedup is not None:
            if len(speedup) > 0:
                avg_speed = {
                    "ref_time": np.mean([s["ref_time"] for s in speedup]),
                    "res_time": np.mean([s["res_time"] for s in speedup]),
                    "speedup": np.mean([s["speedup"] for s in speedup]),
                    "params": "avg",
                }
                speedup.append(avg_speed)

        if not success:
            break
    if total == 0:
        logger.warning(f"[\033[91mFail\033[0m] No valid test cases found for {name}")
        return VerifyResult(
            op_name=name,
            success=None, 
            traceback=None,
            params=None, 
            info = {
                "total": total,
                "failed": failed,
                "success": total - failed
            }
        )

    log_flag = "[Fail]" if failed > 0 else "[Success]"
    flag = "[green]Success[/green]" if failed == 0 else "[red]Fail[/red]"
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=default_converter)
        logger.info(f"{log_flag} Test results saved to {json_path}")
        console.print(f"{flag} Test results saved to {json_path}")

    return VerifyResult(
        op_name=name,
        success=failed == 0, 
        traceback=tb_str, 
        params=recorded_params,
        speedup=speedup, 
        info={
            "total": total,
            "failed": failed,
            "success": total - failed
        }
    )

