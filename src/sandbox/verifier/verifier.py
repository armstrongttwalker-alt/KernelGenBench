from .utils import expand_params, save_benchmark_result, add_register_decorator
from sandbox.utils.accuracy_utils import VerifyResult
from sandbox.register_scanner import auto_register_module
from .test_parametrize import get_funcs_by_label, _label_registry, get_params
from fastapi.encoders import jsonable_encoder
import os
DISPATCH_TORCH_LIB = os.environ.get("DISPATCH_TORCH_LIB", "1") == "1"
from tqdm import tqdm
import traceback
import tempfile
from typing import Callable, Union, List, Optional, Any
import json
import torch
import multiprocessing as mp
import numpy as np
import importlib
from pydantic import BaseModel
from copy import deepcopy
from dataclasses import dataclass, asdict
from flagbench.perfermance.attri_util import BenchmarkResult, CustomBenchmarkResult


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pass

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class VerifyConfig:
    run_name: str
    test_type: str = "accuracy" # accuracy, performance, both
    run_dir: str = os.path.join(REPO_TOP_DIR, "runs")
    store_type: str = "local"
    strict_check: bool = False
    seed: int = 42
    sample_id: int = 0
    save_log: bool = True
    acc_timeout: int = 300    # seconds
    perf_timeout: int = 600    # seconds

@dataclass
class Source:
    source: str
    function_name: str
    namespace: str = ""

@dataclass
class VerifyRequest:
    source: List[Source]
    test_func: List[str | None] | None = None
    test_func_mark: str | None = None

import logging
from .utils import CustomRichHandler, generate_speedup_html
from rich.console import Console
console = Console()


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[CustomRichHandler(
        rich_tracebacks=True,
        show_path=True, 
        show_level=True, 
        show_time=True 
    )]
)

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

def default_converter(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (np.generic,)):
        return o.item()
    if "torch" in str(type(o)):
        return str(o)
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Callable):
        return o.__name__
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

from typing import List, Dict, Tuple

# USE_GEMS = True

def get_name_list_from_code_dir(code_dir: str) -> List[str]:
    name_list = []
    for file in os.listdir(code_dir):
        if file.endswith(".py") and file.startswith("problem_"):
            name = file[len("problem_"):].split("_sample_")[0]
            name_list.append(name)
    return name_list


class Verifier:
    def __init__(self, config: VerifyConfig):
        self.config = config
        self._running_config = deepcopy(config)
        self.accuracy_modules = []
        self.perf_modules = []

    def set_modules(self, modules: list, mode: str = "accuracy"):
        assert mode in ["accuracy", "performance"], f"mode must be accuracy or performance, got {mode}"
        if mode == "accuracy":
            self.accuracy_modules = modules
        if mode == "performance":
            self.perf_modules = modules


    def _import_module_or_path(self, module_or_path: str):
        """
        动态导入模块或文件路径
        - 如果是文件路径(.py结尾或路径分隔符),则从文件加载
        - 否则作为模块名导入
        """
        import sys
        import importlib.util
        
        # 判断是否是文件路径
        if os.path.exists(module_or_path) or module_or_path.endswith('.py') or os.path.sep in module_or_path:
            # 作为文件路径处理
            if not os.path.exists(module_or_path):
                raise FileNotFoundError(f"Module file not found: {module_or_path}")
            
            # 生成模块名 (从文件名提取)
            module_name = os.path.splitext(os.path.basename(module_or_path))[0]
            
            # 使用 importlib.util 从文件加载
            spec = importlib.util.spec_from_file_location(module_name, module_or_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {module_or_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info(f"Loaded module from file: {module_or_path}")
        else:
            # 作为模块名处理
            importlib.import_module(module_or_path)
            logger.info(f"Imported module: {module_or_path}")


    def import_tests(self, mode: str = "accuracy"):
        if not self.accuracy_modules:
            from flagbench import accuracy_modules
            self.accuracy_modules = accuracy_modules
        if not self.perf_modules:
            from flagbench import perf_modules
            self.perf_modules = perf_modules
        if mode == "accuracy":
            modules = self.accuracy_modules
        elif mode == "performance":
            modules = self.perf_modules
        elif mode == "both":
            modules = self.accuracy_modules + self.perf_modules
        for module in modules:
            self._import_module_or_path(module)


    def _init_test_func(self):
        self.import_tests(self._running_config.test_type)

    def _update_config(self, config: VerifyConfig):
        self.config = config
        self._running_config = deepcopy(config)

    def _summary(self, result: List):
        summary = {
            "total": len(result),
            "passed": sum(1 for r in result if r.success is True),
            "failed": sum(1 for r in result if r.success is False),
        }
        no_test = sum(1 for r in result if r.success is None)
        no_test_list = [r.op_name for r in result if r.success is None]
        info = {
            "info": f"there are {no_test} tests without results: {no_test_list}", 
            "no_test_list": no_test_list, 
            "failed_name_list": [r.op_name for r in result if r.success is False]
        }
        # save result and summary
        log_dir = os.path.join(
            self._running_config.run_dir, 
            self._running_config.run_name, 
            f"log_{self._running_config.sample_id}"
        )
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=default_converter)
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=default_converter)
        logger.info(f"failed {info}")
        return summary, info

    def _check_code(self, code: str, name: str, namespace: str = None):
        if not code:
            return None
        def ensure_import_torch(code: str) -> str:
            package_list = ["torch", "triton"]
            # 检查字符串中是否包含 "torch"
            for package in package_list:
                if package in code:
                    # 如果包含 "torch"，检查是否已经有 "import torch"
                    if f"import {package}" not in code:
                        # 如果没有 "import torch"，在字符串前面加上
                        code = f"import {package}\n" + code
            if "tl." in code:
                if "import triton.language as tl" not in code:
                    code = "import triton.language as tl\n" + code
            return code
        
        name = name.split(".")[-1]
        compile(code, name, "exec")
        from flagbench.dataset.kernel_list import IMPL_INFO
        # assert f"def {name}(" in code, f"no func {name} in code"
        ops = IMPL_INFO.get(name, [(name, None)])
        ops = [op.replace(".", "_") for op, _ in ops]
        for op in ops:
            if f"def {op}(" not in code:
                logger.error(f"no func {op} in code \n{code}")
                raise ValueError(f"no func {op} in code, must include def {op}")
        
        # check package import 
        code = ensure_import_torch(code)
        if "@register" not in code:
            code = "from flagbench import register\n" + code
            for op in ops:
                code = add_register_decorator(code, op, namespace, api=name)
        if not os.path.isfile(code):
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                tmp.write(code)
                triton_code_paths = tmp.name
                logger.info(f"save triton code to {triton_code_paths}")
        auto_register_module(triton_code_paths)
        return triton_code_paths

    def only_verify(
        self, 
        name_source_map: List[VerifyRequest],
        test_type: str = "accuracy", # accuracy or performance
        device_count: int = 1, 
    ) -> Tuple[Dict, List[VerifyResult]]:
        self._running_config.test_type = test_type
        results = self.verify(name_source_maps=name_source_map, device_count=device_count)
        summary, info = self._summary(results)
        return summary, results


    def _verify_with_one_device(self, task_collections: List[VerifyRequest], device_id: int, result_queue: mp.Queue):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        ctx = mp.get_context("spawn")
        for verifyrequest in task_collections:
            p  = ctx.Process(target=self._verify, kwargs={
                "verifyrequest": verifyrequest, 
                "result_queue": result_queue,
            })
            p.start()
            p.join(timeout=self.config.acc_timeout)
            if p.is_alive():
                logger.error(f"TimeoutError: Test for {verifyrequest.source[0].function_name} timed out after {self.config.acc_timeout}s.")
                p.terminate()
                p.join()
                result_queue.put(VerifyResult(
                    op_name=verifyrequest.source[0].function_name,
                    success=False,
                    traceback=f"TimeoutError: Test timed out after {self.config.acc_timeout} seconds.",
                    code=verifyrequest.source[0].source if verifyrequest.source else None,
                    test_func=verifyrequest.test_func[0] if verifyrequest.test_func else None,
                ))
            elif p.exitcode != 0:
                logger.error(f"Process for {verifyrequest.source[0].function_name} exited with code {p.exitcode}")
                result_queue.put(VerifyResult(
                    op_name=verifyrequest.source[0].function_name,
                    success=False,
                    # TODO: catch actual traceback from process, stdout/err
                    traceback=f"Process exited with code {p.exitcode}",
                    code=verifyrequest.source[0].source if verifyrequest.source else None,
                    test_func=verifyrequest.test_func[0] if verifyrequest.test_func else None,
                ))

    def verify(
        self,
        name_source_maps: List[VerifyRequest],
        device_count: int = 1
    ) -> List[VerifyResult]:
        """

        """
        check_result = []
        # if device_count > 1 and len(name_source_maps) > 1:
        total_tasks = len(name_source_maps)
        task_chunks = [[] for _ in range(device_count)]
        for i, name_source_map in enumerate(name_source_maps):
            task_chunks[i % device_count].append(name_source_map)
        result_queue = mp.Queue()

        # 每个gpu一个进程
        processes: List[mp.Process] = []
        for i, chunk in enumerate(task_chunks):
            p = mp.Process(target=self._verify_with_one_device, args=(chunk, i, result_queue))
            p.start()
            processes.append(p)
        
        with tqdm(total=total_tasks, desc="All GPUs") as pbar:
            finished = 0
            total = 0
            success = 0
            while finished < total_tasks:
                result = result_queue.get()  # 等子进程消息
                s = result.success == True
                success += s
                check_result.append(result)
                total += 1 if result.success is not None else 0
                finished += 1
                pbar.update(1)
                pbar.set_postfix(success=f"{success}/{total}")
        for p in processes:
            p.join()

        return check_result

    def run_tests(
        self, 
        name, 
        json_path: str=None, 
        max_failures: str | int = "all", 
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
                try:
                    recorded_params = jsonable_encoder(combo, custom_encoder={torch.dtype: str, Callable: lambda x: x.__name__})
                except Exception as e:
                    print(combo)
                    raise e
                try:
                    ret = func(**combo)
                except Exception as e:
                    tb_str = traceback.format_exc()
                    success = False
                if isinstance(ret, (BenchmarkResult, CustomBenchmarkResult)):
                    if speedup is None:
                        speedup = []
                    if isinstance(ret, BenchmarkResult):
                        speed_save_path = json_path.replace(".json", "_speedup.txt") if json_path else None
                        save_benchmark_result(ret, speed_save_path)
                    speed = json.loads(ret.to_json()) if isinstance(ret, BenchmarkResult) else ret.model_dump()
                    speed["params"] = recorded_params
                    # speedup.append(
                    #     speed
                    # )
                    speedup = speed['result']
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
                        "ref_time": np.mean([s["latency_base"] for s in speedup]).item(),
                        "res_time": np.mean([s["latency"] for s in speedup]).item(),
                        "speedup": np.mean([s["speedup"] for s in speedup]).item(),
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

    def run_test(
        self, 
        op_names: str, 
        config: VerifyConfig, 
    ) -> VerifyResult:
        op_mark = op_names.split(".")[-1]
        if config.save_log:
            log_dir = os.path.join(
                config.run_dir, 
                config.run_name, 
                f"log_{config.sample_id}"
            )
        else:
            log_dir = None

        # === 处理 JSON 保存路径 ===
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            json_path = os.path.join(log_dir, f"test_report_{op_mark}.json")
            delete_after = False
        else:
            tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            json_path = tmpfile.name
            tmpfile.close()
            delete_after = True

        verifyresult = self.run_tests(
            name=op_mark, 
            json_path=json_path, 
            max_failures=config.strict_check and 1 or "all",
            seed=config.seed, 
            strict_check=config.strict_check,
        )

        if delete_after:
            os.remove(json_path)
        return verifyresult
    
    def _verify(
        self,
        # source: Union[str, List[str]],
        # function_name: Union[str, List[str]],
        # test_func: Union[str, List[str]] = None, 
        # test_func_mark: str = None,
        verifyrequest: VerifyRequest,
        result_queue: mp.Queue = None, 
        namespace: Union[str, List[str]] = None, 
    ) -> VerifyResult:
        # assert type(source) == type(function_name), f"source and function_name should be same type, got {type(source)} and {type(function_name)}"
        # source = [source] if isinstance(source, str) else source
        # namespace = [namespace] if isinstance(namespace, str) or namespace is None else namespace
        # function_name = [function_name] if isinstance(function_name, str) else function_name
        # test_func = [test_func] if isinstance(test_func, str) or test_func is None else test_func
        # assert len(source) == len(function_name), f"source and function_name should be same length, got {len(source)} and {len(function_name)}"
        # if os.path.isfile(source):
        #     with open(source, "r") as f:
        #         source = f.read()
        source = [s.source for s in verifyrequest.source]
        namespace = [s.namespace for s in verifyrequest.source]
        function_name = [s.function_name for s in verifyrequest.source]
        function_name = [fn.split(".")[-1] for fn in function_name]
        test_func = verifyrequest.test_func
        test_func_mark = verifyrequest.test_func_mark
        read_file = lambda s: open(s, "r").read()
        source = [s if not s or not os.path.isfile(s) else read_file(s) for s in source]
        # source = [s if not os.path.isfile(s) else read_file(s) for s in source_or_path]
        # for s in source_or_path:
        #     if s and os.path.isfile(s):
        #         logger.info(f"read source code from file {s}")

        try:
            self._init_test_func()    # 对于外部的triton kernel开发者，没有必要初始化 flaggems 的测试函数，如果用作benchmark，可以用这一行
        except Exception as e:
            logger.error(f"Init test functions failed: {e}")
            raise e
        try:
            if DISPATCH_TORCH_LIB:
                checked_source = [self._check_code(s, fn_name, ns) for s, fn_name, ns in zip(source, function_name, namespace)]
            # TODO 
            # should check the test_func
            if test_func is not None:
                for tf in test_func:
                    self._register_test_func_from_str(tf)
        except Exception as e:
            if result_queue is not None:
                result_queue.put(
                    VerifyResult(
                        op_name=function_name[0], 
                        success=False,
                        traceback=traceback.format_exc(), 
                        info={
                            "total": 0,
                            "failed": 0,
                            "success": 0,
                        }, 
                        code=source[-1],
                        test_func=test_func[-1] if test_func is not None else None,
                    )
                )
            return VerifyResult(
                op_name=function_name[0], 
                success=False,
                traceback=traceback.format_exc(), 
                code=source[-1],
                test_func=test_func[-1] if test_func is not None else None,
            )
        res = self.run_test(
            test_func_mark if test_func_mark else function_name[0], 
            self._running_config, 
        )
        res.code = source[-1]
        res.test_func = test_func[-1] if test_func else None
        if result_queue is not None:
            result_queue.put(res)
        return res

    @staticmethod
    def _register_test_func_from_str(code: str):
        assert "@label" in code, "No @label decorator found"
        if "@parametrize" not in code:
            console.rule(f"[bold yellow]No @parametrize decorator found")

        before = set(_label_registry.keys())
        # local_ns = {}
        try:
            exec(code, globals())
        except Exception as e:
            console.rule(f"[bold red]Verify failed: {e}")
            raise ValueError(f"Verify failed: {e}")
        print(_label_registry)
        after = set(_label_registry.keys())
        new_labels = after - before

        if not new_labels:
            console.rule("[bold red]No new test function found")
            raise ValueError("No new test function found")

        console.rule(f"[bold green]New test function found: {new_labels}")
        return

    def verify_test_func(
        self, 
        test_func_name: str, 
        test_func_code: str, 
        torch_kernel_name: str, 
        torch_kernel_code: str = None, 
        mark_suffix: str = None,
    ):
        # replace the "bench.triton." to "bench." and handle bench.triton.{torch_kernel_name} patterns
        import re
        
        # First, do the simple replacement
        mocked_test_func_code = test_func_code.replace("bench.triton.", "bench.")
        
        # Then, handle more complex patterns like bench.triton.{torch_kernel_name}
        # This regex finds lines containing bench.triton.{torch_kernel_name} and replaces the entire line
        # pattern = rf'(\s*.*?)bench\.triton\.{re.escape(torch_kernel_name)}(.*?)(\n|$)'
        # replacement = rf'\1bench.{torch_kernel_name}\2\3'
        # mocked_test_func_code = re.sub(pattern, replacement, mocked_test_func_code, flags=re.MULTILINE)
        mocked_test_func_code = test_func_code.replace(f"flagbench.{torch_kernel_name}", f"torch.{torch_kernel_name}")
        
        results_with_mocked_test_func = self.only_verify(
            # name_source_map={
            #     torch_kernel_name: {
            #         "source": torch_kernel_code,
            #         "test_func": test_func_code,
            #     }
            # }
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=torch_kernel_code, 
                        function_name=torch_kernel_name,
                    )],
                    test_func=[mocked_test_func_code],
                    test_func_mark=torch_kernel_name if mark_suffix is None else torch_kernel_name + mark_suffix
                )
            ]
        )[1][0]
        results_with_mocked_test_func.test_func = test_func_code
        return results_with_mocked_test_func

    def verify_triton_kernel(
        self, 
        triton_kernel_name: str,
        triton_kernel_code: str,
        torch_kernel_name: str, 
        torch_kernel_code: str, 
        test_func_name: str = None, 
        test_func_code: str = None, 
        benchmark_func_name: str = None, 
        benchmark_func_code: str = None,
        language: str = "zh_CN",        # en_US
    ):
        assert test_func_code is not None or benchmark_func_code is not None, "either test_func_code or benchmark_func_code should be provided"
        if test_func_code is not None:
            check_result = self.only_verify(
                # name_source_map={
                #     torch_kernel_name: {
                #         "source": torch_kernel_code,
                #         "test_func": test_func_code,
                #     }
                # }
                name_source_map=[
                    VerifyRequest(
                        source=[Source(
                            source=torch_kernel_code, 
                            function_name=torch_kernel_name, 
                            namespace=None, 
                        ), Source(
                            source=triton_kernel_code, 
                            function_name=triton_kernel_name,
                            namespace="triton",
                        )],
                        test_func=[test_func_code],
                        test_func_mark=torch_kernel_name
                    )
                ]
            )[1][0]

            if benchmark_func_code is None:
                return check_result

        # strict_check = self._running_config.strict_check

        # if check_result.success is not True:
        #     if not strict_check and "Tensor-likes are not close" in check_result.traceback:
        #         logger.warning("check failed, but ignore the error and continue to benchmark")
        #     else:
        #         return check_result
        
        benchmark_result = self.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=triton_kernel_code, 
                        function_name=triton_kernel_name,
                        namespace="triton",
                    ), Source(
                        source=torch_kernel_code, 
                        function_name=torch_kernel_name, 
                        namespace=None, 
                    )],
                    test_func=[benchmark_func_code],
                    test_func_mark=benchmark_func_name
                )
            ]
        )[1][0]
        if benchmark_result.speedup is None:
            return benchmark_result
        speedup_info = [su.model_dump() for su in benchmark_result.speedup]
        benchmark_result.info["html"] = generate_speedup_html(speedup_info, title="性能对比结果", language=language)

        return benchmark_result