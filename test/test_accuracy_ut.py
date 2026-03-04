"""
Accuracy Unit Tests - 支持多框架多测试集的精度验证脚本

支持的测试集:
  - v2: V2 算子 (50 operators)
  - v2_1: V2.1 算子 (111 operators)
  - cupy: cuBLAS 算子 (47 operators)

使用方式:
    # 测试 V2 全部算子
    python test/test_accuracy_ut.py --test-set v2 --name all

    # 测试 V2.1 指定算子
    python test/test_accuracy_ut.py --test-set v2_1 --name add,mul,softmax

    # 测试 cuBLAS 算子
    python test/test_accuracy_ut.py --test-set cupy --name all

    # 使用自定义测试文件
    python test/test_accuracy_ut.py --name abs --test-file flagbench.accuracy.test_v2_ops

    # 列出可用的测试集
    python test/test_accuracy_ut.py --list-sets
"""

import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import argparse
import sys
from typing import Dict, List, Any

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

# ============ Test Set Definitions ============

def get_test_sets() -> Dict[str, Dict[str, Any]]:
    """获取所有测试集配置"""
    from flagbench.dataset.kernel_list import (
        V2_OPERATORS,
        V2_1_OPERATORS,
        CUPY_OPERATORS,
    )

    return {
        "v2": {
            "operators": V2_OPERATORS,
            "modules": ["flagbench.accuracy.test_v2_ops"],
            "description": "V2 PyTorch operators (50 operators)",
            "label_format": "aten::{op}",  # 算子标签格式
        },
        "v2_1": {
            "operators": V2_1_OPERATORS,
            "modules": ["flagbench.accuracy.test_v2_1_ops"],
            "description": "V2.1 PyTorch operators (111 operators)",
            "label_format": "aten::{op}",
        },
        "cupy": {
            "operators": CUPY_OPERATORS,
            "modules": _get_cupy_modules(),
            "description": "cuBLAS operators via cupy (47 operators)",
            "label_format": "cupy::{op}",
        },
    }


def _get_cupy_modules() -> List[str]:
    """获取 cupy accuracy 模块列表"""
    # cupy 模块在 flagbench.accuracy.cupy 目录下
    cupy_ops = [
        "caxpy", "cdgmm", "cdotc", "cdotu", "cgeam", "cgemm", "cgemv", "cgerc", "cgeru", "cscal", "csyrk",
        "dasum", "daxpy", "ddgmm", "ddot", "dgeam", "dgemm", "dgemv", "dger", "dnrm2", "dsbmv", "dscal", "dsyrk",
        "hgemm", "sasum", "saxpy", "sdgmm", "sdot", "sgeam", "sgemm", "sgemv", "sger", "snrm2", "ssbmv", "sscal", "ssyrk",
        "zaxpy", "zdgmm", "zdotc", "zdotu", "zgeam", "zgemm", "zgemv", "zgerc", "zgeru", "zscal", "zsyrk",
    ]
    return [f"flagbench.accuracy.cupy.accuracy_{op}_cublas_ops" for op in cupy_ops]


# ============ Main Functions ============

MOCK_TRITON_CODE = "mock triton code"


def list_test_sets():
    """列出所有可用的测试集"""
    print("\nAvailable test sets:\n")
    test_sets = get_test_sets()
    for name, config in test_sets.items():
        op_count = len(config["operators"])
        print(f"  {name:10s} - {config['description']}")
        print(f"             Operators: {op_count}")
        print(f"             Modules: {len(config['modules'])} file(s)")
        print()


def list_operators(test_set: str):
    """列出指定测试集的所有算子"""
    test_sets = get_test_sets()
    if test_set not in test_sets:
        print(f"Error: Unknown test set '{test_set}'")
        print(f"Available: {', '.join(test_sets.keys())}")
        sys.exit(1)

    config = test_sets[test_set]
    operators = list(config["operators"].keys())

    print(f"\nOperators in '{test_set}' ({len(operators)} total):\n")
    for i, op in enumerate(sorted(operators), 1):
        print(f"  {i:3d}. {op}")
    print()


def test_verifier_operator(
    name: str,
    test_set: str = "v2",
    device_count: int = 1,
    timeout: int = 300,
    test_file: str = None,
    run_dir: str = None,
):
    """
    运行精度测试

    Args:
        name: 算子名称，支持逗号分隔多个或 "all"
        test_set: 测试集名称 (v2, v2_1, cupy)
        device_count: GPU 数量
        timeout: 超时时间
        test_file: 自定义测试文件（覆盖 test_set 的默认模块）
        run_dir: 运行目录
    """
    test_sets = get_test_sets()

    # 验证 test_set
    if test_set not in test_sets:
        print(f"Error: Unknown test set '{test_set}'")
        print(f"Available: {', '.join(test_sets.keys())}")
        sys.exit(1)

    config = test_sets[test_set]
    operators = config["operators"]

    # 设置运行目录
    if run_dir is None:
        run_dir = "/root/tmp/runs"

    # 创建 Verifier
    verifier_config = VerifyConfig(
        run_name=f"test_{test_set}_accuracy",
        test_type="both",
        run_dir=run_dir,
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=timeout,
    )
    verifier = Verifier(verifier_config)

    # 设置测试模块
    if test_file:
        test_files = [f.strip() for f in test_file.split(",") if f.strip()]
        print(f"Using custom test file(s): {test_files}")
        verifier.set_modules(modules=test_files, mode="accuracy")
    else:
        print(f"Using test set '{test_set}': {config['description']}")
        verifier.set_modules(modules=config["modules"], mode="accuracy")

    # 解析算子名称
    if name == "all":
        names = list(operators.keys())
    else:
        names = [n.strip() for n in name.split(",") if n.strip()]

    print(f"Testing {len(names)} operator(s)...")

    # 创建测试请求
    requests = []
    for op_name in names:
        requests.append(
            VerifyRequest(
                source=[Source(
                    source=MOCK_TRITON_CODE,
                    function_name=op_name
                )]
            )
        )

    # 执行验证
    result = verifier.only_verify(
        name_source_map=requests,
        test_type="accuracy",
        device_count=device_count
    )[-1][0]

    # 打印结果
    if len(names) == 1:
        print(f"\nVerification Result for '{names[0]}':")
        print(f"  {result}")
    else:
        print(f"\nVerification completed for {len(names)} operators")

    return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FlagBench Accuracy Unit Tests - Multi-framework support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all V2 operators
  python test/test_accuracy_ut.py --test-set v2 --name all

  # Test specific V2.1 operators
  python test/test_accuracy_ut.py --test-set v2_1 --name add,mul,softmax

  # Test cuBLAS operators
  python test/test_accuracy_ut.py --test-set cupy --name all

  # List available test sets
  python test/test_accuracy_ut.py --list-sets

  # List operators in a test set
  python test/test_accuracy_ut.py --list-ops v2_1
        """
    )

    parser.add_argument(
        "--name", type=str,
        help="Operator name(s) to test. Supports comma-separated list or 'all'."
    )
    parser.add_argument(
        "--test-set", type=str, default="v2",
        choices=["v2", "v2_1", "cupy"],
        help="Test set to use (default: v2)"
    )
    parser.add_argument(
        "--device-count", type=int, default=1,
        help="Number of GPU devices (default: 1)"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--test-file", type=str, default=None,
        help="Custom test file(s), comma-separated. Overrides --test-set modules."
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Directory for run logs (default: /root/tmp/runs)"
    )
    parser.add_argument(
        "--list-sets", action="store_true",
        help="List available test sets and exit"
    )
    parser.add_argument(
        "--list-ops", type=str, metavar="TEST_SET",
        help="List operators in specified test set and exit"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # 处理列表命令
    if args.list_sets:
        list_test_sets()
        sys.exit(0)

    if args.list_ops:
        list_operators(args.list_ops)
        sys.exit(0)

    # 验证必需参数
    if not args.name:
        print("Error: --name is required (or use --list-sets / --list-ops)")
        sys.exit(1)

    # 运行测试
    test_verifier_operator(
        name=args.name,
        test_set=args.test_set,
        device_count=args.device_count,
        timeout=args.timeout,
        test_file=args.test_file,
        run_dir=args.run_dir,
    )
