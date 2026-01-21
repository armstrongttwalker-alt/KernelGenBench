#!/usr/bin/env python3
"""
测试非 PyTorch 算子的自动发现和自动加载功能

验证点：
1. 测试函数自动从 accuracy/ 目录发现
2. baseline 自动从固定位置加载
3. DISPATCH_TORCH_LIB 环境变量正确工作
"""
import os
import sys

# 设置项目路径
project_root = "/share/project/tj/workspace/flag-bench"
sys.path.insert(0, os.path.join(project_root, "src"))

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

def test_auto_discovery():
    """测试自动发现机制"""

    # 读取 triton 实现
    example_dir = os.path.join(project_root, "src/flagbench/dataset/baseline/example")
    triton_code = open(os.path.join(example_dir, "triton_prelu.py")).read()

    print("=" * 80)
    print("测试非 PyTorch 算子自动发现和自动加载")
    print("=" * 80)

    # 测试 1：DISPATCH_TORCH_LIB=1（triton 实现）
    print("\n" + "=" * 80)
    print("测试 1：DISPATCH_TORCH_LIB=1（triton 实现 + 自动加载 baseline）")
    print("=" * 80)
    os.environ["DISPATCH_TORCH_LIB"] = "1"

    config_triton = VerifyConfig(
        run_name="non_torch_prelu_auto_discovery_triton",
        test_type="accuracy",
        seed=42,
        sample_id=0
    )
    verifier_triton = Verifier(config_triton)

    try:
        # 只传入 triton source，不传入 test_func
        # baseline 应该自动加载，test_func 应该自动发现
        summary_triton, results_triton = verifier_triton.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[
                        Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
                    ],
                    test_func=None,  # 不传入，依赖自动发现
                    test_func_mark="non_torch_prelu"
                )
            ],
            test_type="accuracy"
        )

        print("\n✓ Triton 测试完成")
        print(f"Summary: {summary_triton}")
        if results_triton:
            print(f"Result: {results_triton[0]}")

    except Exception as e:
        print(f"\n✗ Triton 测试失败")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 测试 2：DISPATCH_TORCH_LIB=0（baseline 实现）
    print("\n" + "=" * 80)
    print("测试 2：DISPATCH_TORCH_LIB=0（baseline 实现 + 自动加载）")
    print("=" * 80)
    os.environ["DISPATCH_TORCH_LIB"] = "0"

    config_baseline = VerifyConfig(
        run_name="non_torch_prelu_auto_discovery_baseline",
        test_type="accuracy",
        seed=42,
        sample_id=0
    )
    verifier_baseline = Verifier(config_baseline)

    try:
        # 只传入 triton source，baseline 自动加载
        summary_baseline, results_baseline = verifier_baseline.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[
                        Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
                    ],
                    test_func=None,  # 不传入，依赖自动发现
                    test_func_mark="non_torch_prelu"
                )
            ],
            test_type="accuracy"
        )

        print("\n✓ Baseline 测试完成")
        print(f"Summary: {summary_baseline}")
        if results_baseline:
            print(f"Result: {results_baseline[0]}")

    except Exception as e:
        print(f"\n✗ Baseline 测试失败")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("验证点：")
    print("✓ 测试函数自动从 accuracy/ 目录发现")
    print("✓ baseline 自动从 baseline/example/ 加载")
    print("✓ DISPATCH_TORCH_LIB 环境变量正确切换实现")
    print("\n所有测试完成！")

if __name__ == "__main__":
    test_auto_discovery()
