#!/usr/bin/env python3
"""
测试非 PyTorch 算子支持：non_torch_prelu 示例
"""
import os
import sys

# 设置项目路径
project_root = "/share/project/tj/workspace/flag-bench"
sys.path.insert(0, os.path.join(project_root, "src"))

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

def test_non_torch_prelu():
    """测试 non_torch_prelu 的baseline 和 triton 实现"""

    # 读取文件内容
    example_dir = os.path.join(project_root, "src/flagbench/dataset/baseline/example")
    baseline_code = open(os.path.join(example_dir, "baseline_prelu.py")).read()
    triton_code = open(os.path.join(example_dir, "triton_prelu.py")).read()
    test_code = open(os.path.join(example_dir, "test_prelu.py")).read()

    print("=" * 80)
    print("测试非 PyTorch 算子：non_torch_prelu")
    print("=" * 80)

    # 测试 1：DISPATCH_TORCH_LIB=0（测试 baseline 实现）
    print("\n" + "=" * 80)
    print("测试 1：DISPATCH_TORCH_LIB=0（baseline 实现）")
    print("=" * 80)
    os.environ["DISPATCH_TORCH_LIB"] = "0"

    # 为 baseline 测试创建单独的 verifier
    config_baseline = VerifyConfig(
        run_name="non_torch_prelu_test_baseline",
        test_type="accuracy",
        seed=42,
        sample_id=0
    )
    verifier_baseline = Verifier(config_baseline)

    try:
        summary_baseline, results_baseline = verifier_baseline.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[
                        Source(source=baseline_code, function_name="non_torch_prelu", namespace="baseline"),
                        Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
                    ],
                    test_func=[test_code],
                    test_func_mark="non_torch_prelu"
                )
            ],
            test_type="accuracy"
        )

        print("\n✓ Baseline 测试完成")
        print(f"Summary: {summary_baseline}")
        print(f"Result: {results_baseline[0]}")

    except Exception as e:
        print(f"\n✗ Baseline 测试失败")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # 测试 2：DISPATCH_TORCH_LIB=1（测试 triton 实现）
    print("\n" + "=" * 80)
    print("测试 2：DISPATCH_TORCH_LIB=1（triton 实现）")
    print("=" * 80)
    os.environ["DISPATCH_TORCH_LIB"] = "1"

    # 为 triton 测试创建单独的 verifier
    config_triton = VerifyConfig(
        run_name="non_torch_prelu_test_triton",
        test_type="accuracy",
        seed=42,
        sample_id=0
    )
    verifier_triton = Verifier(config_triton)

    try:
        summary_triton, results_triton = verifier_triton.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[
                        Source(source=baseline_code, function_name="non_torch_prelu", namespace="baseline"),
                        Source(source=triton_code, function_name="non_torch_prelu", namespace="triton")
                    ],
                    test_func=[test_code],
                    test_func_mark="non_torch_prelu"
                )
            ],
            test_type="accuracy"
        )

        print("\n✓ Triton 测试完成")
        print(f"Summary: {summary_triton}")
        print(f"Result: {results_triton[0]}")

    except Exception as e:
        print(f"\n✗ Triton 测试失败")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print("所有测试完成！")

if __name__ == "__main__":
    test_non_torch_prelu()
