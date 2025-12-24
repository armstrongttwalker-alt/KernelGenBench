import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from utils import (
    load_right_test_function_from_test_func_dir,
    load_right_kernel_code_from_acc_verify_dir,
    convert_accuracy_to_performance_test,
    today
)

from sandbox.verifier import Verifier, VerifyRequest, VerifyConfig, Source

def check_args_validity(args: argparse.Namespace) -> None:
    # TODO: implement argument validity checks
    assert args.test_func_path != "" or args.benchmark_func_path != "", "At least one of --test-func-path or --benchmark-func-path must be provided."
    assert os.path.exists(args.path), f"The specified path {args.path} does not exist."
    assert args.device_count > 0, "Device count must be a positive integer."
    assert args.num_samples > 0, "Number of samples must be a positive integer."

def load_samples_from_path(path: Path) -> Dict[str, str]:
    paths = [p for p in Path(path).glob("*.py")]
    samples = {}
    for p in paths:
        # FIXME: namespace handling
        if "::" not in p.name:
            namespace = "aten"
        else:
            namespace = p.name.split("::")[0]
        with open(p, "r") as f:
            code = f.read()
        # samples.append({
        #     "code": code,
        #     "file_name": p.name.split(".")[0],
        # })
        samples[f"{p.name.split('.')[0]}"] = code
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated samples from a specified path.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing generated sample files.",
    )
    parser.add_argument(
        "--name", 
        type=str,
        default="all",
        help="Name of the operator to evaluate (default: all).",
    )
    parser.add_argument(
        "--test-func-path",
        type=str,
        default="",
        help="Path to the accuracy performance test function result file.",
    )
    parser.add_argument(
        "--benchmark-func-path",
        type=str,
        default="",
        help="Path to the benchmark function test result file.",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=1,
        help="Number of devices to use for evaluation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for accuracy evaluation in seconds.",
    )
    parser.add_argument(
        "--convert-to-perf",
        action="store_true",
        help="Convert accuracy test functions to performance test functions.",
    )
    args = parser.parse_args()

    check_args_validity(args)

    # Load existing performance test functions from test-func-path
    existing_test_funcs = load_right_test_function_from_test_func_dir(Path(args.test_func_path), get_success=True)
    print(f"Loaded {len(existing_test_funcs)} existing test functions from {args.test_func_path}")

    # Convert accuracy tests to performance tests if requested
    if args.convert_to_perf:
        print("Loading accuracy test functions from --path for conversion...")
        # Load accuracy test functions from the path directory
        accuracy_test_funcs = load_right_test_function_from_test_func_dir(Path(args.path), get_success=True)
        print(f"Loaded {len(accuracy_test_funcs)} accuracy test functions.")

        # Filter out operators that already have performance tests
        ops_to_convert = {
            op_name: test_func
            for op_name, test_func in accuracy_test_funcs.items()
            if op_name not in existing_test_funcs
        }
        print(f"Found {len(ops_to_convert)} operators that need conversion (not in existing tests).")

        # Convert the filtered accuracy tests to performance tests
        if ops_to_convert:
            print("Converting accuracy test functions to performance test functions...")
            converted_test_funcs = convert_accuracy_to_performance_test(ops_to_convert)
            print(f"Successfully converted {len(converted_test_funcs)} test functions.")

            # Merge existing and converted test functions
            # test_funcs = {**existing_test_funcs, **converted_test_funcs}
            test_funcs = converted_test_funcs
            # print(f"Total {len(test_funcs)} test functions available (existing + converted).")
            print(f"Total {len(test_funcs)} test functions available (converted only).")
        else:
            print("No operators need conversion, using existing test functions only.")
            test_funcs = existing_test_funcs

        test_type = "performance"
    else:
        test_funcs = existing_test_funcs
        test_type = "performance"

    # verify
    config = VerifyConfig(
        run_name="eval_perf_" + Path(args.path).name + "_" + today(),
        test_type=test_type,
        run_dir="./runs",
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=args.timeout,
    )
    verifier = Verifier(config)
    results = []
    success = []
    for sample_idx in range(args.num_samples):
        samples = load_right_kernel_code_from_acc_verify_dir(Path(args.path) / f"log_{sample_idx}" / "result.json")
        print(f"Loaded {len(samples)} kernel codes for sample index {sample_idx}.")
        if len(results) > 0:
            samples = {k: v for k, v in samples.items() if k not in success}
        if args.name != "all":
            samples = {k: v for k, v in samples.items() if k.endswith(f"::{args.name}")}
        results = verifier.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=code,
                        function_name=file_name
                    )],
                    test_func=[test_funcs[file_name]] if file_name in test_funcs else []
                ) for file_name, code in samples.items()
            ],
            test_type=test_type,
            device_count=args.device_count
        )
        verifier._running_config.sample_id += 1
        print(f"Results for sample index {sample_idx}:")
        print(results[0])
        success += [res.op_name for res in results[1] if res.success]
    failed = [res for res in results[1] if not res.success]
    print(f"Total samples evaluated: {len(success) + len(failed)}")
    print(f"Successful samples: {len(success)}")
    print(f"Failed samples: {len(failed)}")
    return success, results


if __name__ == "__main__":
    success, evaluation_results = main()
    # for result in evaluation_results[1]:
    #     print("Operator:", result.op_name)
    #     print("Success:", result.success)
    #     if result.traceback:
    #         print("Traceback:", result.traceback)
    #     print("-----")