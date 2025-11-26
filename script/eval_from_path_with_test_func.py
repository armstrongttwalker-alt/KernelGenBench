import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm
from utils import load_right_test_function_from_result_path

from sandbox.verifier import Verifier, VerifyRequest, VerifyConfig, Source

def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def load_samples_from_path(path: Path) -> Dict[str, str]:
    paths = [p for p in Path(path).glob("*.py")]
    samples = {}
    for p in paths:
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
        samples[f"{namespace}::{p.name.split('.')[0]}"] = code
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
        required=True,
        help="Path to the test function result file.",
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
    args = parser.parse_args()

    # get right test function
    test_funcs = load_right_test_function_from_result_path(Path(args.test_func_path) / "result.json")
    

    # verify
    config = VerifyConfig(
        run_name="eval_" + Path(args.path).name + "_" + today(),
        test_type="accuracy",
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
        samples = load_samples_from_path(Path(args.path) / f"code_{sample_idx}")
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
            test_type="accuracy",
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