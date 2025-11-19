import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

from sandbox.verifier import Verifier, VerifyRequest, VerifyConfig, Source

def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def load_right_samples_from_eval_path(path: Path) -> Dict[str, str]:
    paths = path / "result.json"
    samples = {}
    with open(paths, "r") as f:
        eval_result = json.load(f)
    # for item in eval_result:
    #     samples[item["op_name"]] = item["code"]
    samples = {item["op_name"]: item["code"] for item in eval_result if item["success"]}
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
    args = parser.parse_args()
    config = VerifyConfig(
        run_name="eval_perf_" + Path(args.path).name + "_" + today(),
        test_type="performance",
        run_dir="./runs",
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=1000,
    )
    verifier = Verifier(config)
    results = []
    success = []
    for sample_idx in range(args.num_samples):
        samples = load_right_samples_from_eval_path(Path(args.path) / f"log_{sample_idx}")
        if len(results) > 0:
            samples = {k: v for k, v in samples.items() if k not in success}
        codes = list(samples.values())
        file_names = list(samples.keys())
        results = verifier.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=codes[i],
                        function_name=file_names[i]
                    )]
                ) for i in range(len(codes))
            ], 
            test_type="performance",   # accuracy, performance, both
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