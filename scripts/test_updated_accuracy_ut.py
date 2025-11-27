import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"
os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

import argparse
import json
from pathlib import Path
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
# from flagbench import PYTORCH_OPERATORS
from flagbench.dataset.dataloader import TorchOpsLoader
from datetime import datetime
from utils import today


mock_triton_code = "mock triton code"


def create_verify_args(paths, names, namespaces = None) -> list[VerifyRequest]:
    if namespaces is None:
        namespaces = [""] * len(names)
    verify_requests = []
    for path, name, namespace in zip(paths, names, namespaces):
        with open(path, "r") as f:
            test_func = f.read()
        verify_requests.append(
            VerifyRequest(
                source=[Source(
                    source="mock triton code",
                    function_name=f"{namespace}::{name}"
                )],
                test_func=[test_func],
            )
        )
    return verify_requests


def test_verifier_operator(name, path: Path, device_count=8):
    config = VerifyConfig(
        run_name=f"{path.parent.name}_accuracy_test_{today()}",
        test_type="both",
        run_dir=f"cache/runs",
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
    )

    verifier = Verifier(config)
    verifier.set_modules([], "accuracy")
    loader = TorchOpsLoader()
    all_operators = loader.load_all()
    
    if name != "all":
        # Single operator test
        test_path = path / f"test_accuracy_{name}.py"
        try:
            assert test_path.exists(), f"Test file {test_path} does not exist."
        except AssertionError as e:
            print(f"Error: {e}")
            return
        
        updated_accuracy_tests = [test_path]
        verifier.set_modules(updated_accuracy_tests, "accuracy")
        
        try:
            result = verifier.only_verify(
                name_source_map=[
                    VerifyRequest(
                        source=[Source(
                            source=mock_triton_code,
                            function_name=name
                        )]
                    )
                ], 
                test_type="accuracy"        # accuracy, performance, both
            )[-1][0]
            print("Verification Result:", result)
        except Exception as e:
            print(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
    else:
        # All operators test with statistics
        namespaces = list(all_operators.keys())
        paths = []
        names_ = []
        namespaces_ = []
        results_summary = []
        
        missing_test = 0
        
        for namespace in namespaces:
            operators = all_operators[namespace]
            names = list(operators.keys())
            for op_name in names:
                test_path = path / f"test_accuracy_{namespace}_{op_name}.py"
                result_entry = {
                    "operator": f"{namespace}::{op_name}",
                    "test_file": str(test_path),
                }
                
                # Check if test file exists
                if not test_path.exists():
                    result_entry["status"] = "missing_test_file"
                    result_entry["success"] = False
                    result_entry["error"] = f"Test file {test_path} does not exist"
                    missing_test += 1
                    results_summary.append(result_entry)
                    print(f"⚠ Missing test file for {op_name}")
                    continue
                paths.append(test_path)
                names_.append(op_name)
                namespaces_.append(namespace)
        
        verify_args = create_verify_args(paths, names_, namespaces_)
        results = verifier.only_verify(
            name_source_map=verify_args,
            device_count=device_count,
        )
        from pprint import pprint
        pprint(results)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--device_count", type=int, default=8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    path = Path(args.path)
    test_verifier_operator(args.name, path, device_count=args.device_count)