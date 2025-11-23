import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import argparse
import json
from pathlib import Path
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
# from flagbench import PYTORCH_OPERATORS
from flagbench.dataset.dataloader import OperatorLoader

mock_triton_code = "mock triton code"

config = VerifyConfig(
    run_name="test_verifier",
    test_type="both",
    run_dir="cache/runs",
    store_type="local",
    strict_check=True,
    seed=42,
    sample_id=0,
    save_log=True,
)

verifier = Verifier(config)

# updated_accuracy_tests = [
#     "cache/accuracy/" + path for path in os.listdir("cache/accuracy") if path.startswith("test_") and path.endswith(".py") and "tensor_wrapper" not in path
# ]

def test_verifier_operator(name, path):
    loader = OperatorLoader()
    all_operators = loader.load_all(merge=True)
    
    if name != "all":
        # Single operator test
        test_path = path / f"test_{name}.py"
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
        names = list(all_operators.keys())
        results_summary = []
        
        total = len(names)
        successful = 0
        failed = 0
        missing_test = 0
        
        for op_name in names:
            test_path = path / f"test_{op_name}.py"
            
            result_entry = {
                "operator": op_name,
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
            
            # Set up verifier for this test
            try:
                updated_accuracy_tests = [test_path]
                verifier.set_modules(updated_accuracy_tests, "accuracy")
                
                result = verifier.only_verify(
                    name_source_map=[
                        VerifyRequest(
                            source=[Source(
                                source=mock_triton_code,
                                function_name=op_name
                            )]
                        )
                    ], 
                    test_type="accuracy"
                )[-1][0]
                
                result_entry["status"] = "tested"
                result_entry["success"] = result.success
                result_entry["info"] = result.info if hasattr(result, 'info') else None
                result_entry["traceback"] = result.traceback if hasattr(result, 'traceback') else None
                
                if result.success:
                    successful += 1
                    print(f"✓ {op_name}: Success")
                else:
                    failed += 1
                    print(f"✗ {op_name}: Failed")
                    
            except Exception as e:
                result_entry["status"] = "error"
                result_entry["success"] = False
                result_entry["error"] = str(e)
                failed += 1
                print(f"✗ {op_name}: Exception - {e}")
            
            results_summary.append(result_entry)
        
        # Save summary
        summary_path = path / "test_summary.json"
        summary_data = {
            "total": total,
            "successful": successful,
            "failed": failed,
            "missing_test": missing_test,
            "success_rate": f"{successful / total * 100:.2f}%" if total > 0 else "0%",
            "results": results_summary
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Test Summary:")
        print(f"Total operators: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Missing test files: {missing_test}")
        print(f"Success rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*60}\n")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    path = Path(args.path)
    test_verifier_operator(args.name, path)