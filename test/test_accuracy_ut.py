import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import argparse
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
# from flagbench import PYTORCH_OPERATORS
from flagbench.dataset.kernel_list import V2_OPERATORS as PYTORCH_OPERATORS

mock_triton_code = "mock triton code"

def test_verifier_operator(name, device_count=1, timeout=300, test_file=None):
    config = VerifyConfig(
        run_name="test_verifier",
        test_type="both",
        run_dir="/root/tmp/runs",
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=timeout,
    )

    verifier = Verifier(config)

    # If test_file is specified, only import that file
    if test_file:
        test_files = [f.strip() for f in test_file.split(",") if f.strip()]
        print(f"Using custom test file(s): {test_files}")
        verifier.set_modules(modules=test_files, mode="accuracy")

    # Parse operator names: support comma-separated list or "all"
    if name != "all":
        names = [n.strip() for n in name.split(",") if n.strip()]
    else:
        names = list(PYTORCH_OPERATORS.keys())

    # Create requests for all operators
    requests = []
    for op_name in names:
        requests.append(
            VerifyRequest(
                source=[Source(
                    source=mock_triton_code,
                    function_name=op_name
                )]
            )
        )

    result = verifier.only_verify(
        name_source_map=requests,
        test_type="accuracy",
        device_count=device_count
    )[-1][0]

    # Print result for single operator
    if len(names) == 1:
        print("Verification Result:", result)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                       help="Operator name(s) to test. Supports comma-separated multiple operators (e.g., 'abs,mul,div') or 'all' for all operators.")
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--test-file", type=str, default=None,
                       help="Path to specific test file(s). Supports comma-separated multiple files. "
                            "When specified, only these files will be imported, ignoring default modules.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    test_verifier_operator(args.name, device_count=args.device_count, timeout=args.timeout, test_file=args.test_file)