import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import argparse
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
from flagbench import PYTORCH_OPERATORS

mock_triton_code = "mock triton code"

def test_verifier_operator(name, device_count=1, timeout=300):
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

    if name != "all":
        result = verifier.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=mock_triton_code,
                        function_name=name
                    )]
                )
            ], 
            test_type="accuracy",       # accuracy, performance, both
        )[-1][0]
        print("Verification Result:", result)
    else:
        names = list(PYTORCH_OPERATORS.keys())
        requests = []
        for name in names:
            requests.append(
                VerifyRequest(
                    source=[Source(
                        source=mock_triton_code,
                        function_name=name
                    )]
                )
            )
        result = verifier.only_verify(
            name_source_map=requests, 
            test_type="accuracy",        # accuracy, performance, both
            device_count=device_count
        )[-1][0]
        # print(f"Verification Result for {name}:", result)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    test_verifier_operator(args.name, device_count=args.device_count, timeout=args.timeout)