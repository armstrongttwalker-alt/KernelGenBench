import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"

import argparse
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

mock_triton_code = "mock triton code"

config = VerifyConfig(
    run_name="test_verifier",
    test_type="both",
    run_dir="/root/tmp/runs",
    store_type="local",
    strict_check=True,
    seed=42,
    sample_id=0,
    save_log=True,
)

verifier = Verifier(config)

def test_verifier_operator(name):
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    test_verifier_operator(args.name)