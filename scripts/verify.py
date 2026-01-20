from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
import os, argparse, json, time
from pathlib import Path
os.environ["FLAGBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"
# 
os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

config = VerifyConfig(
    run_name="test_verifier",
    test_type="both",
    run_dir="/root/tmp/runs",
    store_type="local",
    strict_check=True,
    seed=42,
    sample_id=int(time.time()),
    save_log=True,
)

verifier = Verifier(config)

def test_verifier_operator():
    parser = argparse.ArgumentParser(description="Test for generated operators")
    
    parser.add_argument("--filename", type=Path, default=None, help="Input filename", required=True)
    parser.add_argument("--device-count", type=int, default=1, help="Device count", required=False)

    args = parser.parse_args()

    test_path = args.filename
    assert test_path.is_file() and test_path.exists()

    verify_requests = []
    op_names = []
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        """Create verification request."""
        for i, item in enumerate(data):
            op_name = item.get('op_name', None).split("::")[-1]
            test_func = item.get('test_func', None)
            kernel_code = item.get('code', None)

            verify_request = VerifyRequest(
                source=[Source(source=kernel_code, function_name=op_name)],
                test_func=[test_func] if test_func else None)

            op_names.append(op_name)
            verify_requests.append(verify_request)
    
    result = verifier.only_verify(
        name_source_map=verify_requests,
        test_type="accuracy",        # accuracy, performance, both
        device_count=args.device_count
    )[-1][0]

if __name__ == "__main__":
    test_verifier_operator()

