from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"
os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

benchmark_code = """from flagbench.perfermance.performance_utils import GenericBenchmark
from sandbox.verifier.test_parametrize import label, parametrize, Param
import torch

@label("scaled_dot_product_attention")
def test_perf_scaled_dot_product_attention():
    class AttentionBenchmark(GenericBenchmark):
        def set_more_shapes(self):
            # self.shapes is a list of tuples, each containing three elements:
            # (batch, num_heads, seq_len, head_size).
            return None
    def scaled_dot_product_attention_kwargs(shape, dtype, device):
        query = torch.randn(shape, device=device, dtype=dtype)
        key = torch.randn(shape, device=device, dtype=dtype)
        value = torch.randn(shape, device=device, dtype=dtype)
        yield query, key, value, None, 0.0, True

    bench = AttentionBenchmark(
        op_name="scaled_dot_product_attention",
        input_fn=scaled_dot_product_attention_kwargs,
        torch_op=torch.nn.functional.scaled_dot_product_attention,
        dtypes=[
            torch.float16,
            torch.bfloat16,
        ],
    )
    return bench.run()"""

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

def test_verifier_benchmark():
    result = verifier.verify_test_func(
        test_func_name="test_perf_scaled_dot_product_attention",
        test_func_code=benchmark_code,
        torch_kernel_name="scaled_dot_product_attention",
    )
    print("Verification Result:", result)


if __name__ == "__main__":
    test_verifier_benchmark()