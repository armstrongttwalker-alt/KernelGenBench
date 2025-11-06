from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source


test_accuracy_addmm_code = """from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.config import DEVICE as device
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
import torch

@label("addmm")
@label("linear")
@label("matmul")
@parametrize("M, N, K", [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)])
@parametrize("scalar", [0.5, 1.0, 2.0])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_addmm(M, N, K, scalar, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device=device)
    mat2 = torch.randn((K, N), dtype=dtype, device=device)
    bias = torch.randn((N,), dtype=dtype, device=device)
    # ref_mat1 = to_reference(mat1, True)
    # ref_mat2 = to_reference(mat2, True)
    # ref_bias = to_reference(bias, True)
    ref_mat1 = mat1.clone()
    ref_mat2 = mat2.clone()
    ref_bias = bias.clone()
    alpha = beta = scalar

    ref_out = torch.addmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flagbench.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    assert_close(res_out, ref_out, dtype, reduce_dim=K)"""


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

def test_verifier_operator():
    result = verifier.verify_test_func(
        test_func_name="test_accuracy_addmm",
        test_func_code=test_accuracy_addmm_code,
        torch_kernel_name="addmm",
    )
    print("Verification Result:", result)


if __name__ == "__main__":
    test_verifier_operator()