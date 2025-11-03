import os
import tempfile
from .run_test import run_tests
from sandbox.utils.accuracy_utils import VerifyResult
from .verifier import VerifyConfig

import traceback


class TestRunner:
    def __init__(
        self, 
        test_file = None, 
        triton_code_path = None, 
        log_path=None
    ):
        self.test_file = test_file
        self.triton_code_paths = triton_code_path
        self.log_path = log_path

    @staticmethod
    def check_python_syntax(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, file_path, "exec")
            return True, None
        except SyntaxError as e:
            return False, traceback.format_exc()

    def run_test(
        self, 
        op_names: str, 
        config: VerifyConfig, 
    ) -> VerifyResult:
        op_mark = op_names.split(".")[-1]
        if config.save_log:
            log_dir = os.path.join(
                config.run_dir, 
                config.run_name, 
                f"log_{config.sample_id}"
            )
        else:
            log_dir = None

        # === 处理 JSON 保存路径 ===
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            json_path = os.path.join(log_dir, f"test_report_{op_mark}.json")
            delete_after = False
        else:
            tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            json_path = tmpfile.name
            tmpfile.close()
            delete_after = True

        verifyresult = run_tests(
            name=op_mark, 
            json_path=json_path, 
            baseline_times=None, 
            max_failures=1,
            seed=config.seed, 
            strict_check=config.strict_check,
        )

        if delete_after:
            os.remove(json_path)
        return verifyresult

