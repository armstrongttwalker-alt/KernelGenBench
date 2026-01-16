#!/usr/bin/env python3
"""
Test for qwen_next operators - reads generated kernel and tests with flagbench

This test:
1. Takes a generated kernel file path as input
2. Reads the kernel code from the file
3. Uses flagbench's qwen_next test framework to verify the operator
"""

import os
import sys
from pathlib import Path
import json
import argparse
import logging

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenNextOperatorTester:
    """Test individual qwen_next operators with generated kernels"""

    def __init__(self, kernel_file_path: str, output_dir: str = None):
        """
        Initialize the tester

        Args:
            kernel_file_path: Path to the generated kernel .py file
            output_dir: Optional output directory for test results
        """
        self.kernel_file_path = Path(kernel_file_path)
        if not self.kernel_file_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file_path}")

        # Extract operator name from file path
        # Expected format: aten::operator_name.py or operator_name.py
        self.operator_name = self.kernel_file_path.stem
        if "::" in self.operator_name:
            self.operator_name = self.operator_name.split("::")[-1]

        # Setup output directory
        if output_dir is None:
            output_dir = self.kernel_file_path.parent / "test_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized tester for operator: {self.operator_name}")
        logger.info(f"Kernel file: {self.kernel_file_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def read_kernel_code(self) -> str:
        """Read the kernel code from the file"""
        logger.info(f"Reading kernel code from: {self.kernel_file_path}")
        with open(self.kernel_file_path, 'r') as f:
            code = f.read()

        logger.info(f"Kernel code length: {len(code)} characters")
        return code

    def create_verify_request(self, kernel_code: str) -> VerifyRequest:
        """
        Create a verification request for the operator

        Args:
            kernel_code: The kernel code to verify

        Returns:
            VerifyRequest object
        """
        return VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=f"aten::{self.operator_name}"
            )],
            test_func=None,  # Will use the qwen_next test module
        )

    def run_test(self, device_count: int = 1, timeout: int = 300) -> dict:
        """
        Run the test for the operator

        Args:
            device_count: Number of devices to use for testing
            timeout: Timeout in seconds for each test

        Returns:
            Dictionary with test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing operator: {self.operator_name}")
        logger.info(f"{'='*60}")

        # Read kernel code
        kernel_code = self.read_kernel_code()

        # Setup environment for qwen_next testing
        os.environ["FLAGBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"
        os.environ["FLAGBENCH_UPCAST"] = "1"
        os.environ["DISPATCH_TORCH_LIB"] = "1"
        os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

        # Create verification config
        verify_config = VerifyConfig(
            run_name=f"qwen_next_{self.operator_name}_test",
            test_type="triton",
            run_dir=str(self.output_dir),
            store_type="local",
            strict_check=True,
            seed=42,
            sample_id=0,
            save_log=True,
            acc_timeout=timeout,
            perf_timeout=timeout,
        )

        # Create verifier and set qwen_next test module
        verifier = Verifier(verify_config)
        verifier.set_modules(
            # modules=["src/flagbench/accuracy/test_qwen_next_ops.py"],
            modules=["src/flagbench/accuracy/test_qwen_next_ops_with_benchmark.py"], 
            mode="accuracy"
        )

        # Create verification request
        verify_request = self.create_verify_request(kernel_code)

        # Run verification
        logger.info(f"Running verification for {self.operator_name}...")
        try:
            _, results = verifier.only_verify(
                name_source_map=[verify_request],
                device_count=device_count,
            )

            # Process results
            if results and len(results) > 0:
                result = results[0]
                success = result.success
                logger.info(f"Test result: {'PASSED' if success else 'FAILED'}")

                # Load detailed test report
                result_file = self.output_dir / "log_0" / "result.json"
                detailed_results = None
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        detailed_results = json.load(f)

                return {
                    "operator": self.operator_name,
                    "kernel_file": str(self.kernel_file_path),
                    "success": success,
                    "result": result,
                    "detailed_results": detailed_results
                }
            else:
                logger.error("No results returned from verification")
                return {
                    "operator": self.operator_name,
                    "kernel_file": str(self.kernel_file_path),
                    "success": False,
                    "error": "No results returned"
                }

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return {
                "operator": self.operator_name,
                "kernel_file": str(self.kernel_file_path),
                "success": False,
                "error": str(e)
            }

    def save_results(self, results: dict):
        """Save test results to JSON file"""
        results_file = self.output_dir / f"test_results_{self.operator_name}.json"

        # Convert non-serializable objects
        serializable_results = {
            "operator": results["operator"],
            "kernel_file": results["kernel_file"],
            "success": results["success"],
        }

        if "error" in results:
            serializable_results["error"] = results["error"]

        if "detailed_results" in results and results["detailed_results"]:
            serializable_results["detailed_results"] = results["detailed_results"]

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_file}")
        return results_file


def main():
    parser = argparse.ArgumentParser(
        description="Test qwen_next operator with generated kernel"
    )

    parser.add_argument(
        "kernel_file",
        type=str,
        help="Path to the generated kernel .py file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results (default: kernel_file_dir/test_results)"
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=1,
        help="Number of devices to use for testing (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each test (default: 300)"
    )

    args = parser.parse_args()

    # Create tester
    tester = QwenNextOperatorTester(
        kernel_file_path=args.kernel_file,
        output_dir=args.output_dir
    )

    # Run test
    results = tester.run_test(
        device_count=args.device_count,
        timeout=args.timeout
    )

    # Save results
    results_file = tester.save_results(results)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Operator: {results['operator']}")
    logger.info(f"Kernel file: {results['kernel_file']}")
    logger.info(f"Success: {results['success']}")
    if "error" in results:
        logger.info(f"Error: {results['error']}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*60}\n")

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
