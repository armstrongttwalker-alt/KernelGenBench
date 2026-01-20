#!/usr/bin/env python3
"""
Batch test for multiple qwen_next operators

This script tests multiple generated kernels in a directory
"""

import os
import sys
from pathlib import Path
import json
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from test_qwen_next_operator import QwenNextOperatorTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenNextBatchTester:
    """Batch test multiple qwen_next operators"""

    def __init__(self, kernel_dir: str, output_dir: str = None, pattern: str = "*.py"):
        """
        Initialize batch tester

        Args:
            kernel_dir: Directory containing generated kernel files
            output_dir: Output directory for test results
            pattern: File pattern to match (default: *.py)
        """
        self.kernel_dir = Path(kernel_dir)
        if not self.kernel_dir.exists():
            raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

        # Find all kernel files
        self.kernel_files = list(self.kernel_dir.glob(pattern))
        logger.info(f"Found {len(self.kernel_files)} kernel files in {kernel_dir}")

        # Setup output directory
        if output_dir is None:
            output_dir = self.kernel_dir / "batch_test_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_single_operator(self, kernel_file: Path, device_count: int, timeout: int) -> Dict:
        """Test a single operator"""
        try:
            tester = QwenNextOperatorTester(
                kernel_file_path=str(kernel_file),
                output_dir=str(self.output_dir / kernel_file.stem)
            )
            results = tester.run_test(device_count=device_count, timeout=timeout)
            tester.save_results(results)
            return results
        except Exception as e:
            logger.error(f"Error testing {kernel_file.name}: {e}")
            return {
                "operator": kernel_file.stem,
                "kernel_file": str(kernel_file),
                "success": False,
                "error": str(e)
            }

    def run_batch_test(
        self,
        device_count: int = 1,
        timeout: int = 300,
        max_workers: int = 1
    ) -> List[Dict]:
        """
        Run batch test for all operators

        Args:
            device_count: Number of devices per test
            timeout: Timeout per test
            max_workers: Number of parallel workers

        Returns:
            List of test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting batch test for {len(self.kernel_files)} operators")
        logger.info(f"{'='*60}")

        all_results = []

        if max_workers == 1:
            # Sequential execution
            for kernel_file in self.kernel_files:
                logger.info(f"\nTesting: {kernel_file.name}")
                result = self.test_single_operator(kernel_file, device_count, timeout)
                all_results.append(result)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.test_single_operator,
                        kernel_file,
                        device_count,
                        timeout
                    ): kernel_file
                    for kernel_file in self.kernel_files
                }

                for future in as_completed(futures):
                    kernel_file = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        logger.info(
                            f"Completed: {kernel_file.name} - "
                            f"{'PASSED' if result['success'] else 'FAILED'}"
                        )
                    except Exception as e:
                        logger.error(f"Error processing {kernel_file.name}: {e}")
                        all_results.append({
                            "operator": kernel_file.stem,
                            "kernel_file": str(kernel_file),
                            "success": False,
                            "error": str(e)
                        })

        return all_results

    def save_summary(self, results: List[Dict]):
        """Save batch test summary"""
        summary_file = self.output_dir / "batch_test_summary.json"

        # Calculate statistics
        total = len(results)
        passed = sum(1 for r in results if r.get("success", False))
        failed = total - passed

        summary = {
            "total_operators": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "results": results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info("Batch Test Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total operators: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Pass rate: {passed / total * 100:.2f}%" if total > 0 else "0%")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"{'='*60}\n")

        return summary_file


def main():
    parser = argparse.ArgumentParser(
        description="Batch test qwen_next operators with generated kernels"
    )

    parser.add_argument(
        "kernel_dir",
        type=str,
        help="Directory containing generated kernel .py files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results (default: kernel_dir/batch_test_results)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.py",
        help="File pattern to match (default: *.py)"
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=1,
        help="Number of devices per test (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per test (default: 300)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    args = parser.parse_args()

    # Create batch tester
    tester = QwenNextBatchTester(
        kernel_dir=args.kernel_dir,
        output_dir=args.output_dir,
        pattern=args.pattern
    )

    # Run batch test
    results = tester.run_batch_test(
        device_count=args.device_count,
        timeout=args.timeout,
        max_workers=args.max_workers
    )

    # Save summary
    summary_file = tester.save_summary(results)

    # Exit with appropriate code
    passed = sum(1 for r in results if r.get("success", False))
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
