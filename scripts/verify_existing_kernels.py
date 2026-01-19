#!/usr/bin/env python3
"""
Verify existing generated kernels without regeneration.
This script reads previously generated kernel files and runs verification on them.
"""

import os
os.environ["FLAGBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Optional
import sys
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
from flagbench.dataset import QWEN_NEXT_OPERATORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerifyExistingKernels:
    """Verify existing generated kernels without regeneration."""

    def __init__(
        self,
        data_dir: Path,
        verify_output_dir: str = "verification_rerun",
        device_count: int = 8,
        timeout: int = 300,
        skip_verified: bool = True,
    ):
        """
        Initialize the verifier.

        Args:
            data_dir: Path to the existing data directory
            verify_output_dir: Name of the verification output directory
            device_count: Number of devices for testing
            timeout: Timeout for each test
            skip_verified: Whether to skip already verified files
        """
        self.data_dir = Path(data_dir)
        self.verify_output_dir = verify_output_dir
        self.device_count = device_count
        self.timeout = timeout
        self.skip_verified = skip_verified

        # Validate data directory
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        # Load original args
        self.original_args = self._load_original_args()

        # Setup environment
        self._setup_environment()

        # Track results
        self.verification_results: Dict[int, Dict] = {}  # {round_idx: results}
        self.all_passed_operators: Set[str] = set()

        logger.info(f"Initialized verifier for: {self.data_dir}")
        logger.info(f"Verification results will be saved to: {self.data_dir / self.verify_output_dir}")

    def _load_original_args(self) -> Dict:
        """Load original arguments from args.json."""
        args_file = self.data_dir / "args.json"
        if not args_file.exists():
            logger.warning(f"args.json not found in {self.data_dir}")
            return {}

        with open(args_file, "r") as f:
            return json.load(f)

    def _setup_environment(self):
        """Setup environment variables based on original args."""
        os.environ["FLAGBENCH_UPCAST"] = "0"
        os.environ["DISPATCH_TORCH_LIB"] = "1"
        os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

        logger.info("Environment setup complete")

    def load_existing_results(self) -> Optional[Dict]:
        """Load existing pass_at_k_results.json if available."""
        results_file = self.data_dir / "pass_at_k_results.json"
        if not results_file.exists():
            logger.warning(f"pass_at_k_results.json not found in {self.data_dir}")
            return None

        with open(results_file, "r") as f:
            return json.load(f)

    def collect_kernels_from_round(self, round_idx: int) -> List[Dict]:
        """
        Collect all kernel files from a specific round.

        Args:
            round_idx: The round index to collect from

        Returns:
            List of dicts with kernel info: {
                'op_name': str,
                'namespace': str,
                'kernel_path': Path,
                'full_name': str
            }
        """
        round_dir = self.data_dir / f"round_{round_idx}"
        if not round_dir.exists():
            logger.warning(f"Round directory does not exist: {round_dir}")
            return []

        kernels = []
        for kernel_file in round_dir.glob("*.py"):
            # Parse filename: aten::add.py -> namespace=aten, op_name=add
            filename = kernel_file.stem
            if "::" in filename:
                namespace, op_name = filename.split("::", 1)
                full_name = filename
            else:
                namespace = ""
                op_name = filename
                full_name = filename

            kernels.append({
                'op_name': op_name,
                'namespace': namespace,
                'kernel_path': kernel_file,
                'full_name': full_name
            })

        logger.info(f"Collected {len(kernels)} kernels from round {round_idx}")
        return kernels

    def create_verify_request(self, kernel_info: Dict) -> VerifyRequest:
        """
        Create verification request for a kernel.

        Args:
            kernel_info: Dict with kernel information

        Returns:
            VerifyRequest object
        """
        kernel_path = kernel_info['kernel_path']
        op_name = kernel_info['op_name']
        namespace = kernel_info['namespace']

        with open(kernel_path, "r") as f:
            kernel_code = f.read()

        return VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=f"{namespace}::{op_name}" if namespace else op_name
            )],
            test_func=None,
        )

    def verify_round(self, round_idx: int, verify_config: VerifyConfig) -> Dict:
        """
        Verify all kernels in a specific round.

        Args:
            round_idx: The round index to verify
            verify_config: Verification configuration

        Returns:
            Dict with verification results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Verifying Round {round_idx}")
        logger.info(f"{'='*60}")

        # Collect kernels from this round
        kernels = self.collect_kernels_from_round(round_idx)
        if not kernels:
            logger.warning(f"No kernels found in round {round_idx}")
            return {
                'round': round_idx,
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'passed_operators': []
            }

        # Check if already verified (if skip_verified is enabled)
        verify_result_path = self.data_dir / self.verify_output_dir / f"log_{round_idx}" / "result.json"
        if self.skip_verified and verify_result_path.exists():
            logger.info(f"Round {round_idx} already verified, loading existing results...")
            with open(verify_result_path, "r") as f:
                existing_results = json.load(f)

            passed_ops = [r['op_name'] for r in existing_results if r.get('success', False)]
            logger.info(f"Loaded existing results: {len(passed_ops)}/{len(existing_results)} passed")

            return {
                'round': round_idx,
                'total': len(existing_results),
                'passed': len(passed_ops),
                'failed': len(existing_results) - len(passed_ops),
                'skipped': len(existing_results),
                'passed_operators': passed_ops
            }

        # Prepare verification requests
        verify_requests = []
        op_names = []

        for kernel_info in kernels:
            verify_req = self.create_verify_request(kernel_info)
            verify_requests.append(verify_req)
            op_names.append(kernel_info['full_name'])

        logger.info(f"Verifying {len(verify_requests)} kernels...")

        # Update verifier config for this round
        verify_config.sample_id = round_idx

        # Create verifier
        verifier = Verifier(verify_config)
        verifier.set_modules(
            modules=["src/flagbench/accuracy/test_qwen_next_ops_with_benchmark.py"],
            mode="accuracy"
        )

        # Run verification
        _, results = verifier.only_verify(
            name_source_map=verify_requests,
            device_count=self.device_count,
        )

        # Collect results
        passed_operators = []
        for result, op_name in zip(results, op_names):
            if result.success:
                passed_operators.append(result.op_name)

        round_result = {
            'round': round_idx,
            'total': len(verify_requests),
            'passed': len(passed_operators),
            'failed': len(verify_requests) - len(passed_operators),
            'skipped': 0,
            'passed_operators': passed_operators
        }

        logger.info(f"Round {round_idx}: {len(passed_operators)}/{len(verify_requests)} passed")
        return round_result

    def run_verification(self, rounds: List[int], verify_config: VerifyConfig) -> None:
        """
        Run verification for specified rounds.

        Args:
            rounds: List of round indices to verify
            verify_config: Verification configuration
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Verification")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Rounds to verify: {rounds}")
        logger.info(f"{'='*60}\n")

        # Verify each round
        for round_idx in rounds:
            round_result = self.verify_round(round_idx, verify_config)
            self.verification_results[round_idx] = round_result
            self.all_passed_operators.update(round_result['passed_operators'])

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self) -> None:
        """Save verification results to JSON."""
        results_file = self.data_dir / self.verify_output_dir / "verification_summary.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "data_dir": str(self.data_dir),
            "verify_output_dir": self.verify_output_dir,
            "timestamp": datetime.now().isoformat(),
            "total_passed_operators": len(self.all_passed_operators),
            "passed_operators": sorted(list(self.all_passed_operators)),
            "rounds": self.verification_results,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_file}")

    def print_summary(self) -> None:
        """Print verification summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Verification Summary")
        logger.info(f"{'='*60}")

        total_kernels = sum(r['total'] for r in self.verification_results.values())
        total_passed = sum(r['passed'] for r in self.verification_results.values())
        total_failed = sum(r['failed'] for r in self.verification_results.values())

        logger.info(f"Total kernels verified: {total_kernels}")
        logger.info(f"Total passed: {total_passed}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Pass rate: {total_passed / total_kernels * 100:.2f}%" if total_kernels > 0 else "N/A")

        logger.info(f"\nPer-round results:")
        for round_idx in sorted(self.verification_results.keys()):
            result = self.verification_results[round_idx]
            logger.info(f"  Round {round_idx}: {result['passed']}/{result['total']} passed")

        logger.info(f"{'='*60}\n")


def parse_rounds(rounds_str: str, data_dir: Path) -> List[int]:
    """
    Parse rounds string to list of round indices.

    Args:
        rounds_str: String like "0,1,2" or "all"
        data_dir: Data directory to scan for available rounds

    Returns:
        List of round indices
    """
    if rounds_str.lower() == "all":
        # Find all round directories
        round_dirs = sorted([d for d in data_dir.glob("round_*") if d.is_dir()])
        rounds = [int(d.name.split("_")[1]) for d in round_dirs]
        logger.info(f"Found {len(rounds)} rounds: {rounds}")
        return rounds
    else:
        # Parse comma-separated list
        return [int(r.strip()) for r in rounds_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Verify existing generated kernels without regeneration"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the existing data directory"
    )
    parser.add_argument(
        "--rounds",
        type=str,
        default="all",
        help="Rounds to verify (e.g., '0,1,2' or 'all')"
    )
    parser.add_argument(
        "--verify-output-dir",
        type=str,
        default="verification_rerun",
        help="Name of the verification output directory (default: verification_rerun)"
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=8,
        help="Number of devices for testing (default: 8)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each test in seconds (default: 300)"
    )
    parser.add_argument(
        "--skip-verified",
        action="store_true",
        default=True,
        help="Skip already verified rounds (default: True)"
    )
    parser.add_argument(
        "--no-skip-verified",
        action="store_false",
        dest="skip_verified",
        help="Do not skip already verified rounds"
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return

    # Parse rounds
    rounds = parse_rounds(args.rounds, args.data_dir)
    if not rounds:
        logger.error("No rounds to verify")
        return

    logger.info(f"Will verify rounds: {rounds}")

    # Create verification config
    verify_output_path = args.data_dir / args.verify_output_dir
    verify_config = VerifyConfig(
        run_name="",
        test_type="triton",
        run_dir=str(verify_output_path),
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=args.timeout,
        perf_timeout=args.timeout,
    )

    # Create verifier
    verifier = VerifyExistingKernels(
        data_dir=args.data_dir,
        verify_output_dir=args.verify_output_dir,
        device_count=args.device_count,
        timeout=args.timeout,
        skip_verified=args.skip_verified,
    )

    # Run verification
    verifier.run_verification(rounds, verify_config)

    logger.info("Verification complete!")


if __name__ == "__main__":
    main()
