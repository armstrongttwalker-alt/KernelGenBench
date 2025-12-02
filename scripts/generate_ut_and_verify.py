import os
os.environ["DISPATCH_TORCH_LIB"] = "0"
os.environ["FLAGBENCH_UPCAST"] = "0"
os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set, List
import sys
from datetime import datetime

from flagbench.dataset import TorchOpsLoader, APIInfo
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
from utils import today

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from generator import GENERATOR
from generator.sampler.generate_samples import (
    TestFuncGenerateArgs,
    BenchmarkFuncGenerateArgs,
    GenerationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mock_triton_code = "mock triton code"


class PassAtKTester:
    def __init__(
        self,
        output_dir: Path,
        test_type: str,
        gen_config: GenerationConfig,
        verify_config: VerifyConfig,
        device_count: int = 8,
    ):
        self.output_dir = output_dir
        self.test_type = test_type
        self.gen_config = gen_config
        self.verify_config = verify_config
        self.device_count = device_count
        self.operator_loader = TorchOpsLoader()
        
        # Track results
        self.all_operators: Dict[str, Dict[str, APIInfo]] = {}
        self.passed_operators: Set[str] = set()
        self.results_by_round: List[Dict] = []
        self.generated_codes: Dict[str, Dict[int, str]] = {}  # {op_name: {round: code}}
        self.generation_summaries: List[Dict] = []  # Track generation summaries for each round
        
    def initialize_operators(self, namespace: str = "all") -> None:
        """Initialize the list of operators to test."""
        if namespace.lower() == "all":
            self.all_operators = self.operator_loader.load_all()
        else:
            self.all_operators = {namespace: self.operator_loader.load_namespace(namespace=namespace)}
        # for debug, use a subset of operators
        self.all_operators = {ns: dict(list(apis.items())[:2]) for ns, apis in self.all_operators.items()}
        total_ops = sum(len(ops) for ops in self.all_operators.values())
        logger.info(f"Initialized {total_ops} operators across {len(self.all_operators)} namespaces")
    
    def get_remaining_operators(self) -> Dict[str, Dict[str, APIInfo]]:
        """Get operators that haven't passed yet."""
        remaining = {}
        for namespace, apis in self.all_operators.items():
            remaining_apis = {}
            for api_name, api_info in apis.items():
                full_name = f"{namespace}::{api_name}"
                if full_name not in self.passed_operators:
                    remaining_apis[api_name] = api_info
            if remaining_apis:
                remaining[namespace] = remaining_apis
        return remaining
    
    def create_ut_generate_args(self, api_name: str, operators: APIInfo, namespace: str) -> TestFuncGenerateArgs:
        """Create test generation arguments."""
        kernel_name = api_name.split('.')[-1]
        return TestFuncGenerateArgs(
            kernel_name=kernel_name,
            operators=operators.schemas,
            test_func_name=f"test_accuracy_{namespace}::{kernel_name}",
            ops_namespace=namespace,
        )
    
    def generate_round(self, round_idx: int, remaining_operators: Dict[str, Dict[str, APIInfo]]) -> Path:
        """Generate tests for remaining operators in this round."""
        round_dir = self.output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx}: Generating tests")
        logger.info(f"{'='*60}")
        
        # Prepare generation arguments
        gen_args = []
        api_names = []
        for namespace, apis in remaining_operators.items():
            for api_name, api_info in apis.items():
                gen_arg = self.create_ut_generate_args(api_name, api_info, namespace)
                gen_arg.sample_id = round_idx
                gen_args.append(gen_arg)
                api_names.append(f"{namespace}::{api_name}")
        
        if not gen_args:
            logger.info("No operators to generate")
            return round_dir
        
        logger.info(f"Generating {len(gen_args)} tests...")
        
        # Generate tests
        generator = GENERATOR[self.test_type](self.gen_config)
        generated_codes = generator(gen_args)
        
        # Process and save the generated codes
        generation_results = []
        saved_count = 0
        
        for idx, (generated_code, name, sample_id) in enumerate(generated_codes):
            full_name = name
            
            result_entry = {
                "operator": full_name,
                "test_file_name": f"{name}.py",
                "success": False,
                "error": None,
                "code_length": 0,
            }
            
            if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
                try:
                    test_file = round_dir / f"{name}.py"
                    with open(test_file, "w") as f:
                        f.write(generated_code)
                    
                    result_entry["success"] = True
                    result_entry["code_length"] = len(generated_code)
                    saved_count += 1
                    
                    # Store the generated code
                    if name not in self.generated_codes:
                        self.generated_codes[name] = {}
                    self.generated_codes[name][round_idx] = generated_code
                    
                except Exception as e:
                    result_entry["error"] = str(e)
                    logger.error(f"Failed to save {name}: {e}")
            else:
                result_entry["error"] = "Empty or invalid generated code"
            
            generation_results.append(result_entry)
        
        # Calculate statistics
        total = len(gen_args)
        successful = sum(1 for r in generation_results if r["success"])
        failed = total - successful
        
        # Create generation summary for this round
        generation_summary = {
            "round": round_idx,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_apis": total,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total if total > 0 else 0,
            },
            "results": generation_results,
            "failed_operators": [r["operator"] for r in generation_results if not r["success"]],
        }
        
        # Save round generation summary
        round_summary_path = round_dir / "generation_summary.json"
        with open(round_summary_path, "w") as f:
            json.dump(generation_summary, f, indent=2, ensure_ascii=False)
        
        # Store in overall summaries
        self.generation_summaries.append(generation_summary)
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx} Generation Summary:")
        logger.info(f"Total APIs: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
        logger.info(f"Summary saved to: {round_summary_path}")
        logger.info(f"{'='*60}\n")
        
        return round_dir
    
    def create_verify_args(self, test_path: Path, op_name: str, namespace: str) -> VerifyRequest:
        """Create verification request."""
        with open(test_path, "r") as f:
            test_func = f.read()
        return VerifyRequest(
            source=[Source(
                source=mock_triton_code,
                function_name=f"{namespace}::{op_name}"
            )],
            test_func=[test_func],
        )
    
    def verify_round(self, round_idx: int, round_dir: Path, remaining_operators: Dict[str, Dict[str, APIInfo]]) -> Set[str]:
        """Verify tests for this round and return newly passed operators."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx}: Verifying tests")
        logger.info(f"{'='*60}")
        
        # Update verifier config
        self.verify_config.run_name = f"pass_at_k_round_{round_idx}_{today()}"
        self.verify_config.sample_id = round_idx
        
        verifier = Verifier(self.verify_config)
        
        # Prepare verification requests
        verify_requests = []
        op_names = []
        
        for namespace, apis in remaining_operators.items():
            for api_name in apis.keys():
                test_file = round_dir / f"test_accuracy_{namespace}::{api_name}.py"
                
                if not test_file.exists():
                    logger.warning(f"Missing test file: {test_file}")
                    continue
                
                verify_req = self.create_verify_args(test_file, api_name, namespace)
                verify_requests.append(verify_req)
                op_names.append(f"{namespace}::{api_name}")
        
        if not verify_requests:
            logger.info("No tests to verify")
            return set()
        
        logger.info(f"Verifying {len(verify_requests)} tests...")
        
        # Run verification
        _, results = verifier.only_verify(
            name_source_map=verify_requests,
            device_count=self.device_count,
        )
        
        # Collect newly passed operators and update first pass round
        newly_passed = set()
        for result in results:
            if result.success:
                newly_passed.add(result.op_name)
                # Update operator details with first pass round
                if result.op_name in self.generated_codes:
                    # This will be used when saving results
                    pass
        
        logger.info(f"Passed: {len(newly_passed)}/{len(verify_requests)} tests")
        return newly_passed
    
    def run_pass_at_k(self, max_rounds: int = 10) -> None:
        """Run pass@k testing for multiple rounds."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Pass@K Testing (max {max_rounds} rounds)")
        logger.info(f"{'='*60}\n")
        
        total_operators = sum(len(ops) for ops in self.all_operators.values())
        
        # Track when each operator first passed
        self.first_pass_round: Dict[str, int] = {}
        
        for round_idx in range(max_rounds):
            # Get remaining operators
            remaining = self.get_remaining_operators()
            remaining_count = sum(len(ops) for ops in remaining.values())
            
            if remaining_count == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"All operators passed! Stopping at round {round_idx}")
                logger.info(f"{'='*60}\n")
                break
            
            logger.info(f"\nRound {round_idx}: {remaining_count}/{total_operators} operators remaining")
            
            # Generate tests
            round_dir = self.generate_round(round_idx, remaining)
            
            # Verify tests
            newly_passed = self.verify_round(round_idx, round_dir, remaining)
            
            # Update passed operators and track first pass round
            for op_name in newly_passed:
                if op_name not in self.passed_operators:
                    self.first_pass_round[op_name] = round_idx
            
            self.passed_operators.update(newly_passed)
            
            # Record round results
            round_result = {
                "round": round_idx,
                "remaining_before": remaining_count,
                "newly_passed": len(newly_passed),
                "newly_passed_operators": sorted(list(newly_passed)),
                "total_passed": len(self.passed_operators),
                "pass_rate": len(self.passed_operators) / total_operators,
            }
            self.results_by_round.append(round_result)
            
            # Save intermediate results
            self.save_results()
            
            logger.info(f"\nRound {round_idx} Summary:")
            logger.info(f"  Newly passed: {len(newly_passed)}")
            logger.info(f"  Total passed: {len(self.passed_operators)}/{total_operators}")
            logger.info(f"  Pass rate: {round_result['pass_rate']:.2%}")
        
        # Final summary
        self.print_final_summary(total_operators, max_rounds)
    
    def save_results(self) -> None:
        """Save current results to JSON."""
        results_file = self.output_dir / "pass_at_k_results.json"
        
        # Prepare detailed results with codes
        operator_details = {}
        for op_name in self.generated_codes.keys():
            namespace, kernel_name = op_name.rsplit("::", 1) if "::" in op_name else ("", op_name)
            operator_details[op_name] = {
                "passed": op_name in self.passed_operators,
                "namespace": namespace,
                "kernel_name": kernel_name,
                "generated_codes_by_round": self.generated_codes[op_name],
                "first_pass_round": self.first_pass_round.get(op_name, None),
                "total_attempts": len(self.generated_codes[op_name])
            }
        
        results = {
            "total_operators": sum(len(ops) for ops in self.all_operators.values()),
            "total_passed": len(self.passed_operators),
            "final_pass_rate": len(self.passed_operators) / sum(len(ops) for ops in self.all_operators.values()) if sum(len(ops) for ops in self.all_operators.values()) > 0 else 0,
            "rounds_summary": self.results_by_round,
            "generation_summaries": self.generation_summaries,
            "passed_operators": sorted(list(self.passed_operators)),
            "operator_details": operator_details,
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
    
    def print_final_summary(self, total_operators: int, max_rounds: int) -> None:
        """Print final summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Pass@K Testing Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total operators: {total_operators}")
        logger.info(f"Total passed: {len(self.passed_operators)}")
        logger.info(f"Final pass rate: {len(self.passed_operators) / total_operators:.2%}")
        
        logger.info(f"\nGeneration Statistics:")
        for gen_summary in self.generation_summaries:
            stats = gen_summary["statistics"]
            logger.info(f"  Round {gen_summary['round']}:")
            logger.info(f"    Generated: {stats['successful']}/{stats['total_apis']}")
            logger.info(f"    Success Rate: {stats['success_rate']:.2%}")
        
        logger.info(f"\nVerification Statistics (Pass@K):")
        for i, round_result in enumerate(self.results_by_round, 1):
            logger.info(f"  Pass@{i}: {round_result['pass_rate']:.2%} ({round_result['total_passed']}/{total_operators})")
        
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test Pass@K for operator generation")
    
    parser.add_argument("--name", type=str, default="all", help="Namespace to test (default: all)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "pass_at_k", help="Output directory")
    parser.add_argument("--test-type", type=str, default="accuracy", choices=["accuracy", "performance"])
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds (default: 10)")
    parser.add_argument("--device-count", type=int, default=8, help="Number of devices for testing")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for each test")
    
    # Generation config
    parser.add_argument("--server-type", type=str, default="panda")
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=150)
    
    args = parser.parse_args()
    
    # Create output directory
    run_name = f"pass_at_{args.max_rounds}_{args.model_name}_{args.test_type}_{today()}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create generation config
    gen_config = GenerationConfig(
        run_name=run_name,
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=1,
        verbose=False,
        run_dir=str(output_dir),
        log_prompt=True,
    )
    
    # Create verification config
    verify_config = VerifyConfig(
        run_name=run_name,
        test_type=args.test_type,
        run_dir=str(output_dir / "verification"),
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=args.timeout,
        perf_timeout=args.timeout,
    )
    
    # Create tester and run
    tester = PassAtKTester(
        output_dir=output_dir,
        test_type=args.test_type,
        gen_config=gen_config,
        verify_config=verify_config,
        device_count=args.device_count,
    )
    
    tester.initialize_operators(args.name)
    tester.run_pass_at_k(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()