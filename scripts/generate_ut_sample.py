#!/usr/bin/env python3
"""
Generate sample script for creating Triton kernels from PyTorch APIs.

This script reads PyTorch APIs from the kernel list in src/flagbench/dataset,
generates TritonKernelGenerateArgs for each API, and uses the triton_kernel_generator
to generate corresponding Triton implementations.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from flagbench.dataset import TorchOpsLoader, APIInfo

def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


from generator.test_func_generator import TestFuncGenerator
from generator.sampler.generate_samples import (
    TestFuncGenerateArgs,
    GenerationConfig,
    InputArg,
    OutputArg,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ut_generate_args(torch_op_name: str, operators: APIInfo) -> TestFuncGenerateArgs:
    """
    Create TestFuncGenerateArgs for a given PyTorch operator.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        TritonKernelGenerateArgs instance
    """
    # Extract the function name from the full path
    # e.g., 'torch.add' -> 'add', 'torch.nn.functional.gelu' -> 'gelu'
    kernel_name = torch_op_name.split('.')[-1]
    
    return TestFuncGenerateArgs(
        kernel_name=kernel_name,
        operators=operators.schemas,
        test_func_name=f"test_accuracy_{operators.namespace}_{kernel_name}",
        ops_namespace=operators.namespace,
    )


def generate_samples(name: str, output_dir: Path, config: GenerationConfig) -> None:
    """
    Generate Triton kernel samples for the specified APIs.
    
    Args:
        name: Name of the API to generate, or "all" for all APIs
        output_dir: Directory to save generated results
        config: Generation configuration
    """

    operator_loader = TorchOpsLoader()

    # Get the list of APIs to process
    if name.lower() == "all":
        namespace_to_process = operator_loader.load_all()
        logger.info(f"Processing all {len(namespace_to_process)} PyTorch Namespaces.")
    else:
        # Check if the specified name exists
        if name not in operator_loader.list_namespaces():
            logger.error(f"Namespace '{name}' not found in the operator list.")
            return
        namespace_to_process = {name: operator_loader.load_namespace(namespace=name)}
        logger.info(f"Processing namespace: {name}")
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(config.num_samples):
        code_dir = output_dir / f"ut_{sample_idx}"
        code_dir.mkdir(exist_ok=True)
    
    # Initialize the generator
    generator = TestFuncGenerator(config)
    
    # Process each API and collect generate args
    gen_args = []
    api_names = []
    skipped_results = []  # Track skipped APIs for summary
    
    for namespace, apis_to_process in namespace_to_process.items():
        logger.info(f"Processing Namespace: {namespace} with {len(apis_to_process)} APIs")
        for api_name, operators in apis_to_process.items():
            logger.info(f"Preparing: {api_name}")
            try:
                # Create generate args for this API
                for sample_idx in range(config.num_samples):
                    gen_arg = create_ut_generate_args(api_name, operators)
                    gen_arg.sample_id = config.sample_id + sample_idx
                    gen_args.append(gen_arg)
                    api_names.append(api_name)
            except Exception as e:
                logger.error(f"✗ Error preparing {api_name}: {e}", exc_info=True)
                skipped_results.append({
                    "namespace": namespace,
                    "api_name": api_name,
                    "success": False,
                    "error": f"Preparation error: {str(e)}",
                    "skipped": True
                })
    
    # Generate all Triton kernels
    logger.info(f"Generating {len(gen_args)} Triton kernels...")
    generated_codes = generator(gen_args)
    
    # Verify generated_codes is a list
    if not isinstance(generated_codes, list):
        logger.error(f"Expected list of generated codes, got {type(generated_codes)}")
        generated_codes = [generated_codes] if generated_codes else []
    
    # Process and save the generated codes
    results = []
    # breakpoint()
    for idx, (generated_code, name, sample_id) in enumerate(generated_codes):
        logger.info(f"Processing result {idx + 1}/{len(api_names)}: {name}")
        
        # Check if generation was successful
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
            try:
                # Save to file
                kernel_filename = f"{name}.py"
                kernel_path = output_dir / f"ut_{sample_id}" / kernel_filename
                
                with open(kernel_path, "w") as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Generated kernel saved to: {kernel_path}")
                
                # Record successful result
                results.append({
                    "namespace": namespace,
                    "api_name": api_name,
                    "file_path": str(kernel_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {api_name}: {e}")
                results.append({
                    "namespace": namespace,
                    "api_name": api_name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate code for {api_name}")
            results.append({
                "namespace": namespace,
                "api_name": api_name,
                "success": False,
                "error": "Empty or invalid generation result",
            })
    
    # Combine all results (generated + skipped)
    all_results = results + skipped_results
    
    # Calculate statistics
    total = len(api_names) + len(skipped_results)
    successful = sum(1 for r in results if r["success"])
    failed_generation = len(api_names) - successful
    skipped = len(skipped_results)
    
    # Save detailed summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "attempted": len(api_names),
            "skipped": skipped,
            "successful": successful,
            "failed": failed_generation,
            "success_rate": f"{successful / len(api_names) * 100:.2f}%" if len(api_names) > 0 else "0%",
            "overall_rate": f"{successful / total * 100:.2f}%" if total > 0 else "0%",
            "results": all_results,
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Summary:")
    logger.info(f"Total APIs: {total}")
    logger.info(f"Skipped (verification failed): {skipped}")
    logger.info(f"Attempted: {len(api_names)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed_generation}")
    logger.info(f"Success Rate (attempted): {successful / len(api_names) * 100:.2f}%" if len(api_names) > 0 else "0%")
    logger.info(f"Overall Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton kernels from PyTorch APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all APIs
  python script/generate_sample.py --name all
  
  # Generate for a specific API
  python script/generate_sample.py --name torch.add
  
  # Use custom output directory
  python script/generate_sample.py --name all --output-dir ./my_output
  
  # Use different model
  python script/generate_sample.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the PyTorch API to generate (default: all). Use 'all' to generate for all APIs."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Output directory for generated samples (default: output)"
    )
    
    parser.add_argument(
        "--server-type",
        type=str,
        default="panda",
        choices=["qwen", "deepseek", "openai", "anthropic", "google", "together", "sglang", "vllm", "panda"],
        help="LLM server type to use (default: panda)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="Model name to use (default: gpt-4o-mini)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=150,
        help="Number of parallel workers (default: 1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_name = f"ut_{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
    # Create generation config
    config = GenerationConfig(
        run_name=run_name,
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        verbose=args.verbose,
        run_dir=str(args.output_dir),
        log_prompt=True,
    )
    
    logger.info("Starting sample generation...")
    logger.info(f"Config: {config}")
    output_dir = args.output_dir / run_name
    # Generate samples
    generate_samples(args.name, output_dir, config)
    
    logger.info("Sample generation completed!")


if __name__ == "__main__":
    main()
