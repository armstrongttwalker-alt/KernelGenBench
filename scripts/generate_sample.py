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
from typing import List, Dict, Any, Callable
from datetime import datetime
import torch
from utils import load_api_to_process_from_test_func_path, get_function_signature, today


# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from flagbench.dataset.kernel_list import IMPL_INFO
from generator.triton_kernel_generator import TritonKernelGenerator
from generator.sampler.generate_samples import (
    TritonKernelGenerateArgs,
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


def get_torch_api_signature(torch_op_name: str, torch_op_func) -> Dict[str, Any]:
    """
    Extract signature information from a PyTorch API.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        Dictionary containing input_args, output_args, and func_desc
    """
    # Try to get the docstring for description
    func_desc = f"PyTorch operator: {torch_op_name}"
    if hasattr(torch_op_func, '__doc__') and torch_op_func.__doc__:
        # Extract first line of docstring as description
        doc_lines = torch_op_func.__doc__.strip().split('\n')
        if doc_lines:
            func_desc = doc_lines[0].strip()
    
    # For now, we'll use generic input/output args since we don't have
    # detailed parameter information without runtime inspection
    # In a real implementation, you might want to use inspect module or
    # parse the docstring to extract parameter information
    
    # TODO : Improve argument extraction logic
    input_args, output_args, _ = get_function_signature(torch_op_func)
    # input_args = [InputArg(arg_name="args", arg_type="Any", arg_desc="Input arguments")]
    # output_args = [OutputArg(arg_type="Any", arg_desc="Output result")]
    
    return {
        "input_args": input_args,
        "output_args": output_args,
        "func_desc": func_desc,
    }


def create_triton_generate_args(torch_op_name: str, torch_op_func: Callable | str, impl_info) -> TritonKernelGenerateArgs:
    """
    Create TritonKernelGenerateArgs for a given PyTorch operator.
    
    Args:
        torch_op_name: Full name of the torch operator (e.g., 'torch.add')
        torch_op_func: The actual torch function object
    
    Returns:
        TritonKernelGenerateArgs instance
    """
    # Extract the function name from the full path
    # e.g., 'torch.add' -> 'add', 'torch.nn.functional.gelu' -> 'gelu'
    kernel_name = torch_op_name.split('.')[-1]
    
    # Get signature information
    if isinstance(torch_op_func, str):
        # check torch.ops has the attribute
        if hasattr(torch.ops, torch_op_func):
            # torch_op_func actually is the namespace
            torch_op_name = f"{torch_op_func}::{torch_op_name}"
            torch_op_namespace = getattr(torch.ops, torch_op_func)
            torch_op_func = getattr(torch_op_namespace, kernel_name)
            torch_op_func_name = f"{torch_op_namespace.__name__}.{kernel_name}"
    sig_info = get_torch_api_signature(torch_op_name, torch_op_func)
    
    # Create a simple torch kernel code snippet as reference
    torch_kernel_code = f"""
# Reference PyTorch implementation for {torch_op_name}
import torch

def {kernel_name}(*args, **kwargs):
    return {torch_op_func_name}(*args, **kwargs)
""".strip()
    
    return TritonKernelGenerateArgs(
        triton_kernel_name=torch_op_name,
        func_desc=sig_info["func_desc"],
        torch_kernel_code=torch_kernel_code,
        input_args=sig_info["input_args"],
        output_args=sig_info["output_args"],
        impl_info=impl_info,
        # func_type="other",  # Could be refined based on operator analysis
        from_mcp=False,
    )


def generate_samples(name: str, output_dir: Path, config: GenerationConfig, test_func_result_path: Path | None = None) -> None:
    """
    Generate Triton kernel samples for the specified APIs.
    
    Args:
        name: Name of the API to generate, or "all" for all APIs
        output_dir: Directory to save generated results
        config: Generation configuration
    """
    if test_func_result_path:
        logger.info(f"Loading test function results from: {test_func_result_path}")
        PYTORCH_OPERATORS = load_api_to_process_from_test_func_path(test_func_result_path)
    else:
        from flagbench.dataset.kernel_list import PYTORCH_OPERATORS
    # Get the list of APIs to process
    if name.lower() == "all":
        apis_to_process = PYTORCH_OPERATORS
        logger.info(f"Processing all {len(apis_to_process)} PyTorch APIs")
    else:
        # Check if the specified name exists
        if name not in PYTORCH_OPERATORS:
            logger.error(f"API '{name}' not found in kernel list")
            logger.info(f"Available APIs: {list(PYTORCH_OPERATORS.keys())}")
            return
        apis_to_process = {name: PYTORCH_OPERATORS[name]}
        logger.info(f"Processing single API: {name}")
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(config.num_samples):
        code_dir = output_dir / f"code_{sample_idx}"
        code_dir.mkdir(exist_ok=True)
    
    # Initialize the generator
    generator = TritonKernelGenerator(config)
    
    # Process each API and collect generate args
    gen_args = []
    api_names = []
    
    # FIXME: DynamicImplInfo do not have items method !
    for api_name, namespace_or_api_func in apis_to_process.items():
        logger.info(f"Preparing: {api_name}")
        api_name_ = api_name.split('.')[-1]
        if isinstance(namespace_or_api_func, str):
            total_name = f"{namespace_or_api_func}::{api_name}"
        try:
            assert total_name in IMPL_INFO, f"Implementation info not found for {api_name_}"
            # Create generate args for this API
            for sample_idx in range(config.num_samples):
                gen_arg = create_triton_generate_args(api_name, namespace_or_api_func, IMPL_INFO[total_name])
                gen_arg.sample_id = config.sample_id + sample_idx
                gen_args.append(gen_arg)
                api_names.append(api_name)
        except Exception as e:
            logger.error(f"✗ Error preparing {api_name}: {e}", exc_info=True)
    
    # Generate all Triton kernels
    logger.info(f"Generating {len(gen_args)} Triton kernels...")
    generated_codes = generator(gen_args)
    
    # Verify generated_codes is a list
    if not isinstance(generated_codes, list):
        logger.error(f"Expected list of generated codes, got {type(generated_codes)}")
        generated_codes = [generated_codes] if generated_codes else []
    
    # Process and save the generated codes
    results = []
    for idx, (generated_code, name, sample_id) in enumerate(generated_codes):
        logger.info(f"Processing result {idx + 1}/{len(api_names)}: {name}")
        
        # Check if generation was successful
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
            try:
                # Save to file
                kernel_filename = f"{name}.py"
                kernel_path = output_dir / f"code_{sample_id}" / kernel_filename
                
                with open(kernel_path, "w") as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Generated kernel saved to: {kernel_path}")
                
                # Record successful result
                results.append({
                    "api_name": api_name,
                    "kernel_name": gen_arg.triton_kernel_name,
                    "file_path": str(kernel_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {api_name}: {e}")
                results.append({
                    "api_name": api_name,
                    "kernel_name": gen_arg.triton_kernel_name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate code for {api_name}")
            results.append({
                "api_name": api_name,
                "kernel_name": gen_arg.triton_kernel_name,
                "success": False,
                "error": "Empty or invalid generation result",
            })
    
    # Calculate statistics
    total = len(api_names)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    
    # Save detailed summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{successful / total * 100:.2f}%" if total > 0 else "0%",
            "results": results,
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Summary:")
    logger.info(f"Total APIs: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
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
        "--test-func-result-path",
        type=Path,
        default=None,
        help="Path to the test function result file"
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
        default="deepseek-v3-0324",
        help="Model name to use (default: deepseek-v3-0324)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.6,
        help="Temperature for generation (default: 1.6)"
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
    run_name = f"{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
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
    # test_func_result_path = args.test_func_result_path / "result.json" if args.test_func_result_path.name != "result.json" else args.test_func_result_path
    generate_samples(args.name, output_dir, config, test_func_result_path=args.test_func_result_path)
    
    logger.info("Sample generation completed!")


if __name__ == "__main__":
    main()
