"""
Test module to verify all critical imports work correctly.
This test is especially important for catching issues when refactoring
module names or directory structures (e.g., renaming 'perfermance' to 'performance').
"""
import pytest
import sys
import importlib


class TestCoreImports:
    """Test core flagbench package imports."""

    def test_main_package_import(self):
        """Test that the main flagbench package can be imported."""
        import flagbench
        assert flagbench is not None

    def test_core_exports(self):
        """Test that core exports are available from flagbench."""
        from flagbench import enable, use_gems, register
        assert callable(enable)
        assert use_gems is not None
        assert callable(register)

    def test_dataset_imports(self):
        """Test dataset module imports."""
        from flagbench.dataset.kernel_list import PYTORCH_OPERATORS
        from flagbench.dataset.dataloader import TorchOpsLoader, OperatorLoader

        assert isinstance(PYTORCH_OPERATORS, dict)
        assert TorchOpsLoader is not None
        assert OperatorLoader is not None


class TestPerformanceImports:
    """Test performance (perfermance) module imports.

    Note: Currently the directory is named 'perfermance' (typo).
    When renaming to 'performance', these tests will fail and need updating.
    """

    def test_performance_utils_import(self):
        """Test performance_utils module import."""
        from flagbench.perfermance.performance_utils import GenericBenchmark
        assert GenericBenchmark is not None

    def test_attri_util_import(self):
        """Test attri_util module import."""
        from flagbench.perfermance.attri_util import BenchmarkResult
        assert BenchmarkResult is not None

    def test_performance_test_modules(self):
        """Test that performance test modules can be imported."""
        perf_modules = [
            "flagbench.perfermance.test_attention_perf",
            "flagbench.perfermance.test_binary_pointwise_perf",
            "flagbench.perfermance.test_blas_perf",
            "flagbench.perfermance.test_unary_pointwise_perf",
        ]

        for module_name in perf_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


class TestAccuracyImports:
    """Test accuracy module imports."""

    def test_accuracy_test_modules(self):
        """Test that accuracy test modules can be imported."""
        accuracy_modules = [
            "flagbench.accuracy.test_binary_pointwise_ops",
            "flagbench.accuracy.test_unary_pointwise_ops",
            "flagbench.accuracy.test_blas_ops",
            "flagbench.accuracy.test_reduction_ops",
        ]

        for module_name in accuracy_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


class TestSandboxImports:
    """Test sandbox module imports."""

    def test_verifier_imports(self):
        """Test verifier module imports."""
        from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

        assert Verifier is not None
        assert VerifyConfig is not None
        assert VerifyRequest is not None
        assert Source is not None

    def test_register_imports(self):
        """Test register module imports."""
        from sandbox.register import Register, register, REGISTERED_OPS

        assert Register is not None
        assert callable(register)
        assert isinstance(REGISTERED_OPS, dict)

    def test_parametrize_imports(self):
        """Test parametrize module imports."""
        from sandbox.verifier.test_parametrize import parametrize, label, Param

        assert callable(parametrize)
        assert callable(label)
        assert Param is not None


class TestGeneratorImports:
    """Test generator module imports."""

    def test_base_generator_import(self):
        """Test base generator import."""
        from generator.generator import BaseGenerator
        assert BaseGenerator is not None

    def test_triton_kernel_generator_import(self):
        """Test Triton kernel generator import."""
        from generator.triton_kernel_generator import TritonKernelGenerator
        assert TritonKernelGenerator is not None

    def test_torch_kernel_generator_import(self):
        """Test Torch kernel generator import."""
        from generator.torch_kernel_generator import TorchKernelGenerator
        assert TorchKernelGenerator is not None

    def test_test_func_generator_import(self):
        """Test test function generator import."""
        from generator.test_func_generator import TestFuncGenerator
        assert TestFuncGenerator is not None

    def test_benchmark_func_generator_import(self):
        """Test benchmark function generator import."""
        from generator.benchmark_func_generator import BenchmarkFuncGenerator
        assert BenchmarkFuncGenerator is not None


class TestCrossModuleImports:
    """Test imports that span multiple modules.

    These tests verify that modules can import from each other correctly,
    which is important for catching circular dependency issues.
    """

    def test_verifier_imports_performance_utils(self):
        """Test that verifier can import from performance module."""
        # This import is used in verifier.py
        from flagbench.perfermance.attri_util import BenchmarkResult
        assert BenchmarkResult is not None

    def test_generator_imports_performance_utils(self):
        """Test that generator can import from performance module."""
        # This import is used in benchmark_func_generator.py
        from flagbench.perfermance.performance_utils import GenericBenchmark
        assert GenericBenchmark is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
