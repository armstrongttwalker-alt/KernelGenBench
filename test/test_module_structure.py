"""
Test module to verify the directory and file structure of the flagbench repository.
This helps catch issues when refactoring directory names or moving files.
"""
import pytest
import os
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"


class TestDirectoryStructure:
    """Test that key directories exist."""

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        assert SRC_DIR.exists(), f"src directory not found at {SRC_DIR}"
        assert SRC_DIR.is_dir(), f"{SRC_DIR} is not a directory"

    def test_flagbench_package_exists(self):
        """Test that flagbench package directory exists."""
        flagbench_dir = SRC_DIR / "flagbench"
        assert flagbench_dir.exists(), f"flagbench directory not found at {flagbench_dir}"
        assert flagbench_dir.is_dir()
        assert (flagbench_dir / "__init__.py").exists(), "flagbench/__init__.py not found"

    def test_sandbox_package_exists(self):
        """Test that sandbox package directory exists."""
        sandbox_dir = SRC_DIR / "sandbox"
        assert sandbox_dir.exists(), f"sandbox directory not found at {sandbox_dir}"
        assert sandbox_dir.is_dir()
        assert (sandbox_dir / "__init__.py").exists(), "sandbox/__init__.py not found"

    def test_generator_package_exists(self):
        """Test that generator package directory exists."""
        generator_dir = SRC_DIR / "generator"
        assert generator_dir.exists(), f"generator directory not found at {generator_dir}"
        assert generator_dir.is_dir()
        assert (generator_dir / "__init__.py").exists(), "generator/__init__.py not found"


class TestFlagbenchSubmodules:
    """Test flagbench submodule structure."""

    def test_dataset_module_exists(self):
        """Test that dataset module exists with required files."""
        dataset_dir = SRC_DIR / "flagbench" / "dataset"
        assert dataset_dir.exists(), f"dataset directory not found"
        assert dataset_dir.is_dir()
        assert (dataset_dir / "__init__.py").exists()
        assert (dataset_dir / "kernel_list.py").exists()
        assert (dataset_dir / "dataloader.py").exists()
        assert (dataset_dir / "all_operators.json").exists()

    def test_accuracy_module_exists(self):
        """Test that accuracy module exists with test files."""
        accuracy_dir = SRC_DIR / "flagbench" / "accuracy"
        assert accuracy_dir.exists(), f"accuracy directory not found"
        assert accuracy_dir.is_dir()
        assert (accuracy_dir / "__init__.py").exists()

        # Check for some key test files
        expected_files = [
            "test_binary_pointwise_ops.py",
            "test_unary_pointwise_ops.py",
            "test_blas_ops.py",
            "test_reduction_ops.py",
        ]
        for filename in expected_files:
            assert (accuracy_dir / filename).exists(), f"{filename} not found in accuracy module"

    def test_perfermance_module_exists(self):
        """Test that perfermance (typo) module exists.

        Note: This directory is currently named 'perfermance' (typo).
        When renaming to 'performance', this test will fail and should be updated.
        """
        perfermance_dir = SRC_DIR / "flagbench" / "perfermance"
        assert perfermance_dir.exists(), (
            f"perfermance directory not found. "
            f"If you renamed it to 'performance', update this test."
        )
        assert perfermance_dir.is_dir()
        assert (perfermance_dir / "__init__.py").exists()

    def test_perfermance_module_files(self):
        """Test that perfermance module contains required files."""
        perfermance_dir = SRC_DIR / "flagbench" / "perfermance"

        required_files = [
            "performance_utils.py",
            "attri_util.py",
            "core_shapes.yaml",
        ]

        for filename in required_files:
            file_path = perfermance_dir / filename
            assert file_path.exists(), f"{filename} not found in perfermance module"

    def test_processing_module_exists(self):
        """Test that processing module exists."""
        processing_dir = SRC_DIR / "flagbench" / "processing"
        assert processing_dir.exists(), f"processing directory not found"
        assert processing_dir.is_dir()


class TestSandboxSubmodules:
    """Test sandbox submodule structure."""

    def test_verifier_module_exists(self):
        """Test that verifier module exists with required files."""
        verifier_dir = SRC_DIR / "sandbox" / "verifier"
        assert verifier_dir.exists(), f"verifier directory not found"
        assert verifier_dir.is_dir()
        assert (verifier_dir / "__init__.py").exists()
        assert (verifier_dir / "verifier.py").exists()
        assert (verifier_dir / "test_parametrize.py").exists()
        assert (verifier_dir / "utils.py").exists()

    def test_register_module_exists(self):
        """Test that register.py exists."""
        register_file = SRC_DIR / "sandbox" / "register.py"
        assert register_file.exists(), f"register.py not found"

    def test_utils_module_exists(self):
        """Test that utils module exists."""
        utils_dir = SRC_DIR / "sandbox" / "utils"
        assert utils_dir.exists(), f"utils directory not found"
        assert utils_dir.is_dir()
        assert (utils_dir / "accuracy_utils.py").exists()


class TestGeneratorFiles:
    """Test generator module files."""

    def test_generator_files_exist(self):
        """Test that all generator files exist."""
        generator_dir = SRC_DIR / "generator"

        expected_files = [
            "generator.py",
            "triton_kernel_generator.py",
            "torch_kernel_generator.py",
            "test_func_generator.py",
            "benchmark_func_generator.py",
        ]

        for filename in expected_files:
            file_path = generator_dir / filename
            assert file_path.exists(), f"{filename} not found in generator module"


class TestScriptsDirectory:
    """Test scripts directory structure."""

    def test_scripts_directory_exists(self):
        """Test that scripts directory exists."""
        scripts_dir = PROJECT_ROOT / "scripts"
        assert scripts_dir.exists(), f"scripts directory not found"
        assert scripts_dir.is_dir()

    def test_key_scripts_exist(self):
        """Test that key execution scripts exist."""
        scripts_dir = PROJECT_ROOT / "scripts"

        expected_scripts = [
            "generate_sample.py",
            "utils.py",
        ]

        for script_name in expected_scripts:
            script_path = scripts_dir / script_name
            assert script_path.exists(), f"{script_name} not found in scripts directory"


class TestConfigFiles:
    """Test configuration files exist."""

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        requirements_file = PROJECT_ROOT / "requirements.txt"
        assert requirements_file.exists(), "requirements.txt not found"

    def test_setup_file_exists(self):
        """Test that setup.py exists."""
        setup_file = PROJECT_ROOT / "setup.py"
        assert setup_file.exists(), "setup.py not found"

    def test_core_shapes_yaml_exists(self):
        """Test that core_shapes.yaml exists in perfermance module."""
        core_shapes = SRC_DIR / "flagbench" / "perfermance" / "core_shapes.yaml"
        assert core_shapes.exists(), "core_shapes.yaml not found"


class TestTestDirectory:
    """Test the test directory structure."""

    def test_test_directory_exists(self):
        """Test that test directory exists."""
        test_dir = PROJECT_ROOT / "test"
        assert test_dir.exists(), f"test directory not found"
        assert test_dir.is_dir()

    def test_existing_test_files(self):
        """Test that existing test files are present."""
        test_dir = PROJECT_ROOT / "test"

        expected_tests = [
            "test_verifier_operator.py",
            "test_verifier_test_func.py",
            "test_accuracy_ut.py",
            "test_verifier_benchmark.py",
            "test_fused_operator.py",
        ]

        for test_file in expected_tests:
            test_path = test_dir / test_file
            assert test_path.exists(), f"{test_file} not found in test directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
