from setuptools import setup, find_packages

setup(
    name="test_verifier",
    version="0.1",  # 与 __init__.py 中的版本一致
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8.0",
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "typer",
        "PyYAML",
        "scipy",
    ],
    # entry_points={
    #     "console_scripts": [
    #         "benchserve=bench.cli:app",
    #     ],
    # },
)