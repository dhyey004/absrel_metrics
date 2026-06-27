from setuptools import setup, find_packages

setup(
    name="absrel_metrics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0"
    ],
    author="Dhyey Pandya",
    description="GPU-accelerated Absolute Relative Error metric for PyTorch",
    python_requires=">=3.7",
)