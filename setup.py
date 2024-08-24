from setuptools import setup, find_packages

setup(
    name="torch-mlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Do not specify torch here
    extras_require={"test": ["pytest"]},
    author="RobDHess",
    description="A PyTorch package for creating multi-layer perceptrons (MLPs).",
    python_requires=">=3.6",
)
