from setuptools import setup, find_packages

setup(
    name='torch-mlp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0'
    ],
    author='Rob Hesselink',
    author_email='rob.hesselink@pm.me',
    description='A PyTorch package for creating multi-layer perceptrons (MLPs).',
    python_requires='>=3.6',
)