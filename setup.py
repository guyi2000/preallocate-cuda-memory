from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="preallocate_cuda_memory",
    version="1.0.0",
    author="Yi Gu",
    author_email="guyi2000@yeah.net",
    description="preallocate CUDA memory for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guyi2000/preallocate-cuda-memory",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
