# coding=utf-8
"""
Setup for gymportal.
"""
from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Experiments for RL with the ACN Research Portal.",
    packages=find_packages(),
    package_data={
        "src": [
            "pv/data/*",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gymportal @ git+https://git.ies.uni-kassel.de/mhassouna/acnsimulation_gym",
        "ray[rllib]==2.10.0",
        "gputil",
        "jupyter",
        "tqdm",
        "grpcio==1.62.0",
        "icecream",
        "tensorflow==2.12.0",
        "torch",
    ],
)
