#!/usr/bin/env python3
"""Setup script for CLAIRE"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="claire-md",
    version="1.0.0",
    author="Takeshi Sato",
    author_email="takeshi@mb.kyoto-phu.ac.jp",
    description="Composition-based Lipid Analysis with Integrated Resolution and Enrichment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/takeshi-sato-dev/claire",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "claire=run_claire:main",
        ],
    },
)
