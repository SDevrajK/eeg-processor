# EEG_Processor/setup.py
from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A comprehensive EEG processing pipeline with quality control and reporting"

setup(
    name="eeg-processor",
    version="0.2.2",
    author="Sayeed Devraj-Kizuk",  
    author_email="sayeed.kizuk64@gmail.com", 
    description="A comprehensive EEG processing pipeline with quality control and reporting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SDevrajK/eeg-processor", 
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",  
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mne>=1.7.0",
        "numpy>=1.21.0",
        "loguru>=0.7.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "pandas>=1.5.0",
        "seaborn>=0.11.0",
        "pingouin>=0.5.0",
        "asrpy>=0.0.3",  # Artifact Subspace Reconstruction
        "jsonschema>=4.0.0",  # JSON schema validation
    ],
    extras_require={
        "cli": ["click>=8.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eeg-processor=eeg_processor.cli:cli",
        ],
    },
)