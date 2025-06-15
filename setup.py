# EEG_Processor/setup.py
from setuptools import setup, find_packages

setup(
    name="eeg-processor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mne",
        "numpy",
        "loguru",
        "matplotlib",
        "pyyaml",
        "pandas",
        "seaborn",
        "pingouin"
    ],
)