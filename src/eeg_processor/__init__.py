"""
EEG Processor: A comprehensive EEG processing pipeline with quality control.

This package provides tools for processing EEG data from multiple formats,
with built-in quality control and reporting capabilities.
"""

__version__ = "0.1.0"
__author__ = "EEG Processor Contributors"

# Configure logging to be less intrusive for library usage
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API exports
from .pipeline import EEGPipeline
from .utils.config_loader import load_config, PipelineConfig

# Quality control exports
from .quality_control.quality_reporter import generate_quality_reports

# Optional CLI - only export if click is available
try:
    from .cli import cli
    __all__ = ["EEGPipeline", "load_config", "PipelineConfig", "generate_quality_reports", "cli"]
except ImportError:
    __all__ = ["EEGPipeline", "load_config", "PipelineConfig", "generate_quality_reports"]