# quality_control/__init__.py

"""
Quality Control Module for EEG Processing Pipeline

This module provides comprehensive quality tracking and reporting capabilities
for EEG data processing pipelines.
"""

# Main public API
from .quality_tracker import QualityTracker
from .quality_reporter import QualityReporter, generate_quality_reports

# Internal components (not part of public API)
from .quality_plot_generator import QualityPlotGenerator
from .quality_html_generator import QualityHTMLGenerator
from .pipeline_detector import PipelineDetector
from .quality_flagging import QualityFlagger

# Legacy functions for backward compatibility
from .quality_tracker import (
    extract_bad_channel_metrics,
    extract_epoch_metrics,
    extract_ica_metrics
)

__version__ = "1.0.0"
__all__ = [
    "QualityTracker",
    "QualityReporter", 
    "generate_quality_reports",
    "extract_bad_channel_metrics",
    "extract_epoch_metrics",
    "extract_ica_metrics"
]