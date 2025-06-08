from pathlib import Path
from src.eeg_processor.pipeline import EEGPipeline
import matplotlib.pyplot as plt
import mne
import numpy as np


def run_test_pipeline(config_path):
    pipeline = EEGPipeline(config_path)
    pipeline.run()

    # Generate quality reports
    pipeline.generate_quality_reports()

if __name__ == "__main__":
    #config_path = Path(__file__).parent.parent / "test_config/RIEEG_baseline_singlesubject_test_processing_params.yml"
    config_path = Path(__file__).parent.parent / "test_config/IURDPM_singlesubject_processing_params.yml"
    run_test_pipeline(config_path)
