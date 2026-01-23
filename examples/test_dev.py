#!/usr/bin/env python3
"""
Quick development test script for rapid iteration.

This script generates synthetic EEG data and runs a test pipeline to quickly
verify changes without needing to reinstall the package or run full analyses.

Usage:
    python examples/test_dev.py

    # Or with custom config
    python examples/test_dev.py --config examples/configs/test_custom_cwt.yml
"""

import sys
from pathlib import Path
import numpy as np
import mne
from mne.datasets import testing
import tempfile
import shutil

# Add src to path for development (no need to reinstall)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eeg_processor.pipeline import EEGPipeline
from loguru import logger


def generate_synthetic_epochs(n_epochs=20, n_channels=32, sfreq=500, duration=1.0):
    """
    Generate synthetic EEG epochs for testing.

    Args:
        n_epochs: Number of epochs to generate
        n_channels: Number of EEG channels
        sfreq: Sampling frequency in Hz
        duration: Epoch duration in seconds

    Returns:
        MNE Epochs object with synthetic data
    """
    logger.info(f"Generating synthetic EEG data: {n_epochs} epochs, {n_channels} channels, {sfreq} Hz")

    n_times = int(sfreq * duration)

    # Create synthetic data with some structure
    # Add alpha oscillation (10 Hz) + noise
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_epochs, n_channels, n_times))

    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Alpha oscillation with random phase
            phase = np.random.rand() * 2 * np.pi
            alpha = np.sin(2 * np.pi * 10 * times + phase)

            # Add some theta (6 Hz)
            theta = 0.5 * np.sin(2 * np.pi * 6 * times + np.random.rand() * 2 * np.pi)

            # Add noise
            noise = np.random.randn(n_times) * 0.3

            # Combine
            data[epoch_idx, ch_idx, :] = alpha + theta + noise

    # Create MNE info
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Add montage
    montage = mne.channels.make_standard_montage('standard_1020')
    # Only use channels that exist in the montage
    available_channels = [ch for ch in ch_names if ch in montage.ch_names]
    if len(available_channels) < n_channels:
        # Use subset of available channels
        n_channels_available = len(available_channels)
        data = data[:, :n_channels_available, :]
        ch_names = available_channels
        ch_types = ['eeg'] * len(available_channels)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    info.set_montage(montage, on_missing='ignore')

    # Create events
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs) * n_times  # Sample indices
    # Alternate between two event types
    events[:, 2] = np.where(np.arange(n_epochs) % 2 == 0, 1, 2)

    event_id = {'standard': 1, 'target': 2}

    # Create epochs
    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id,
                             tmin=-0.2, baseline=None)

    logger.success(f"Created synthetic epochs: {epochs}")
    return epochs


def save_synthetic_raw(output_dir: Path, participant_id: str = "test-01"):
    """
    Generate and save synthetic raw data as BrainVision format.

    Args:
        output_dir: Directory to save files
        participant_id: Participant identifier

    Returns:
        Path to saved .vhdr file
    """
    logger.info(f"Creating synthetic raw data for {participant_id}")

    # Generate longer continuous data
    n_channels = 32
    sfreq = 500
    duration = 20.0  # 20 seconds
    n_times = int(sfreq * duration)

    # Create synthetic continuous data
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_channels, n_times))

    for ch_idx in range(n_channels):
        # Alpha + theta + noise
        alpha = np.sin(2 * np.pi * 10 * times + np.random.rand() * 2 * np.pi)
        theta = 0.5 * np.sin(2 * np.pi * 6 * times + np.random.rand() * 2 * np.pi)
        noise = np.random.randn(n_times) * 0.3
        data[ch_idx, :] = alpha + theta + noise

    # Create MNE info
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Add montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    # Add events
    events = np.array([
        [500, 0, 1],   # Standard at 1s
        [1500, 0, 2],  # Target at 3s
        [2500, 0, 1],  # Standard at 5s
        [3500, 0, 2],  # Target at 7s
        [4500, 0, 1],  # Standard at 9s
        [5500, 0, 2],  # Target at 11s
        [6500, 0, 1],  # Standard at 13s
        [7500, 0, 2],  # Target at 15s
        [8500, 0, 1],  # Standard at 17s
        [9500, 0, 2],  # Target at 19s
    ])

    # Create annotations from events
    onset = events[:, 0] / sfreq
    duration = np.zeros(len(events))
    description = [str(e) for e in events[:, 2]]
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)

    # Save as FIF format (simpler, no extra dependencies)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{participant_id}_raw.fif"

    raw.save(output_file, overwrite=True, verbose=False)
    logger.success(f"Saved synthetic raw data to {output_file}")

    return output_file


def run_quick_test(config_file: Path = None, cleanup: bool = True):
    """
    Run a quick test of the pipeline with synthetic data.

    Args:
        config_file: Path to config YAML (or None to use default test config)
        cleanup: Whether to delete temporary files after test
    """
    # Create temporary directory for test data
    temp_dir = Path(tempfile.mkdtemp(prefix="eeg_processor_test_"))
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        # Create directories
        raw_data_dir = temp_dir / "raw"
        results_dir = temp_dir / "results"

        # Generate synthetic data
        participant_file = save_synthetic_raw(raw_data_dir, "test-01")

        # Use provided config or create default
        if config_file is None:
            # Create a simple test config
            config_file = temp_dir / "test_config.yml"
            config_content = f"""
study:
  name: "Quick_Dev_Test"
  description: "Rapid development testing"

paths:
  raw_data: "{raw_data_dir}"
  results: "{results_dir}"
  file_extension: ".fif"

participants:
  - "{participant_file.name}"

processing:
  - filter: {{l_freq: 1.0, h_freq: 40.0}}
  - epoch: {{tmin: -0.2, tmax: 0.8, baseline: [-0.2, 0], event_id: {{'1': 1, '2': 2}}}}
  - custom_cwt:
      wavelet_type: "morse"
      freq_range: [4, 40]
      n_freqs: 30
      morse_gamma: 3.0
      morse_beta: 3.0
      compute_itc: true
      baseline: [-0.2, 0]
      baseline_mode: "mean"

conditions:
  - name: "Standard"
    condition_markers: ['1']
  - name: "Target"
    condition_markers: ['2']

output:
  save_intermediates: true
  create_report: false
"""
            config_file.write_text(config_content)
            logger.info(f"Created test config: {config_file}")

        # Run pipeline
        logger.info("="*60)
        logger.info("RUNNING TEST PIPELINE")
        logger.info("="*60)

        pipeline = EEGPipeline(str(config_file))
        pipeline.run()

        logger.success("="*60)
        logger.success("TEST COMPLETED SUCCESSFULLY!")
        logger.success("="*60)

        # Show results
        if results_dir.exists():
            logger.info(f"\nResults saved to: {results_dir}")
            logger.info("Output files:")
            for file in sorted(results_dir.rglob("*")):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    logger.info(f"  {file.relative_to(results_dir)} ({size_kb:.1f} KB)")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        # Cleanup
        if cleanup:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            logger.info(f"Temporary files kept at: {temp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick development test for EEG Processor")
    parser.add_argument("--config", type=Path, help="Path to config file (optional)")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep temporary files")

    args = parser.parse_args()

    logger.info("Starting quick development test...")
    run_quick_test(config_file=args.config, cleanup=not args.no_cleanup)
