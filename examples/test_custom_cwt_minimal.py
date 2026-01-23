#!/usr/bin/env python3
"""
Minimal test for custom_cwt development - no file I/O, just function testing.

This is the fastest way to test changes to custom_wavelet.py during development.

Usage:
    python examples/test_custom_cwt_minimal.py
"""

import sys
from pathlib import Path
import numpy as np
import mne

# Add src to path for development (no need to reinstall)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eeg_processor.processing.custom_wavelet import compute_custom_cwt_tfr
from loguru import logger


def generate_test_epochs(n_epochs=20, n_channels=10, sfreq=500, duration=1.0):
    """Generate minimal synthetic epochs for fast testing."""
    logger.info(f"Generating test epochs: {n_epochs} epochs, {n_channels} channels")

    n_times = int(sfreq * duration)
    times = np.arange(n_times) / sfreq

    # Simple synthetic data with 10 Hz oscillation
    data = np.zeros((n_epochs, n_channels, n_times))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # 10 Hz + noise
            signal = np.sin(2 * np.pi * 10 * times + np.random.rand() * 2 * np.pi)
            noise = np.random.randn(n_times) * 0.2
            data[epoch_idx, ch_idx, :] = signal + noise

    # Create minimal info
    ch_names = [f'CH{i+1}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create events
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs) * n_times
    events[:, 2] = 1  # All same event type for simplicity
    event_id = {'test': 1}

    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id,
                             tmin=-0.2, baseline=None, verbose=False)

    return epochs


def test_custom_cwt():
    """Test custom_cwt_tfr function directly."""
    logger.info("="*60)
    logger.info("Testing compute_custom_cwt_tfr()")
    logger.info("="*60)

    # Generate test data
    epochs = generate_test_epochs(n_epochs=20, n_channels=10)
    logger.info(f"Test epochs: {epochs}")

    # Test 1: Basic Morse wavelet analysis
    logger.info("\nTest 1: Basic Morse wavelet (gamma=3, beta=3)")
    try:
        power = compute_custom_cwt_tfr(
            epochs,
            wavelet_type="morse",
            freq_range=[4, 40],
            n_freqs=20,
            morse_gamma=3.0,
            morse_beta=3.0,
            compute_itc=False,
            baseline=None,
            pad_len=5  # Shorter padding for speed
        )
        logger.success(f"✓ Test 1 passed: {power}")
        logger.info(f"  Shape: {power.data.shape}")
        logger.info(f"  Frequencies: {power.freqs[0]:.1f} - {power.freqs[-1]:.1f} Hz")
    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        raise

    # Test 2: With ITC computation
    logger.info("\nTest 2: With ITC computation")
    try:
        power = compute_custom_cwt_tfr(
            epochs,
            wavelet_type="morse",
            freq_range=[4, 40],
            n_freqs=20,
            morse_gamma=3.0,
            morse_beta=3.0,
            compute_itc=True,
            baseline=None,
            pad_len=5
        )
        logger.success(f"✓ Test 2 passed")
        if hasattr(power, '_itc_data'):
            logger.info(f"  ITC data present: {power._itc_data.data.shape}")
        else:
            logger.warning("  ITC data not found")
    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        raise

    # Test 3: With baseline correction
    logger.info("\nTest 3: With baseline correction")
    try:
        power = compute_custom_cwt_tfr(
            epochs,
            wavelet_type="morse",
            freq_range=[4, 40],
            n_freqs=20,
            morse_gamma=3.0,
            morse_beta=3.0,
            compute_itc=False,
            baseline=(-0.2, 0),
            baseline_mode="mean",
            pad_len=5
        )
        logger.success(f"✓ Test 3 passed")
    except Exception as e:
        logger.error(f"✗ Test 3 failed: {e}")
        raise

    # Test 4: High temporal resolution (gamma=6)
    logger.info("\nTest 4: High temporal resolution (gamma=6)")
    try:
        power = compute_custom_cwt_tfr(
            epochs,
            wavelet_type="morse",
            freq_range=[4, 40],
            n_freqs=20,
            morse_gamma=6.0,  # Higher gamma
            morse_beta=3.0,
            compute_itc=False,
            baseline=None,
            pad_len=5
        )
        logger.success(f"✓ Test 4 passed")
    except Exception as e:
        logger.error(f"✗ Test 4 failed: {e}")
        raise

    # Test 5: High frequency resolution (beta=10)
    logger.info("\nTest 5: High frequency resolution (beta=10)")
    try:
        power = compute_custom_cwt_tfr(
            epochs,
            wavelet_type="morse",
            freq_range=[4, 40],
            n_freqs=20,
            morse_gamma=3.0,
            morse_beta=10.0,  # Higher beta
            compute_itc=False,
            baseline=None,
            pad_len=5
        )
        logger.success(f"✓ Test 5 passed")
    except Exception as e:
        logger.error(f"✗ Test 5 failed: {e}")
        raise

    logger.success("="*60)
    logger.success("ALL TESTS PASSED!")
    logger.success("="*60)


if __name__ == "__main__":
    logger.info("Starting minimal custom_cwt test...\n")
    test_custom_cwt()
