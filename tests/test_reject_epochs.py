"""
Tests for reject_epochs stage.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import mne
from mne import create_info, Epochs
from mne.io import RawArray

from src.eeg_processor.processing.reject_epochs import reject_bad_epochs


@pytest.fixture
def mock_epochs_with_artifacts():
    """Create mock Epochs with clean and artifact epochs."""
    n_channels = 4
    sfreq = 250.0
    tmin = -0.2
    tmax = 0.6

    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw and Epochs first to get correct dimensions
    raw = RawArray(np.random.randn(n_channels, 3000) * 1e-5, info, verbose=False)
    events = np.array([[400, 0, 1], [700, 0, 1], [1000, 0, 1],
                       [1300, 0, 1], [1600, 0, 1], [1900, 0, 1]])

    epochs = Epochs(
        raw,
        events=events,
        event_id={'stimulus': 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Get the actual dimensions from MNE
    n_epochs, n_channels, n_times = epochs._data.shape
    t = epochs.times

    # Create controlled signals with known PTP
    data = np.zeros((n_epochs, n_channels, n_times))

    # Epoch 0: Clean (10 µV PTP = ±5 µV amplitude)
    data[0, :, :] = 5e-6 * np.sin(2 * np.pi * 10 * t)

    # Epoch 1: Clean (20 µV PTP = ±10 µV amplitude)
    data[1, :, :] = 10e-6 * np.sin(2 * np.pi * 10 * t)

    # Epoch 2: High amplitude artifact (200 µV PTP - should be rejected)
    data[2, :, :] = 100e-6 * np.sin(2 * np.pi * 10 * t)

    # Epoch 3: Flat signal (0.4 µV PTP - should be rejected)
    data[3, :, :] = 0.2e-6 * np.sin(2 * np.pi * 10 * t)

    # Epoch 4: Clean (30 µV PTP = ±15 µV amplitude)
    data[4, :, :] = 15e-6 * np.sin(2 * np.pi * 10 * t)

    # Epoch 5: Very high amplitude (300 µV PTP - should be rejected)
    data[5, :, :] = 150e-6 * np.sin(2 * np.pi * 10 * t)

    # Replace data with our controlled signal
    epochs._data = data

    return epochs


class TestRejectEpochs:
    """Test reject_bad_epochs function."""

    def test_reject_high_amplitude(self, mock_epochs_with_artifacts):
        """Test rejection of high-amplitude epochs."""
        n_before = len(mock_epochs_with_artifacts)

        # Reject epochs with PTP > 100 µV
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 100e-6},
            flat=None,
            verbose=False
        )

        # Should reject epochs 2 and 5 (high amplitude)
        assert len(epochs_clean) < n_before
        assert len(epochs_clean) >= 4  # At least 4 clean epochs should remain

    def test_reject_flat_epochs(self, mock_epochs_with_artifacts):
        """Test rejection of flat/disconnected epochs."""
        n_before = len(mock_epochs_with_artifacts)

        # Reject epochs with PTP < 1 µV
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject=None,
            flat={'eeg': 1e-6},
            verbose=False
        )

        # Should reject epoch 3 (flat)
        assert len(epochs_clean) < n_before

    def test_reject_combined(self, mock_epochs_with_artifacts):
        """Test combined reject and flat criteria."""
        n_before = len(mock_epochs_with_artifacts)

        # Apply both criteria
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 100e-6},
            flat={'eeg': 1e-6},
            verbose=False
        )

        # Should reject epochs 2, 3, 5 (high amplitude and flat)
        n_expected_good = 3  # Epochs 0, 1, 4
        assert len(epochs_clean) == n_expected_good

    def test_no_rejection(self, mock_epochs_with_artifacts):
        """Test with very lenient criteria (no rejection)."""
        n_before = len(mock_epochs_with_artifacts)

        # Very lenient criteria - should keep all
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 500e-6},  # Very high threshold
            flat={'eeg': 0.01e-6},   # Very low threshold
            verbose=False
        )

        assert len(epochs_clean) == n_before

    def test_returns_epochs_object(self, mock_epochs_with_artifacts):
        """Test that function returns Epochs object."""
        result = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 100e-6},
            verbose=False
        )

        assert isinstance(result, mne.Epochs)

    def test_invalid_input_type(self):
        """Test error on invalid input type."""
        with pytest.raises(TypeError, match="Expected mne.Epochs"):
            reject_bad_epochs(
                "not an epochs object",
                reject={'eeg': 100e-6}
            )

    def test_not_preloaded(self, mock_epochs_with_artifacts):
        """Test error when data not preloaded."""
        epochs_copy = mock_epochs_with_artifacts.copy()
        epochs_copy.preload = False

        with pytest.raises(ValueError, match="must be preloaded"):
            reject_bad_epochs(epochs_copy, reject={'eeg': 100e-6})

        # Restore for cleanup
        epochs_copy.preload = True

    def test_gradient_check_placeholder(self, mock_epochs_with_artifacts):
        """Test that gradient checking parameter is accepted (placeholder)."""
        # Gradient checking should be accepted but not cause errors
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 100e-6},
            check_gradient=True,
            gradient_threshold=50.0,
            verbose=False
        )

        # Should still apply amplitude-based rejection
        assert isinstance(epochs_clean, mne.Epochs)

    def test_brainvision_compatible_settings(self, mock_epochs_with_artifacts):
        """Test BrainVision Analyzer-compatible settings."""
        # BrainVision defaults
        epochs_clean = reject_bad_epochs(
            mock_epochs_with_artifacts.copy(),
            reject={'eeg': 200e-6},   # Max difference: 200 µV
            flat={'eeg': 0.5e-6},     # Min activity: 0.5 µV
            verbose=False
        )

        # Should reject high amplitude and flat epochs
        assert isinstance(epochs_clean, mne.Epochs)
        assert len(epochs_clean) < len(mock_epochs_with_artifacts)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
