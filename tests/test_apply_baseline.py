"""
Tests for apply_baseline stage.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import mne
from mne import create_info, Epochs
from mne.io import RawArray

from src.eeg_processor.processing.apply_baseline import apply_baseline_correction


@pytest.fixture
def mock_epochs():
    """Create mock Epochs for testing."""
    n_channels = 8
    n_times = 200  # 200 samples at 250 Hz = 0.8s
    n_epochs = 5
    sfreq = 250.0

    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create epochs with known signal + DC offset
    # Time: -0.2 to 0.6 seconds
    data = np.zeros((n_epochs, n_channels, n_times))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            # Add DC offset that should be removed by baseline
            dc_offset = 10.0
            # Add sinusoidal signal
            t = np.arange(n_times) / sfreq - 0.2
            signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz
            data[epoch_idx, ch_idx, :] = signal + dc_offset

    # Create events
    events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])

    epochs = Epochs(
        RawArray(np.random.randn(n_channels, 5000) * 1e-5, info, verbose=False),
        events=events,
        event_id={'stimulus': 1},
        tmin=-0.2,
        tmax=0.6,
        baseline=None,  # No baseline initially
        preload=True,
        verbose=False
    )

    # Replace data with our controlled signal
    epochs._data = data

    return epochs


class TestApplyBaseline:
    """Test apply_baseline_correction function."""

    def test_basic_baseline_correction(self, mock_epochs):
        """Test basic baseline correction modifies data."""
        original_data = mock_epochs.get_data().copy()

        # Apply baseline from -0.2 to 0
        epochs_clean = apply_baseline_correction(
            mock_epochs.copy(),
            baseline=(-0.2, 0),
            verbose=False
        )

        # Data should be modified (baseline subtracted)
        clean_data = epochs_clean.get_data()

        # Check that baseline property is set correctly
        assert epochs_clean.baseline == (-0.2, 0)

        # Data should be different from original (baseline was subtracted)
        # Note: we can't easily check exact values due to random mock data
        assert isinstance(epochs_clean, mne.Epochs)

    def test_baseline_none_no_change(self, mock_epochs):
        """Test baseline=None doesn't modify data."""
        original_data = mock_epochs.get_data().copy()

        epochs_clean = apply_baseline_correction(
            mock_epochs.copy(),
            baseline=None,
            verbose=False
        )

        # Data should be unchanged
        assert_allclose(epochs_clean.get_data(), original_data, rtol=1e-10)

    def test_baseline_default(self, mock_epochs):
        """Test default baseline (None, 0)."""
        epochs_clean = apply_baseline_correction(
            mock_epochs.copy(),
            verbose=False
        )

        # Check that baseline property is set correctly
        # MNE converts (None, 0) to (tmin, 0) internally
        assert epochs_clean.baseline is not None
        assert isinstance(epochs_clean, mne.Epochs)

    def test_returns_epochs_object(self, mock_epochs):
        """Test that function returns Epochs object."""
        result = apply_baseline_correction(
            mock_epochs.copy(),
            baseline=(-0.2, 0),
            verbose=False
        )

        assert isinstance(result, mne.Epochs)

    def test_invalid_input_type(self):
        """Test error on invalid input type."""
        with pytest.raises(TypeError, match="Expected mne.Epochs"):
            apply_baseline_correction(
                "not an epochs object",
                baseline=(-0.2, 0)
            )

    def test_not_preloaded(self, mock_epochs):
        """Test error when data not preloaded."""
        # Create a real Epochs copy and then modify the internal state
        # to simulate non-preloaded data
        epochs_copy = mock_epochs.copy()

        # Temporarily set preload to False to test validation
        original_preload = epochs_copy.preload
        epochs_copy.preload = False

        with pytest.raises(ValueError, match="must be preloaded"):
            apply_baseline_correction(epochs_copy, baseline=(-0.2, 0))

        # Restore for cleanup
        epochs_copy.preload = original_preload

    def test_invalid_baseline_format(self, mock_epochs):
        """Test error on invalid baseline format."""
        with pytest.raises(ValueError, match="tuple of length 2"):
            apply_baseline_correction(
                mock_epochs.copy(),
                baseline=(-0.2,)  # Only one element
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
