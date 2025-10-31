"""
Essential unit tests for EOG regression-based artifact removal.

Focuses on core functionality, input validation, and critical edge cases.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import mne
from mne.io import RawArray
from mne import create_info, BaseEpochs, Epochs

from src.eeg_processor.processing.regression import (
    remove_artifacts_regression,
    _validate_inputs,
    _calculate_regression_metrics
)


@pytest.fixture
def mock_raw_with_eog():
    """Create mock Raw data with EEG and EOG channels."""
    n_eeg = 8
    n_eog = 2
    n_times = 2000
    sfreq = 250.0

    # Channel setup
    ch_names = [f'EEG{i:03d}' for i in range(1, n_eeg + 1)] + ['HEOG', 'VEOG']
    ch_types = ['eeg'] * n_eeg + ['eog'] * n_eog
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create data with blink artifacts
    data = np.random.randn(n_eeg + n_eog, n_times) * 1e-5

    # Add blink artifacts (EOG propagates to frontal EEG)
    blink_times = [500, 1000, 1500]
    for t in blink_times:
        # EOG blink
        data[-2:, t-10:t+20] += np.random.randn(n_eog, 30) * 5e-5
        # Propagate to frontal channels (first 2 EEG)
        data[0:2, t-10:t+20] += data[-2:, t-10:t+20].mean(axis=0) * 0.3

    raw = RawArray(data, info, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    return raw


@pytest.fixture
def mock_epochs_with_eog(mock_raw_with_eog):
    """Create mock Epochs from Raw data."""
    events = np.array([[500, 0, 1], [1000, 0, 1], [1500, 0, 1]])
    epochs = Epochs(
        mock_raw_with_eog,
        events,
        event_id={'stimulus': 1},
        tmin=-0.2,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs


@pytest.fixture
def mock_raw_without_eog():
    """Create mock Raw data without EOG channels."""
    n_channels = 8
    n_times = 2000
    sfreq = 250.0

    ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
    ch_types = ['eeg'] * n_channels
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    data = np.random.randn(n_channels, n_times) * 1e-5
    raw = RawArray(data, info, verbose=False)

    return raw


class TestRegressionBasicFunctionality:
    """Test core regression functionality."""

    def test_regression_raw_basic(self, mock_raw_with_eog):
        """Test basic regression on Raw data."""
        cleaned = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        # Should return Raw object
        assert isinstance(cleaned, mne.io.BaseRaw)

        # Should have metrics attached
        assert hasattr(cleaned, '_regression_metrics')
        assert cleaned._regression_metrics['method'] == 'regression'
        assert cleaned._regression_metrics['data_type'] == 'Raw'

        # Signal should be modified
        orig_data = mock_raw_with_eog.get_data(picks='eeg')
        clean_data = cleaned.get_data(picks='eeg')
        assert not np.allclose(orig_data, clean_data)

        # Correlation should be high (signal preserved)
        corr = np.corrcoef(orig_data.ravel(), clean_data.ravel())[0, 1]
        assert 0.7 < corr < 1.0

    def test_regression_epochs_basic(self, mock_epochs_with_eog):
        """Test basic regression on Epochs data."""
        cleaned = remove_artifacts_regression(
            mock_epochs_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=False,  # Direct fitting for test speed
            verbose=False
        )

        # Should return Epochs object
        assert isinstance(cleaned, mne.BaseEpochs)

        # Should have metrics attached
        assert hasattr(cleaned, '_regression_metrics')
        assert cleaned._regression_metrics['method'] == 'regression'
        assert cleaned._regression_metrics['data_type'] == 'Epochs'
        assert cleaned._regression_metrics['subtract_evoked'] == False

        # Signal should be modified
        orig_data = mock_epochs_with_eog.get_data(picks='eeg')
        clean_data = cleaned.get_data(picks='eeg')
        assert not np.allclose(orig_data, clean_data)

    def test_regression_epochs_with_evoked_subtraction(self, mock_epochs_with_eog):
        """Test Gratton & Coles method with evoked subtraction."""
        cleaned = remove_artifacts_regression(
            mock_epochs_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=True,  # Gratton method
            verbose=False
        )

        # Should have metrics with correct flag
        assert cleaned._regression_metrics['subtract_evoked'] == True
        assert cleaned._regression_metrics['data_type'] == 'Epochs'


class TestInputValidation:
    """Test input validation and error handling."""

    def test_missing_eog_channels(self, mock_raw_without_eog):
        """Test error when EOG channels are missing."""
        with pytest.raises(ValueError, match="Missing EOG channels"):
            remove_artifacts_regression(
                mock_raw_without_eog,
                eog_channels=['HEOG', 'VEOG'],
                verbose=False
            )

    def test_empty_eog_channels(self, mock_raw_with_eog):
        """Test error with empty EOG channel list."""
        # Empty list triggers auto-detection, then EOGRegression fails
        # The function returns original with error metrics instead of raising
        result = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=[],
            verbose=False
        )

        # Should return original data with error metrics
        assert hasattr(result, '_regression_metrics')
        assert result._regression_metrics['correction_applied'] == False
        assert 'error' in result._regression_metrics

    def test_data_not_preloaded(self, mock_raw_with_eog):
        """Test warning when data is not preloaded (MNE will handle)."""
        # Note: MNE's EOGRegression handles preloading internally
        # Our validation just checks data.preload attribute
        # This test verifies the check exists

        # Create a mock object with preload=False
        class MockRaw:
            preload = False
            ch_names = ['HEOG', 'VEOG']

        mock_raw = MockRaw()

        with pytest.raises(ValueError, match="must be preloaded"):
            _validate_inputs(mock_raw, ['HEOG', 'VEOG'])

    def test_auto_detect_eog_channels(self, mock_raw_with_eog):
        """Test automatic EOG channel detection."""
        # Should auto-detect HEOG and VEOG
        validated_channels = _validate_inputs(mock_raw_with_eog, None)

        assert 'HEOG' in validated_channels
        assert 'VEOG' in validated_channels
        assert len(validated_channels) == 2


class TestMetricsCalculation:
    """Test quality metrics calculation."""

    def test_metrics_structure(self, mock_raw_with_eog):
        """Test metrics dictionary structure matches FR4 specification."""
        cleaned = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        metrics = cleaned._regression_metrics

        # Required top-level fields
        assert metrics['method'] == 'regression'
        assert metrics['implementation'] == 'mne_eog_regression'
        assert metrics['data_type'] == 'Raw'
        assert metrics['eog_channels'] == ['HEOG', 'VEOG']
        assert metrics['correction_applied'] == True

        # Nested structures
        assert 'regression_coefficients' in metrics
        assert 'shape' in metrics['regression_coefficients']
        assert 'max_coeff' in metrics['regression_coefficients']
        assert 'mean_coeff' in metrics['regression_coefficients']

        assert 'artifact_reduction' in metrics
        assert 'mean_correlation_preserved' in metrics['artifact_reduction']

        assert 'quality_flags' in metrics
        assert isinstance(metrics['quality_flags'], dict)

    def test_quality_flags(self, mock_raw_with_eog):
        """Test quality flags are set correctly."""
        cleaned = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        flags = cleaned._regression_metrics['quality_flags']

        # Check all required flags exist
        assert 'low_correlation' in flags
        assert 'acceptable_correction' in flags
        assert 'high_correlation' in flags
        assert 'extreme_coefficients' in flags
        assert 'minimal_correction' in flags

        # Flags should be boolean
        assert isinstance(flags['low_correlation'], bool)
        assert isinstance(flags['acceptable_correction'], bool)

    def test_correlation_metric_range(self, mock_raw_with_eog):
        """Test correlation metric is in valid range."""
        cleaned = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        mean_corr = cleaned._regression_metrics['artifact_reduction']['mean_correlation_preserved']

        # Correlation should be between 0 and 1
        assert 0.0 <= mean_corr <= 1.0

        # For good correction, should be high
        assert mean_corr > 0.7


class TestEdgeCases:
    """Test critical edge cases."""

    def test_inplace_parameter_ignored(self, mock_raw_with_eog):
        """Test that inplace parameter is ignored (always creates copy)."""
        original = mock_raw_with_eog.copy()

        # Call with inplace=True (should still create copy)
        cleaned = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            inplace=True,
            verbose=False
        )

        # Original should be unchanged
        assert_allclose(
            original.get_data(),
            mock_raw_with_eog.get_data(),
            rtol=1e-10
        )

        # Cleaned should be different
        assert not np.allclose(
            cleaned.get_data(picks='eeg'),
            original.get_data(picks='eeg')
        )

    def test_subtract_evoked_ignored_for_raw(self, mock_raw_with_eog):
        """Test that subtract_evoked parameter is ignored for Raw data."""
        # Both should produce identical results
        cleaned_true = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=True,
            verbose=False
        )

        cleaned_false = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=False,
            verbose=False
        )

        # Results should be identical for Raw data
        assert_allclose(
            cleaned_true.get_data(),
            cleaned_false.get_data(),
            rtol=1e-10
        )

        # Metrics should show subtract_evoked=False for Raw
        assert cleaned_true._regression_metrics['subtract_evoked'] == False

    def test_error_handling_returns_original(self, mock_raw_with_eog):
        """Test that errors return original data with error metrics."""
        # Force an error by using invalid data
        raw_invalid = mock_raw_with_eog.copy()

        # Patch EOGRegression to raise error
        from unittest.mock import patch
        with patch('src.eeg_processor.processing.regression.EOGRegression') as mock_eog:
            mock_eog.side_effect = RuntimeError("Test error")

            result = remove_artifacts_regression(
                raw_invalid,
                eog_channels=['HEOG', 'VEOG'],
                verbose=False
            )

            # Should return data (original)
            assert result is not None

            # Should have error metrics
            assert hasattr(result, '_regression_metrics')
            assert result._regression_metrics['correction_applied'] == False
            assert 'error' in result._regression_metrics


class TestDataTypeSupport:
    """Test support for both Raw and Epochs data types."""

    def test_raw_input_raw_output(self, mock_raw_with_eog):
        """Test Raw input returns Raw output."""
        result = remove_artifacts_regression(
            mock_raw_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        assert isinstance(result, mne.io.BaseRaw)
        assert result._regression_metrics['data_type'] == 'Raw'

    def test_epochs_input_epochs_output(self, mock_epochs_with_eog):
        """Test Epochs input returns Epochs output."""
        result = remove_artifacts_regression(
            mock_epochs_with_eog,
            eog_channels=['HEOG', 'VEOG'],
            verbose=False
        )

        assert isinstance(result, mne.BaseEpochs)
        assert result._regression_metrics['data_type'] == 'Epochs'
        assert 'n_epochs' in result._regression_metrics
        assert result._regression_metrics['n_epochs'] == len(mock_epochs_with_eog)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
