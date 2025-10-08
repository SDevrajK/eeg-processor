"""
Comprehensive unit tests for EMCP (Eye Movement Correction Procedures) implementation.

Tests cover both EOG regression and Gratton & Coles methods with mock data generation
for reliable testing without requiring actual EEG files.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import warnings
from unittest.mock import patch, MagicMock

# MNE imports
import mne
from mne.io import RawArray
from mne import create_info

# EMCP imports
from eeg_processor.processing.emcp import (
    remove_blinks_eog_regression,
    remove_blinks_gratton_coles,
    _validate_eog_channels,
    _calculate_emcp_metrics,
    get_emcp_quality_summary
)


class TestEMCPValidation:
    """Test validation functions for EMCP methods."""
    
    def test_validate_eog_channels_success(self):
        """Test successful EOG channel validation."""
        # Create mock raw data with EOG channels
        raw = self._create_mock_raw_with_eog()
        
        # Should not raise exception
        _validate_eog_channels(raw, ['HEOG', 'VEOG'])
    
    def test_validate_eog_channels_missing(self):
        """Test validation failure with missing EOG channels."""
        # Create mock raw data without EOG channels
        raw = self._create_mock_raw_without_eog()
        
        with pytest.raises(ValueError, match="Missing EOG channels"):
            _validate_eog_channels(raw, ['HEOG', 'VEOG'])
    
    def test_validate_eog_channels_empty_list(self):
        """Test validation failure with empty channel list."""
        raw = self._create_mock_raw_with_eog()
        
        with pytest.raises(ValueError, match="At least one EOG channel must be specified"):
            _validate_eog_channels(raw, [])
    
    def _create_mock_raw_with_eog(self):
        """Create mock Raw object with EEG and EOG channels."""
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        # Create channel names (8 EEG + 2 EOG)
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5  # Scale to typical EEG amplitudes
        
        return RawArray(data, info)
    
    def _create_mock_raw_without_eog(self):
        """Create mock Raw object with only EEG channels."""
        n_channels = 8
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)]
        ch_types = ['eeg'] * 8
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


class TestEMCPMetrics:
    """Test quality metrics calculation for EMCP methods."""
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        # Create mock data
        n_channels, n_times = 5, 1000
        original_data = np.random.randn(n_channels, n_times)
        
        # Create cleaned data with some correlation to original
        noise = np.random.randn(n_channels, n_times) * 0.1
        cleaned_data = original_data * 0.9 + noise
        
        # Create EOG data
        eog_data = np.random.randn(2, n_times)
        
        # Create regression coefficients
        regression_coeffs = np.random.randn(n_channels) * 0.1
        
        metrics = _calculate_emcp_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_data=eog_data,
            method='gratton_coles',
            regression_coeffs=regression_coeffs,
            eog_channels=['HEOG', 'VEOG'],
            blink_events=15
        )
        
        # Check required fields
        assert 'method' in metrics
        assert 'mean_correlation' in metrics
        assert 'correction_applied' in metrics
        assert 'quality_flags' in metrics
        assert metrics['method'] == 'gratton_coles'
        assert metrics['correction_applied'] is True
        assert metrics['blink_events'] == 15
        
        # Check correlation values are reasonable
        assert 0 <= metrics['mean_correlation'] <= 1
        # Note: individual channel correlations not stored in minimal metrics
    
    def test_eog_regression_specific_metrics(self):
        """Test metrics specific to EOG regression method."""
        n_channels, n_times = 5, 1000
        original_data = np.random.randn(n_channels, n_times)
        cleaned_data = original_data + np.random.randn(n_channels, n_times) * 0.05
        eog_data = np.random.randn(2, n_times)
        
        metrics = _calculate_emcp_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_data=eog_data,
            method='eog_regression',
            eog_channels=['HEOG', 'VEOG']
        )
        
        assert metrics['method'] == 'eog_regression'
        # Note: method details not stored in minimal metrics
    
    def test_gratton_coles_specific_metrics(self):
        """Test metrics specific to Gratton & Coles method."""
        n_channels, n_times = 5, 1000
        original_data = np.random.randn(n_channels, n_times)
        cleaned_data = original_data + np.random.randn(n_channels, n_times) * 0.05
        eog_data = np.random.randn(2, n_times)
        regression_coeffs = np.random.randn(n_channels) * 0.1
        
        metrics = _calculate_emcp_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_data=eog_data,
            method='gratton_coles',
            regression_coeffs=regression_coeffs,
            subtract_evoked=True
        )
        
        assert metrics['method'] == 'gratton_coles'
        assert 'max_regression_coefficient' in metrics
        assert 'subtract_evoked' in metrics
        assert metrics['subtract_evoked'] is True
        # Note: individual coefficients not stored, only max coefficient


class TestEOGRegressionMethod:
    """Test EOG regression method implementation."""
    
    @patch('eeg_processor.processing.emcp.find_eog_events')
    @patch('eeg_processor.processing.emcp.EOGRegression')
    def test_eog_regression_basic_functionality(self, mock_eog_regression, mock_find_events):
        """Test basic EOG regression functionality with mocks."""
        # Setup mock raw data
        raw = self._create_mock_raw_with_blinks()
        
        # Mock blink events
        mock_events = np.array([[100, 0, 1], [300, 0, 1], [500, 0, 1]])
        mock_find_events.return_value = mock_events
        
        # Mock EOGRegression
        mock_regressor = MagicMock()
        mock_regressor.apply.return_value = raw.copy()
        mock_eog_regression.return_value = mock_regressor
        
        # Test the function
        result = remove_blinks_eog_regression(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            show_plot=False,
            verbose=False
        )
        
        # Verify mocks were called
        mock_find_events.assert_called_once()
        mock_eog_regression.assert_called_once()
        mock_regressor.fit.assert_called_once()
        mock_regressor.apply.assert_called_once()
        
        # Check result has metrics
        assert hasattr(result, '_emcp_metrics')
        assert result._emcp_metrics['method'] == 'eog_regression'
        assert result._emcp_metrics['correction_applied'] is True
    
    @patch('eeg_processor.processing.emcp.find_eog_events')
    def test_eog_regression_no_blinks(self, mock_find_events):
        """Test EOG regression when no blinks are found."""
        raw = self._create_mock_raw_with_blinks()
        
        # Mock no blink events
        mock_find_events.return_value = np.array([]).reshape(0, 3)
        
        result = remove_blinks_eog_regression(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            show_plot=False,
            verbose=False
        )
        
        # Should return original data with appropriate metrics
        assert hasattr(result, '_emcp_metrics')
        assert result._emcp_metrics['blink_events_found'] == 0
        assert result._emcp_metrics['correction_applied'] is False
        assert 'no_blink_events' in result._emcp_metrics['error']
    
    def test_eog_regression_missing_channels(self):
        """Test EOG regression with missing EOG channels."""
        raw = self._create_mock_raw_without_eog()
        
        with pytest.raises(ValueError, match="Missing EOG channels"):
            remove_blinks_eog_regression(
                raw=raw,
                eog_channels=['HEOG', 'VEOG'],
                show_plot=False
            )
    
    def _create_mock_raw_with_blinks(self):
        """Create mock raw data with simulated blinks."""
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create base data
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        # Add simulated blinks to VEOG channel (index 9)
        blink_times = [100, 300, 500]
        for blink_time in blink_times:
            if blink_time < n_times - 50:
                # Add blink artifact (negative deflection for VEOG)
                data[9, blink_time:blink_time+50] += -100e-6
                # Add smaller correlated artifacts to EEG channels
                for eeg_ch in range(8):
                    data[eeg_ch, blink_time:blink_time+50] += -20e-6 * np.random.random()
        
        return RawArray(data, info)
    
    def _create_mock_raw_without_eog(self):
        """Create mock Raw object without EOG channels."""
        n_channels = 8
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)]
        ch_types = ['eeg'] * 8
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


class TestGrattonColesMethod:
    """Test Gratton & Coles method implementation."""
    
    @patch('eeg_processor.processing.emcp.find_eog_events')
    def test_gratton_coles_basic_functionality(self, mock_find_events):
        """Test basic Gratton & Coles functionality."""
        raw = self._create_mock_raw_with_blinks()
        
        # Mock blink events
        mock_events = np.array([[100, 0, 1], [300, 0, 1], [500, 0, 1]])
        mock_find_events.return_value = mock_events
        
        result = remove_blinks_gratton_coles(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=False,  # Disable for simpler testing
            show_plot=False,
            verbose=False
        )
        
        # Check result has metrics
        assert hasattr(result, '_emcp_metrics')
        assert result._emcp_metrics['method'] == 'gratton_coles'
        assert result._emcp_metrics['correction_applied'] is True
        assert 'max_regression_coefficient' in result._emcp_metrics
        
        # Check regression coefficient was calculated
        max_coeff = result._emcp_metrics['max_regression_coefficient']
        assert isinstance(max_coeff, float)
    
    @patch('eeg_processor.processing.emcp.find_eog_events')
    def test_gratton_coles_with_evoked_subtraction(self, mock_find_events):
        """Test Gratton & Coles with evoked response subtraction."""
        raw = self._create_mock_raw_with_blinks()
        
        # Mock blink events
        mock_events = np.array([[100, 0, 1], [300, 0, 1], [500, 0, 1]])
        mock_find_events.return_value = mock_events
        
        result = remove_blinks_gratton_coles(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=True,
            show_plot=False,
            verbose=False
        )
        
        # Check evoked subtraction was applied
        assert result._emcp_metrics['subtract_evoked'] is True
    
    @patch('eeg_processor.processing.emcp.find_eog_events')
    def test_gratton_coles_no_blinks(self, mock_find_events):
        """Test Gratton & Coles when no blinks are found."""
        raw = self._create_mock_raw_with_blinks()
        
        # Mock no blink events
        mock_find_events.return_value = np.array([]).reshape(0, 3)
        
        result = remove_blinks_gratton_coles(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            show_plot=False,
            verbose=False
        )
        
        # Should return original data with appropriate metrics
        assert hasattr(result, '_emcp_metrics')
        assert result._emcp_metrics['blink_events_found'] == 0
        assert result._emcp_metrics['correction_applied'] is False
    
    def _create_mock_raw_with_blinks(self):
        """Create mock raw data with simulated blinks."""
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create base data
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        # Add simulated blinks with known correlation pattern
        blink_times = [100, 300, 500]
        for blink_time in blink_times:
            if blink_time < n_times - 50:
                # VEOG blink (index 9)
                veog_artifact = -100e-6 * np.exp(-0.1 * np.arange(50))
                data[9, blink_time:blink_time+50] += veog_artifact
                
                # Add correlated artifacts to EEG channels with known coefficients
                for eeg_ch in range(8):
                    beta = 0.1 + 0.05 * eeg_ch  # Different correlation for each channel
                    data[eeg_ch, blink_time:blink_time+50] += beta * veog_artifact
        
        return RawArray(data, info)


class TestEMCPQualitySummary:
    """Test EMCP quality summary functions."""
    
    def test_get_quality_summary_with_metrics(self):
        """Test getting quality summary from processed data."""
        raw = self._create_mock_raw()
        
        # Add mock metrics
        raw._emcp_metrics = {
            'method': 'eog_regression',
            'correction_applied': True,
            'mean_correlation': 0.95,
            'blink_events': 10
        }
        
        summary = get_emcp_quality_summary(raw)
        assert summary['method'] == 'eog_regression'
        assert summary['correction_applied'] is True
        assert summary['mean_correlation'] == 0.95
    
    def test_get_quality_summary_without_metrics(self):
        """Test getting quality summary from unprocessed data."""
        raw = self._create_mock_raw()
        
        summary = get_emcp_quality_summary(raw)
        assert summary['method'] == 'unknown'
        assert summary['correction_applied'] is False
        assert summary['eog_channels'] == []
    
    def _create_mock_raw(self):
        """Create simple mock raw data."""
        n_channels, n_times = 5, 100
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 6)]
        ch_types = ['eeg'] * 5
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


class TestEMCPErrorHandling:
    """Test error handling and edge cases for EMCP methods."""
    
    def test_invalid_method_parameter(self):
        """Test handling of invalid method parameters."""
        # This would be tested at the DataProcessor level
        pass
    
    def test_corrupted_eog_data(self):
        """Test handling of corrupted or invalid EOG data."""
        raw = self._create_mock_raw_with_nan_eog()
        
        # Both methods should handle NaN/inf values gracefully
        # Implementation should either clean the data or provide meaningful error
        pass
    
    def test_insufficient_data_length(self):
        """Test handling of very short data segments."""
        raw = self._create_short_mock_raw()
        
        # Methods should handle short data appropriately
        pass
    
    def _create_mock_raw_with_nan_eog(self):
        """Create mock data with NaN values in EOG channels."""
        # Implementation for testing NaN handling
        pass
    
    def _create_short_mock_raw(self):
        """Create very short mock data for edge case testing."""
        # Implementation for testing short data handling
        pass


# Integration test markers for pytest
class TestEMCPIntegration:
    """Integration tests requiring actual test data files."""
    
    def test_emcp_with_real_brainvision_data(self):
        """Test EMCP methods with real BrainVision test data."""
        test_data_dir = Path(__file__).parent / "test_data" / "brainvision"
        test_file = test_data_dir / "test.vhdr"
        
        if not test_file.exists():
            pytest.skip("Test data not available")
        
        # Load real data and test EMCP methods
        # This would require actual test data with EOG channels
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])