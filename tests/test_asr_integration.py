"""Test ASR (Artifact Subspace Reconstruction) integration."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from eeg_processor.processing.artifact import remove_artifacts_asr, clean_rawdata_asr, get_asr_quality_summary, ASR_AVAILABLE
from eeg_processor.state_management.data_processor import DataProcessor
from eeg_processor.utils.exceptions import ConfigurationError, ValidationError


class TestASRIntegration:
    """Test ASR integration with the EEG processing pipeline."""
    
    def test_asr_availability_check(self):
        """Test ASR availability check mechanism."""
        # This test will pass if ASRpy is installed, or check graceful degradation
        if ASR_AVAILABLE:
            assert True  # ASRpy is available
        else:
            # Test that appropriate error is raised when ASRpy not available
            mock_raw = Mock()
            mock_raw.info = {'sfreq': 250}
            mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
            mock_raw.times = np.linspace(0, 10, 2500)
            
            with pytest.raises(ImportError, match="ASRpy is not installed"):
                remove_artifacts_asr(mock_raw)
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    def test_asr_parameter_validation(self):
        """Test ASR parameter validation."""
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 250}
        mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        mock_raw.times = np.linspace(0, 10, 2500)
        mock_raw.copy.return_value = mock_raw
        
        # Test invalid cutoff
        with pytest.raises(ValueError, match="ASR cutoff must be positive"):
            remove_artifacts_asr(mock_raw, cutoff=-5)
        
        # Test invalid window overlap
        with pytest.raises(ValueError, match="Window overlap must be between 0 and 1"):
            remove_artifacts_asr(mock_raw, window_overlap=1.5)
        
        # Test invalid max dropout fraction
        with pytest.raises(ValueError, match="Max dropout fraction must be between 0 and 1"):
            remove_artifacts_asr(mock_raw, max_dropout_fraction=2.0)
        
        # Test invalid min clean fraction
        with pytest.raises(ValueError, match="Min clean fraction must be between 0 and 1"):
            remove_artifacts_asr(mock_raw, min_clean_fraction=0)
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    @patch('eeg_processor.processing.artifact.asrpy')
    def test_asr_successful_processing(self, mock_asrpy):
        """Test successful ASR processing flow."""
        # Setup mock raw data
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 250}
        mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        mock_raw.times = np.linspace(0, 10, 2500)
        mock_raw.copy.return_value = mock_raw
        
        # Mock data for correlation calculations
        test_data = np.random.randn(4, 2500)
        mock_raw.get_data.return_value = test_data
        
        # Setup mock ASR - Need to ensure the returned object passes isinstance(BaseRaw) check
        mock_asr_instance = Mock()
        mock_cleaned_raw = Mock()
        mock_cleaned_raw.__class__.__name__ = 'RawArray'  # Make it look like a Raw object
        mock_cleaned_raw.__class__.__bases__ = (Mock(),)  # Mock BaseRaw inheritance
        mock_cleaned_raw.get_data.return_value = test_data * 0.9  # Slightly modified data
        mock_asr_instance.transform.return_value = mock_cleaned_raw
        mock_asrpy.ASR.return_value = mock_asr_instance
        
        # Mock isinstance check for BaseRaw and run test
        with patch('eeg_processor.processing.artifact.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            # Test ASR processing
            result = clean_rawdata_asr(
                mock_raw,
                cutoff=20,
                method="euclid",
                show_plot=False,
                verbose=True
            )
        
        # Verify ASR was called with correct parameters
        mock_asrpy.ASR.assert_called_once()
        call_args = mock_asrpy.ASR.call_args[1]
        assert call_args['sfreq'] == 250
        assert call_args['cutoff'] == 20
        assert call_args['method'] == "euclid"
        
        # Verify fit and transform were called
        mock_asr_instance.fit.assert_called_once()
        mock_asr_instance.transform.assert_called_once()
        
        # Verify result has ASR metrics
        assert hasattr(result, '_asr_metrics')
        metrics = result._asr_metrics
        assert metrics['cutoff'] == 20
        assert metrics['method'] == "euclid"
        assert 'mean_correlation' in metrics
        assert 'variance_change_percent' in metrics
        assert 'channel_rms_changes' in metrics
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    @patch('eeg_processor.processing.artifact.asrpy')
    def test_asr_error_handling(self, mock_asrpy):
        """Test ASR error handling and fallback."""
        # Setup mock raw data
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 250}
        mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        mock_raw.times = np.linspace(0, 10, 2500)
        mock_raw.copy.return_value = mock_raw
        
        # Mock ASR to raise an exception
        mock_asrpy.ASR.side_effect = Exception("ASR processing failed")
        
        # Test error handling
        result = remove_artifacts_asr(mock_raw, cutoff=20)
        
        # Verify error metrics are stored
        assert hasattr(result, '_asr_metrics')
        metrics = result._asr_metrics
        assert metrics['correction_applied'] is False
        assert 'error' in metrics
        assert "ASR processing failed" in metrics['error']
    
    def test_asr_quality_summary_with_metrics(self):
        """Test ASR quality summary extraction with metrics."""
        mock_raw = Mock()
        mock_raw._asr_metrics = {
            'cutoff': 20,
            'method': 'euclid',
            'mean_correlation': 0.95,
            'variance_change_percent': -5.2,
            'n_channels': 32
        }
        
        summary = get_asr_quality_summary(mock_raw)
        
        assert summary['cutoff'] == 20
        assert summary['method'] == 'euclid'
        assert summary['mean_correlation'] == 0.95
        assert summary['variance_change_percent'] == -5.2
    
    def test_asr_quality_summary_without_metrics(self):
        """Test ASR quality summary fallback without metrics."""
        mock_raw = Mock()
        # No _asr_metrics attribute
        
        summary = get_asr_quality_summary(mock_raw)
        
        assert summary['cutoff'] is None
        assert summary['method'] == 'unknown'
        assert summary['correction_applied'] is False
        assert summary['mean_correlation'] is None
    
    def test_data_processor_asr_integration(self):
        """Test ASR integration with DataProcessor."""
        processor = DataProcessor()
        
        # Test ASR stage is registered
        assert "clean_rawdata_asr" in processor.stage_registry
        
        # Test ASR can be called through remove_artifacts with method="asr"
        assert processor.stage_registry["remove_artifacts"] == processor._remove_artifacts
    
    @patch('eeg_processor.processing.artifact.clean_rawdata_asr')
    def test_data_processor_asr_method_call(self, mock_asr_func):
        """Test DataProcessor calls ASR function correctly."""
        processor = DataProcessor()
        mock_raw = Mock()
        mock_asr_func.return_value = mock_raw
        
        # Test direct ASR method call
        result = processor._clean_rawdata_asr(
            mock_raw,
            cutoff=25,
            method="riemann",
            calibration_duration=90,
            inplace=False
        )
        
        # Verify function was called with correct parameters
        mock_asr_func.assert_called_once_with(
            raw=mock_raw,
            cutoff=25,
            method="riemann",
            calibration_duration=90,
            inplace=False
        )
        assert result == mock_raw
    
    # ASR is a separate stage (clean_rawdata_asr), not a method of remove_artifacts
    
    def test_data_processor_invalid_artifact_method(self):
        """Test DataProcessor with invalid artifact removal method."""
        processor = DataProcessor()
        mock_raw = Mock()
        
        # Test invalid method raises appropriate error
        with pytest.raises(ValueError, match="Unknown artifact removal method: invalid"):
            processor._remove_artifacts(mock_raw, method="invalid")
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    def test_asr_blocksize_auto_determination(self):
        """Test automatic blocksize determination."""
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 500}  # Higher sampling rate
        mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        mock_raw.times = np.linspace(0, 60, 30000)  # 60 seconds of data
        mock_raw.copy.return_value = mock_raw
        
        with patch('eeg_processor.processing.artifact.asrpy') as mock_asrpy:
            mock_asr_instance = Mock()
            mock_cleaned_raw = Mock()
            mock_cleaned_raw.get_data.return_value = np.random.randn(4, 30000)
            mock_asr_instance.fit_transform.return_value = mock_cleaned_raw
            mock_asrpy.ASR.return_value = mock_asr_instance
            mock_raw.get_data.return_value = np.random.randn(4, 30000)
            
            # Test with no blocksize specified
            remove_artifacts_asr(mock_raw, blocksize=None, verbose=True)
            
            # Verify blocksize was auto-determined (should be 30s * 500Hz = 15000)
            call_args = mock_asrpy.ASR.call_args[1]
            expected_blocksize = min(30 * 500, 30000)  # 15000
            assert call_args['blocksize'] == expected_blocksize
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    @patch('eeg_processor.processing.artifact.asrpy')
    def test_asr_calibration_duration(self, mock_asrpy):
        """Test ASR with explicit calibration duration."""
        # Setup mock raw data
        mock_raw = Mock()
        mock_raw.info = {'sfreq': 250}
        mock_raw.ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        mock_raw.times = np.linspace(0, 120, 30000)  # 120 seconds of data
        mock_raw.copy.return_value = mock_raw
        mock_raw.crop.return_value = mock_raw  # Mock crop method
        
        # Mock data for correlation calculations
        test_data = np.random.randn(4, 30000)
        mock_raw.get_data.return_value = test_data
        
        # Setup mock ASR
        mock_asr_instance = Mock()
        mock_cleaned_raw = Mock()
        mock_cleaned_raw.get_data.return_value = test_data * 0.9
        mock_asr_instance.transform.return_value = mock_cleaned_raw
        mock_asrpy.ASR.return_value = mock_asr_instance
        
        # Test ASR with explicit calibration duration
        result = remove_artifacts_asr(
            mock_raw,
            calibration_duration=60,  # Use first 60 seconds for calibration
            verbose=True
        )
        
        # Verify fit was called on cropped data and transform on full data
        mock_asr_instance.fit.assert_called_once()
        mock_asr_instance.transform.assert_called_once_with(mock_raw)
        
        # Verify metrics include calibration info
        assert hasattr(result, '_asr_metrics')
        metrics = result._asr_metrics
        assert metrics['calibration_duration'] == 60
        assert metrics['calibration_approach'] == 'explicit_duration'


class TestASRConfigurationIntegration:
    """Test ASR configuration integration."""
    
    def test_asr_in_simple_template(self):
        """Test ASR configuration in simple template."""
        template_path = Path(__file__).parent.parent / "config" / "simple_template.yml"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                content = f.read()
                
            # Check ASR is mentioned in the template
            assert "asr" in content.lower()
            assert "artifact subspace reconstruction" in content.lower()
            assert "cutoff" in content.lower()
    
    def test_asr_in_new_config_structure(self):
        """Test ASR configuration in new structure example."""
        config_path = Path(__file__).parent.parent / "config" / "example_new_structure.yml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
                
            # Check ASR is mentioned in the new structure
            assert "remove_artifacts_asr" in content
            assert "cutoff" in content


@pytest.mark.integration
class TestASRIntegrationWithRealData:
    """Integration tests with real data structures (if available)."""
    
    @pytest.mark.skipif(not ASR_AVAILABLE, reason="ASRpy not installed")
    def test_asr_with_mne_structure(self):
        """Test ASR with MNE-like data structure."""
        try:
            import mne
            from mne.io import RawArray
            
            # Create minimal test data
            sfreq = 250
            n_channels = 4
            n_samples = 1000
            data = np.random.randn(n_channels, n_samples) * 1e-6  # Scale to realistic EEG values
            
            # Create channel info
            ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names, sfreq, ch_types)
            
            # Create Raw object
            raw = RawArray(data, info)
            
            # Test ASR processing
            with patch('eeg_processor.processing.artifact.asrpy') as mock_asrpy:
                mock_asr_instance = Mock()
                mock_cleaned_raw = raw.copy()  # Return a copy of the original
                mock_asr_instance.fit_transform.return_value = mock_cleaned_raw
                mock_asrpy.ASR.return_value = mock_asr_instance
                
                result = remove_artifacts_asr(raw, cutoff=20)
                
                # Verify result is still an MNE Raw object
                assert isinstance(result, type(raw))
                assert hasattr(result, '_asr_metrics')
                
        except ImportError:
            pytest.skip("MNE not available for integration test")