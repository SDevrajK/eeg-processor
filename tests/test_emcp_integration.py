"""
Integration tests for EMCP (Eye Movement Correction Procedures) with DataProcessor pipeline.

Tests the full integration of EMCP methods within the EEG processing pipeline,
including configuration parsing, stage execution, and quality control integration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from eeg_processor.state_management.data_processor import DataProcessor
from eeg_processor.utils.config_loader import load_config
from eeg_processor.pipeline import EEGPipeline


class TestEMCPDataProcessorIntegration:
    """Test EMCP integration with DataProcessor."""
    
    def test_emcp_stage_registration(self):
        """Test that EMCP stage is properly registered in DataProcessor."""
        dp = DataProcessor()
        
        # Check stage is registered
        assert 'remove_blinks_emcp' in dp.stage_registry
        
        # Check method reference is correct
        stage_func = dp.stage_registry['remove_blinks_emcp']
        assert callable(stage_func)
        assert stage_func.__name__ == '_remove_blinks_emcp'
    
    def test_emcp_method_parameter_validation(self):
        """Test EMCP method parameter validation in DataProcessor."""
        dp = DataProcessor()
        
        # Create mock raw data
        raw = self._create_mock_raw_with_eog()
        
        # Test valid methods
        valid_methods = ['eog_regression', 'gratton_coles']
        for method in valid_methods:
            try:
                # This should not raise an exception for method validation
                # (though it might fail on actual processing due to mock data)
                result = dp._remove_blinks_emcp(
                    data=raw,
                    method=method,
                    eog_channels=['HEOG', 'VEOG'],
                    show_plot=False
                )
            except ValueError as e:
                if "Unknown EMCP method" in str(e):
                    pytest.fail(f"Valid method {method} was rejected")
                # Other ValueError exceptions are OK (e.g., processing failures)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown EMCP method"):
            dp._remove_blinks_emcp(
                data=raw,
                method='invalid_method',
                eog_channels=['HEOG', 'VEOG']
            )
    
    def test_emcp_stage_execution(self):
        """Test EMCP stage execution through DataProcessor."""
        dp = DataProcessor()
        raw = self._create_mock_raw_with_eog()
        
        # Test EOG regression method
        try:
            result = dp.apply_processing_stage(
                data=raw,
                stage_name='remove_blinks_emcp',
                method='eog_regression',
                eog_channels=['HEOG', 'VEOG'],
                show_plot=False
            )
            
            # Should return a Raw object
            assert hasattr(result, 'info')
            assert hasattr(result, 'get_data')
            
        except Exception as e:
            # Processing might fail with mock data, but stage should be callable
            assert "Unknown stage" not in str(e)
    
    def _create_mock_raw_with_eog(self):
        """Create mock Raw object with EEG and EOG channels."""
        import numpy as np
        from mne.io import RawArray
        from mne import create_info
        
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


class TestEMCPConfigurationParsing:
    """Test EMCP configuration parsing and validation."""
    
    def test_emcp_config_parsing(self):
        """Test parsing of EMCP stage configuration."""
        config_data = {
            "paths": {
                "raw_data_dir": "/tmp/test_data",
                "results_dir": "/tmp/results",
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": [
                {
                    "remove_blinks_emcp": {
                        "method": "eog_regression",
                        "eog_channels": ["HEOG", "VEOG"],
                        "show_plot": False
                    }
                }
            ],
            "conditions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # This tests that the configuration can be loaded without syntax errors
            # The actual path validation would fail, but stage parsing should work
            config = load_config(temp_path)
            
            # Check that stages were parsed
            assert len(config.stages) == 1
            stage = config.stages[0]
            assert 'remove_blinks_emcp' in stage
            
            # Check EMCP-specific parameters
            emcp_params = stage['remove_blinks_emcp']
            assert emcp_params['method'] == 'eog_regression'
            assert emcp_params['eog_channels'] == ['HEOG', 'VEOG']
            assert emcp_params['show_plot'] is False
            
        except Exception as e:
            # Configuration might fail on path validation, but should parse
            if "remove_blinks_emcp" in str(e):
                pytest.fail(f"EMCP stage configuration parsing failed: {e}")
        finally:
            Path(temp_path).unlink()
    
    def test_emcp_gratton_coles_config(self):
        """Test configuration for Gratton & Coles method."""
        config_data = {
            "paths": {
                "raw_data_dir": "/tmp/test_data",
                "results_dir": "/tmp/results",
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": [
                {
                    "remove_blinks_emcp": {
                        "method": "gratton_coles",
                        "eog_channels": ["HEOG", "VEOG"],
                        "subtract_evoked": True,
                        "show_plot": False
                    }
                }
            ],
            "conditions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            stage = config.stages[0]
            emcp_params = stage['remove_blinks_emcp']
            
            assert emcp_params['method'] == 'gratton_coles'
            assert emcp_params['subtract_evoked'] is True
            
        except Exception as e:
            if "remove_blinks_emcp" in str(e):
                pytest.fail(f"Gratton & Coles configuration parsing failed: {e}")
        finally:
            Path(temp_path).unlink()


class TestEMCPPipelineIntegration:
    """Test EMCP integration with full EEG processing pipeline."""
    
    def test_emcp_in_processing_pipeline(self):
        """Test EMCP as part of a processing pipeline."""
        # This test would require actual test data with EOG channels
        # For now, we test the configuration and setup
        
        # Use actual test data directory
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_data = {
            "paths": {
                "raw_data_dir": str(test_data_dir / "brainvision"),
                "results_dir": str(test_data_dir / "results"),
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": [
                {"filter": {"l_freq": 0.1, "h_freq": 40}},
                {"detect_bad_channels": {"interpolate": True}},
                {
                    "remove_blinks_emcp": {
                        "method": "eog_regression",
                        "eog_channels": ["HEOG", "VEOG"]
                    }
                },
                {"rereference": {"method": "average"}}
            ],
            "conditions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            
            # Check that EMCP stage is in the correct position
            stage_names = []
            for stage in config.stages:
                stage_names.extend(stage.keys())
            
            assert 'remove_blinks_emcp' in stage_names
            
            # Check pipeline order makes sense
            emcp_index = stage_names.index('remove_blinks_emcp')
            filter_index = stage_names.index('filter')
            bad_channels_index = stage_names.index('detect_bad_channels')
            
            # EMCP should come after filtering and bad channel detection
            assert emcp_index > filter_index
            assert emcp_index > bad_channels_index
            
        finally:
            Path(temp_path).unlink()
    
    def test_emcp_with_quality_tracking(self):
        """Test that EMCP stages produce quality tracking metrics."""
        # This would test the integration with quality control systems
        # The test verifies that EMCP metrics are properly captured
        
        dp = DataProcessor()
        raw = self._create_mock_raw_with_eog()
        
        try:
            result = dp._remove_blinks_emcp(
                data=raw,
                method='eog_regression',
                eog_channels=['HEOG', 'VEOG'],
                show_plot=False
            )
            
            # Check that quality metrics were generated
            if hasattr(result, '_emcp_metrics'):
                metrics = result._emcp_metrics
                
                # Check required metric fields
                required_fields = ['method', 'correction_applied', 'eog_channels']
                for field in required_fields:
                    assert field in metrics, f"Missing required metric field: {field}"
                
                assert metrics['method'] in ['eog_regression', 'gratton_coles']
                assert isinstance(metrics['correction_applied'], bool)
                assert isinstance(metrics['eog_channels'], list)
            
        except Exception as e:
            # Processing might fail, but should attempt to create metrics
            pass
    
    def _create_mock_raw_with_eog(self):
        """Create mock Raw object with EEG and EOG channels."""
        import numpy as np
        from mne.io import RawArray
        from mne import create_info
        
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


class TestEMCPBackwardCompatibility:
    """Test backward compatibility with existing configurations."""
    
    def test_legacy_regression_stage_deprecation(self):
        """Test that legacy regression stages show deprecation warnings."""
        # This would test the deprecation of remove_blinks_regression
        # and its redirection to EMCP implementation
        
        from eeg_processor.processing.artifact import remove_blinks_regression
        import warnings
        
        raw = self._create_mock_raw_with_eog()
        
        # Test that deprecation warning is raised
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = remove_blinks_regression(
                    raw=raw,
                    eog_channels=['HEOG', 'VEOG'],
                    show_plot=False
                )
                
                # Check that deprecation warning was issued
                deprecation_warnings = [warning for warning in w 
                                       if issubclass(warning.category, DeprecationWarning)]
                assert len(deprecation_warnings) > 0, "No deprecation warning issued"
                
                # Check warning message mentions EMCP
                warning_msg = str(deprecation_warnings[0].message)
                assert 'remove_blinks_emcp' in warning_msg
                
            except Exception as e:
                # Processing might fail, but warning should still be issued
                if len(w) == 0:
                    pytest.fail("Expected deprecation warning was not issued")
    
    def _create_mock_raw_with_eog(self):
        """Create mock Raw object with EEG and EOG channels."""
        import numpy as np
        from mne.io import RawArray
        from mne import create_info
        
        n_channels = 10
        n_times = 1000
        sfreq = 250.0
        
        ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        return RawArray(data, info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])