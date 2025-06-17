"""Core functionality tests - essential for publication readiness."""

import tempfile
import pytest
from pathlib import Path
import yaml

from eeg_processor.utils.config_loader import load_config, PipelineConfig
from eeg_processor.utils.exceptions import ConfigurationError, ValidationError


class TestCoreConfiguration:
    """Essential configuration tests for research reproducibility."""

    def test_load_valid_config_with_real_files(self):
        """Test loading configuration that points to actual test files."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_data = {
            "paths": {
                "raw_data_dir": str(test_data_dir / "brainvision"),
                "results_dir": str(test_data_dir / "results"),
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": ["load_data"],
            "conditions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert isinstance(config, PipelineConfig)
            assert config.participants == {"sub-01": "test.vhdr"}
            assert config.raw_data_dir.name == "brainvision"
        finally:
            Path(temp_path).unlink()

    def test_config_validation_catches_missing_files(self):
        """Test that missing data directories are caught."""
        config_with_bad_path = {
            "paths": {
                "raw_data_dir": "/definitely/nonexistent/directory",
                "results_dir": "/tmp/results",
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": ["load_data"],
            "conditions": []
        }
        
        with pytest.raises(ValidationError, match="Raw data directory does not exist"):
            load_config(config_with_bad_path)

    def test_dataset_name_feature(self):
        """Test optional dataset name for organizing results."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_with_dataset = {
            "paths": {
                "raw_data_dir": str(test_data_dir / "brainvision"),
                "results_dir": str(test_data_dir / "results"),
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": ["load_data"],
            "conditions": [],
            "dataset_name": "test_study"
        }
        
        config = load_config(config_with_dataset)
        assert config.dataset_name == "test_study"


class TestCoreErrorHandling:
    """Essential error handling for clear user feedback."""

    def test_missing_config_file(self):
        """Test clear error when config file doesn't exist."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("/nonexistent/config.yml")

    def test_invalid_yaml_syntax(self):
        """Test handling of malformed YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: syntax: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML syntax"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_required_sections(self):
        """Test validation of required configuration sections."""
        config_without_paths = {
            "stages": ["load_data"],
            "conditions": []
        }
        
        with pytest.raises(ValidationError, match="Missing required configuration keys"):
            load_config(config_without_paths)


class TestCoreFileSupport:
    """Essential file format support tests."""

    def test_brainvision_loader_available(self):
        """Test BrainVision loader can be imported."""
        from eeg_processor.file_io.brainvision import BrainVisionLoader
        loader = BrainVisionLoader()
        assert hasattr(loader, 'load')

    def test_edf_loader_available(self):
        """Test EDF loader can be imported."""
        from eeg_processor.file_io.edf import EDFLoader  
        loader = EDFLoader()
        assert hasattr(loader, 'load')

    def test_multiple_format_support(self):
        """Test that multiple EEG formats are supported."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        # Should have test files for different formats
        assert (test_data_dir / "brainvision" / "test.vhdr").exists()
        assert (test_data_dir / "edf" / "test.edf").exists()
        assert (test_data_dir / "fif" / "test.fif").exists()


class TestCoreDataStructures:
    """Essential data structure validation."""

    def test_pipeline_config_has_required_fields(self):
        """Test PipelineConfig dataclass structure."""
        from dataclasses import fields
        
        field_names = [f.name for f in fields(PipelineConfig)]
        essential_fields = ['raw_data_dir', 'results_dir', 'participants', 'stages', 'conditions']
        
        for field in essential_fields:
            assert field in field_names, f"Essential field {field} missing from PipelineConfig"

    def test_config_override_functionality(self):
        """Test parameter override works for research flexibility."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        base_config = {
            "paths": {
                "raw_data_dir": str(test_data_dir / "brainvision"),
                "results_dir": str(test_data_dir / "results"),
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": ["load_data"],
            "conditions": []
        }
        
        # Test override capability
        override_params = {"paths": {"file_extension": ".edf"}}
        config = load_config(base_config, override_params)
        
        # Should handle overrides without error
        assert isinstance(config, PipelineConfig)