"""Fixed tests for config_loader module - matching actual codebase structure."""

import tempfile
import pytest
import yaml
from pathlib import Path

from eeg_processor.utils.config_loader import load_config, PipelineConfig
from eeg_processor.utils.exceptions import ConfigurationError, ValidationError


class TestConfigLoader:
    """Test cases for config_loader functionality."""

    def test_load_complete_valid_config(self):
        """Test loading a complete, valid configuration."""
        # Create test directories that actually exist
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_data = {
            "study": {
                "name": "Test Study",
                "dataset": "test"
            },
            "paths": {
                "raw_data": str(test_data_dir / "brainvision"),
                "results": str(test_data_dir / "results"),
                "file_extension": ".vhdr"
            },
            "participants": {"sub-01": "test.vhdr"},
            "processing": ["load_data"],
            "conditions": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert isinstance(config, PipelineConfig)
            assert config.file_extension == ".vhdr"
            assert "sub-01" in config.participants
        finally:
            Path(temp_path).unlink()

    def test_missing_required_config_sections(self):
        """Test validation catches missing required sections."""
        incomplete_config = {
            "paths": {
                "raw_data": "/tmp/empty_directory_that_doesnt_exist"
                # This will cause no participants to be found
            }
            # Missing: study, participants, processing  
        }
        
        with pytest.raises(ValidationError, match="Missing required configuration keys"):
            load_config(incomplete_config)

    def test_missing_paths_section(self):
        """Test validation catches missing paths section."""
        config_without_paths = {
            "study": {"name": "Test"},
            "participants": {"sub-01": "test.vhdr"},
            "processing": ["load_data"],
            "conditions": []
        }
        
        with pytest.raises(ValidationError, match="Missing required configuration keys"):
            load_config(config_without_paths)

    def test_nonexistent_raw_data_directory(self):
        """Test validation catches nonexistent raw data directory."""
        config_with_bad_path = {
            "study": {"name": "Test"},
            "paths": {
                "raw_data": "/definitely/nonexistent/directory",
                "results": "/tmp/results"
            },
            "participants": {"sub-01": "test.vhdr"},
            "processing": ["load_data"],
            "conditions": []
        }
        
        with pytest.raises(ValidationError, match="Raw data directory does not exist"):
            load_config(config_with_bad_path)

    def test_config_file_not_found(self):
        """Test appropriate error when config file doesn't exist."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("/nonexistent/path/config.yml")

    def test_invalid_yaml_syntax(self):
        """Test handling of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid: yaml: syntax: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Invalid YAML syntax"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_config_override_parameters(self):
        """Test configuration parameter override functionality."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        base_config = {
            "study": {"name": "Test Study"},
            "paths": {
                "raw_data": str(test_data_dir / "brainvision"),
                "results": str(test_data_dir / "results"),
                "file_extension": ".vhdr"
            },
            "participants": {"sub-01": "test.vhdr"},
            "processing": ["load_data"],
            "conditions": []
        }
        
        # Test with override parameters
        override_params = {
            "paths": {"file_extension": ".edf"}
        }
        
        config = load_config(base_config, override_params)
        
        # Override should take effect
        assert config.file_extension == ".edf"

    def test_dataset_name_optional(self):
        """Test that dataset_name is optional in configuration."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_with_dataset = {
            "study": {
                "name": "Test Study",
                "dataset": "test_dataset"
            },
            "paths": {
                "raw_data": str(test_data_dir / "brainvision"),
                "results": str(test_data_dir / "results")
            },
            "participants": {"sub-01": "test.vhdr"},
            "processing": ["load_data"],
            "conditions": []
        }
        
        config = load_config(config_with_dataset)
        assert config.dataset_name == "test_dataset"


class TestPipelineConfig:
    """Test cases for PipelineConfig dataclass."""
    
    def create_test_config(self):
        """Helper method to create a test PipelineConfig instance."""
        from pathlib import Path
        return PipelineConfig(
            raw_data_dir=Path("/test/raw"),
            results_dir=Path("/test/results"),
            file_extension=".vhdr",
            participants={"sub-01": "test.vhdr"},
            stages=[{"load_data": {}}],
            conditions=[],
            study_info={"name": "Test"},
            output_config={},
            dataset_name="test"
        )

    def test_pipeline_config_has_required_fields(self):
        """Test that PipelineConfig has all required fields."""
        from dataclasses import fields
        
        field_names = [f.name for f in fields(PipelineConfig)]
        required_fields = [
            'raw_data_dir', 'results_dir', 'file_extension', 
            'participants', 'stages', 'conditions', 'study_info', 'output_config'
        ]
        
        for field in required_fields:
            assert field in field_names, f"Required field {field} missing from PipelineConfig"
        
        # Test that properties exist
        config = self.create_test_config()
        assert hasattr(config, 'processed_dir'), "processed_dir property missing"
        assert hasattr(config, 'figures_dir'), "figures_dir property missing"
        assert hasattr(config, 'reports_dir'), "reports_dir property missing"

    def test_pipeline_config_optional_dataset_name(self):
        """Test that dataset_name is optional with default None."""
        from dataclasses import fields
        
        dataset_field = next(f for f in fields(PipelineConfig) if f.name == 'dataset_name')
        assert dataset_field.default is None