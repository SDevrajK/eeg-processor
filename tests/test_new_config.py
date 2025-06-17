"""Test the new configuration structure."""

import tempfile
import pytest
from pathlib import Path
import yaml

from eeg_processor.utils.config_loader_new import (
    load_config, PipelineConfig, StudyInfo, ParticipantInfo, 
    Condition, ProcessingStep, OutputConfig, create_example_config
)
from eeg_processor.utils.exceptions import ConfigurationError, ValidationError


class TestNewConfigStructure:
    """Test the new configuration structure."""
    
    def test_new_format_config_loading(self):
        """Test loading configuration in new format."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_data = {
            "study": {
                "name": "Test_Study",
                "dataset": "experimental",
                "description": "Test configuration",
                "researcher": "Test Lab"
            },
            "participants": {
                "sub-01": {
                    "file": "test.vhdr",
                    "age": 25,
                    "gender": "F",
                    "group": "control"
                },
                "sub-02": "test.vhdr"  # Backward compatibility
            },
            "conditions": [
                {
                    "name": "Rest",
                    "description": "Resting state",
                    "triggers": {"start": 1, "end": 2},
                    "markers": [1, 9]
                }
            ],
            "paths": {
                "raw_data": str(test_data_dir / "brainvision"),
                "processed": str(test_data_dir / "processed"),
                "figures": str(test_data_dir / "figures"),
                "reports": str(test_data_dir / "reports")
            },
            "processing": [
                {"load_data": {}},
                {"filter": {"highpass": 0.1, "lowpass": 40}}
            ],
            "output": {
                "save_intermediate": True,
                "generate_plots": True,
                "quality_report": True
            }
        }
        
        config = load_config(config_data)
        
        # Test study info
        assert isinstance(config.study, StudyInfo)
        assert config.study.name == "Test_Study"
        assert config.study.dataset == "experimental"
        
        # Test participants with metadata
        assert len(config.participants) == 2
        assert config.participants[0].id == "sub-01"
        assert config.participants[0].file == "test.vhdr"
        assert config.participants[0].metadata["age"] == 25
        assert config.participants[0].metadata["gender"] == "F"
        
        # Test backward compatibility
        assert config.participants[1].id == "sub-02"
        assert config.participants[1].file == "test.vhdr"
        assert config.participants[1].metadata == {}
        
        # Test conditions
        assert len(config.conditions) == 1
        assert config.conditions[0].name == "Rest"
        assert config.conditions[0].triggers["start"] == 1
        
        # Test processing steps
        assert len(config.processing_steps) == 2
        assert config.processing_steps[0].name == "load_data"
        assert config.processing_steps[1].name == "filter"
        assert config.processing_steps[1].parameters["highpass"] == 0.1
    
    def test_legacy_format_compatibility(self):
        """Test that legacy format still works."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        # Legacy format
        legacy_config = {
            "paths": {
                "raw_data_dir": str(test_data_dir / "brainvision"),
                "results_dir": str(test_data_dir / "results"),
                "participants": {"sub-01": "test.vhdr"}
            },
            "stages": ["load_data"],
            "conditions": []
        }
        
        config = load_config(legacy_config)
        
        # Should convert to new structure
        assert isinstance(config, PipelineConfig)
        assert isinstance(config.study, StudyInfo)
        assert len(config.participants) == 1
        assert config.participants[0].id == "sub-01"
    
    def test_participant_flexible_metadata(self):
        """Test flexible metadata system for participants."""
        config_data = {
            "study": {"name": "Test_Study"},
            "participants": {
                "S_001_F": {
                    "file": "S_001_F.fif",
                    "age": 25,
                    "gender": "F",
                    "group": "control",
                    "tinnitus_severity": 7,
                    "hearing_threshold": 20,
                    "custom_field": "any_value"
                }
            },
            "conditions": [],
            "paths": {"raw_data": str(Path(__file__).parent / "test_data" / "brainvision")},
            "processing": []
        }
        
        config = load_config(config_data)
        
        participant = config.participants[0]
        assert participant.id == "S_001_F"
        assert participant.file == "S_001_F.fif"
        assert participant.metadata["age"] == 25
        assert participant.metadata["gender"] == "F"
        assert participant.metadata["tinnitus_severity"] == 7
        assert participant.metadata["custom_field"] == "any_value"
    
    def test_processing_pipeline_structure(self):
        """Test new processing pipeline structure."""
        config_data = {
            "study": {"name": "Test_Study"},
            "participants": {"sub-01": "test.vhdr"},
            "conditions": [],
            "paths": {"raw_data": str(Path(__file__).parent / "test_data" / "brainvision")},
            "processing": [
                "load_data",  # Simple format
                {"filter": {"highpass": 0.1, "lowpass": 40}},  # With parameters
                {"detect_bad_channels": {"threshold": 2.0, "interpolate": True}},
                {"rereference": {"method": "average", "exclude": ["VEOG", "HEOG"]}}
            ]
        }
        
        config = load_config(config_data)
        
        steps = config.processing_steps
        assert len(steps) == 4
        
        # Simple format
        assert steps[0].name == "load_data"
        assert steps[0].parameters == {}
        
        # With parameters
        assert steps[1].name == "filter"
        assert steps[1].parameters["highpass"] == 0.1
        assert steps[1].parameters["lowpass"] == 40
        
        assert steps[2].name == "detect_bad_channels"
        assert steps[2].parameters["threshold"] == 2.0
        assert steps[2].parameters["interpolate"] == True
    
    def test_example_config_creation(self):
        """Test example config creation helper."""
        example = create_example_config()
        
        assert "study" in example
        assert "participants" in example
        assert "conditions" in example
        assert "paths" in example
        assert "processing" in example
        assert "output" in example
        
        # Should be loadable
        config = load_config(example)
        assert isinstance(config, PipelineConfig)
    
    def test_config_validation_errors(self):
        """Test validation error handling."""
        # Missing study name
        with pytest.raises(ValidationError, match="Study name is required"):
            load_config({"study": {}, "paths": {"raw_data": "/tmp"}})
        
        # Missing raw_data path
        with pytest.raises(ValidationError, match="Missing required paths"):
            load_config({"study": {"name": "test"}, "paths": {}})
        
        # Invalid participant without file
        with pytest.raises(ValidationError, match="missing required 'file' field"):
            load_config({
                "study": {"name": "test"},
                "paths": {"raw_data": str(Path(__file__).parent / "test_data" / "brainvision")},
                "participants": {"sub-01": {"age": 25}}  # Missing file
            })
    
    def test_config_file_loading(self):
        """Test loading config from YAML file."""
        test_data_dir = Path(__file__).parent / "test_data"
        
        config_data = {
            "study": {
                "name": "File_Test_Study",
                "dataset": "test"
            },
            "participants": {
                "sub-01": {
                    "file": "test.vhdr",
                    "age": 25
                }
            },
            "conditions": [],
            "paths": {
                "raw_data": str(test_data_dir / "brainvision")
            },
            "processing": [{"load_data": {}}]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.study.name == "File_Test_Study"
            assert len(config.participants) == 1
            assert config.participants[0].metadata["age"] == 25
        finally:
            Path(temp_path).unlink()


class TestDataClassStructures:
    """Test the new dataclass structures."""
    
    def test_study_info_creation(self):
        """Test StudyInfo dataclass."""
        study = StudyInfo(
            name="Test Study",
            dataset="experimental",
            description="Test description",
            researcher="Test Lab"
        )
        
        assert study.name == "Test Study"
        assert study.dataset == "experimental"
        assert study.description == "Test description"
        assert study.researcher == "Test Lab"
    
    def test_participant_info_creation(self):
        """Test ParticipantInfo dataclass."""
        participant = ParticipantInfo(
            id="sub-01",
            file="sub-01.vhdr",
            metadata={"age": 25, "gender": "F"}
        )
        
        assert participant.id == "sub-01"
        assert participant.file == "sub-01.vhdr"
        assert participant.metadata["age"] == 25
        assert participant.metadata["gender"] == "F"
    
    def test_condition_creation(self):
        """Test Condition dataclass."""
        condition = Condition(
            name="Rest",
            description="Resting state",
            triggers={"start": 1, "end": 2},
            markers=[1, 9]
        )
        
        assert condition.name == "Rest"
        assert condition.description == "Resting state"
        assert condition.triggers["start"] == 1
        assert condition.markers == [1, 9]
    
    def test_processing_step_creation(self):
        """Test ProcessingStep dataclass."""
        step = ProcessingStep(
            name="filter",
            parameters={"highpass": 0.1, "lowpass": 40}
        )
        
        assert step.name == "filter"
        assert step.parameters["highpass"] == 0.1
        assert step.parameters["lowpass"] == 40