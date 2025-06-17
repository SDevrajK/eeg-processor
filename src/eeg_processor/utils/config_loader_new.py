"""New configuration loader with improved structure for scientific research."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import yaml
from .exceptions import ConfigurationError, ValidationError


@dataclass
class StudyInfo:
    """Study information for research metadata."""
    name: str
    dataset: Optional[str] = None
    description: Optional[str] = None
    researcher: Optional[str] = None


@dataclass
class ParticipantInfo:
    """Participant information with flexible metadata."""
    id: str
    file: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Condition:
    """Experimental condition definition."""
    name: str
    description: Optional[str] = None
    triggers: Optional[Dict[str, int]] = None
    markers: Optional[List[int]] = None


@dataclass
class ProcessingStep:
    """Processing pipeline step."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Output and quality control configuration."""
    save_intermediate: bool = True
    generate_plots: bool = True
    quality_report: bool = True
    file_format: str = "fif"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration with new structure."""
    # Study information
    study: StudyInfo
    
    # Participants with flexible metadata
    participants: List[ParticipantInfo]
    
    # Experimental conditions
    conditions: List[Condition]
    
    # Data paths
    raw_data_dir: Path
    processed_dir: Path
    figures_dir: Path
    reports_dir: Path
    
    # Processing pipeline
    processing_steps: List[ProcessingStep]
    
    # Output configuration
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Legacy support
    interim_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    file_extension: str = ".vhdr"


def load_config(config_path: Union[str, Dict[str, Any]], override_params: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load and validate configuration from file or dictionary.
    
    Args:
        config_path: Path to YAML config file or config dictionary
        override_params: Optional parameters to override in config
        
    Returns:
        Validated PipelineConfig object
        
    Raises:
        ConfigurationError: If config file cannot be loaded or parsed
        ValidationError: If config validation fails
    """
    try:
        if isinstance(config_path, dict):
            raw_config = config_path.copy()
            config_base = Path.cwd()
        else:
            config_path = Path(config_path).resolve()
            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
            except UnicodeDecodeError as e:
                raise ConfigurationError(f"Unable to decode config file {config_path}: {e}")
            
            config_base = config_path.parent
        
        # Apply overrides if provided
        if override_params:
            raw_config = _deep_merge_config(raw_config, override_params)
            
        return validate_config(raw_config, config_base)
        
    except (ConfigurationError, ValidationError):
        raise
    except Exception as e:
        raise ConfigurationError(f"Unexpected error loading config: {e}")


def validate_config(raw_config: Dict, config_base: Path) -> PipelineConfig:
    """Validate and convert raw config dict to PipelineConfig.
    
    Args:
        raw_config: Raw configuration dictionary
        config_base: Base directory for resolving relative paths
        
    Returns:
        Validated PipelineConfig object
        
    Raises:
        ValidationError: If configuration validation fails
    """
    if not isinstance(raw_config, dict):
        raise ValidationError("Configuration must be a dictionary")
    
    # Check if this is the new format or legacy format
    has_new_format = 'study' in raw_config or 'processing' in raw_config
    
    if has_new_format:
        return _validate_new_format(raw_config, config_base)
    else:
        return _validate_legacy_format(raw_config, config_base)


def _validate_new_format(raw_config: Dict, config_base: Path) -> PipelineConfig:
    """Validate new configuration format."""
    
    # Validate study information
    study_info = raw_config.get('study', {})
    if not isinstance(study_info, dict):
        raise ValidationError("'study' must be a dictionary")
    
    if 'name' not in study_info:
        raise ValidationError("Study name is required")
    
    study = StudyInfo(
        name=study_info['name'],
        dataset=study_info.get('dataset'),
        description=study_info.get('description'),
        researcher=study_info.get('researcher')
    )
    
    # Validate paths
    paths = raw_config.get('paths', {})
    if not isinstance(paths, dict):
        raise ValidationError("'paths' must be a dictionary")
    
    required_paths = ['raw_data']
    missing_paths = [p for p in required_paths if p not in paths]
    if missing_paths:
        raise ValidationError(f"Missing required paths: {missing_paths}")
    
    # Convert paths to absolute
    raw_data_dir = (config_base / paths['raw_data']).resolve()
    if not raw_data_dir.exists():
        raise ValidationError(f"Raw data directory does not exist: {raw_data_dir}")
    
    processed_dir = (config_base / paths.get('processed', 'processed')).resolve()
    figures_dir = (config_base / paths.get('figures', 'figures')).resolve()
    reports_dir = (config_base / paths.get('reports', 'reports')).resolve()
    
    # Validate participants
    participants = _validate_participants(raw_config.get('participants', {}))
    
    # Validate conditions
    conditions = _validate_conditions(raw_config.get('conditions', []))
    
    # Validate processing steps
    processing_steps = _validate_processing_steps(raw_config.get('processing', []))
    
    # Validate output configuration
    output_config = _validate_output_config(raw_config.get('output', {}))
    
    return PipelineConfig(
        study=study,
        participants=participants,
        conditions=conditions,
        raw_data_dir=raw_data_dir,
        processed_dir=processed_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        processing_steps=processing_steps,
        output=output_config
    )


def _validate_legacy_format(raw_config: Dict, config_base: Path) -> PipelineConfig:
    """Validate legacy configuration format and convert to new structure."""
    
    # Import the old validation function
    from .config_loader import validate_config as legacy_validate
    
    # Use legacy validation
    legacy_config = legacy_validate(raw_config, config_base)
    
    # Convert to new format
    study = StudyInfo(
        name=legacy_config.dataset_name or "Unnamed_Study",
        dataset=legacy_config.dataset_name
    )
    
    # Convert legacy participants to new format
    participants = []
    if isinstance(legacy_config.participants, dict):
        for participant_id, file_info in legacy_config.participants.items():
            if isinstance(file_info, str):
                # Simple format: participant_id: "filename"
                participants.append(ParticipantInfo(
                    id=participant_id,
                    file=file_info,
                    metadata={}
                ))
            elif isinstance(file_info, dict):
                # New format already embedded in legacy
                participants.append(ParticipantInfo(
                    id=participant_id,
                    file=file_info.get('file', ''),
                    metadata={k: v for k, v in file_info.items() if k != 'file'}
                ))
    
    # Convert legacy conditions
    conditions = []
    for cond in legacy_config.conditions:
        conditions.append(Condition(
            name=cond.get('name', ''),
            description=cond.get('description'),
            triggers=cond.get('epoch_events'),
            markers=cond.get('condition_markers')
        ))
    
    # Convert legacy stages to processing steps
    processing_steps = []
    for stage in legacy_config.stages:
        if isinstance(stage, str):
            processing_steps.append(ProcessingStep(name=stage))
        elif isinstance(stage, dict):
            for step_name, params in stage.items():
                processing_steps.append(ProcessingStep(
                    name=step_name,
                    parameters=params if params else {}
                ))
    
    return PipelineConfig(
        study=study,
        participants=participants,
        conditions=conditions,
        raw_data_dir=legacy_config.raw_data_dir,
        processed_dir=legacy_config.results_dir,
        figures_dir=legacy_config.figures_dir,
        reports_dir=legacy_config.results_dir,  # Use results_dir as fallback
        processing_steps=processing_steps,
        output=OutputConfig(),
        # Keep legacy fields for backward compatibility
        interim_dir=legacy_config.interim_dir,
        results_dir=legacy_config.results_dir,
        file_extension=legacy_config.file_extension
    )


def _validate_participants(participants_data: Union[Dict, List]) -> List[ParticipantInfo]:
    """Validate participants with flexible metadata structure."""
    if not participants_data:
        raise ValidationError("Participants section is required")
    
    participants = []
    
    if isinstance(participants_data, dict):
        for participant_id, file_info in participants_data.items():
            if isinstance(file_info, str):
                # Backward compatibility: participant_id: "filename"
                participants.append(ParticipantInfo(
                    id=participant_id,
                    file=file_info,
                    metadata={}
                ))
            elif isinstance(file_info, dict):
                # New format: participant_id: {file: "filename", metadata...}
                if 'file' not in file_info:
                    raise ValidationError(f"Participant {participant_id} missing required 'file' field")
                
                metadata = {k: v for k, v in file_info.items() if k != 'file'}
                participants.append(ParticipantInfo(
                    id=participant_id,
                    file=file_info['file'],
                    metadata=metadata
                ))
            else:
                raise ValidationError(f"Participant {participant_id} must be string or dictionary")
    else:
        raise ValidationError("Participants must be a dictionary")
    
    return participants


def _validate_conditions(conditions_data: List) -> List[Condition]:
    """Validate experimental conditions."""
    if not isinstance(conditions_data, list):
        raise ValidationError("Conditions must be a list")
    
    conditions = []
    for i, cond_data in enumerate(conditions_data):
        if not isinstance(cond_data, dict):
            raise ValidationError(f"Condition {i} must be a dictionary")
        
        if 'name' not in cond_data:
            raise ValidationError(f"Condition {i} missing required 'name' field")
        
        conditions.append(Condition(
            name=cond_data['name'],
            description=cond_data.get('description'),
            triggers=cond_data.get('triggers'),
            markers=cond_data.get('markers')
        ))
    
    return conditions


def _validate_processing_steps(processing_data: List) -> List[ProcessingStep]:
    """Validate processing pipeline steps."""
    if not isinstance(processing_data, list):
        raise ValidationError("Processing must be a list")
    
    processing_steps = []
    for i, step_data in enumerate(processing_data):
        if isinstance(step_data, str):
            # Simple format: just step name
            processing_steps.append(ProcessingStep(name=step_data))
        elif isinstance(step_data, dict):
            # Format: {step_name: {parameters}}
            if len(step_data) != 1:
                raise ValidationError(f"Processing step {i} must have exactly one key-value pair")
            
            step_name, params = next(iter(step_data.items()))
            processing_steps.append(ProcessingStep(
                name=step_name,
                parameters=params if params else {}
            ))
        else:
            raise ValidationError(f"Processing step {i} must be string or dictionary")
    
    return processing_steps


def _validate_output_config(output_data: Dict) -> OutputConfig:
    """Validate output configuration."""
    if not isinstance(output_data, dict):
        raise ValidationError("Output configuration must be a dictionary")
    
    return OutputConfig(
        save_intermediate=output_data.get('save_intermediate', True),
        generate_plots=output_data.get('generate_plots', True),
        quality_report=output_data.get('quality_report', True),
        file_format=output_data.get('file_format', 'fif')
    )


def _deep_merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration in the new format."""
    return {
        "study": {
            "name": "Example_EEG_Study",
            "dataset": "experimental",
            "description": "Example EEG analysis configuration",
            "researcher": "Research Lab"
        },
        
        "participants": {
            "sub-01": {
                "file": "sub-01.vhdr",
                "age": 25,
                "gender": "F",
                "group": "control"
            },
            "sub-02": {
                "file": "sub-02.vhdr", 
                "age": 30,
                "gender": "M",
                "group": "experimental"
            },
            # Backward compatibility
            "sub-03": "sub-03.vhdr"
        },
        
        "conditions": [
            {
                "name": "Rest",
                "description": "Resting state condition",
                "triggers": {"start": 1, "end": 2},
                "markers": [1, 9]
            },
            {
                "name": "Task",
                "description": "Active task condition", 
                "triggers": {"start": 10, "end": 20},
                "markers": [10, 19]
            }
        ],
        
        "paths": {
            "raw_data": "data/raw/",
            "processed": "data/processed/",
            "figures": "results/figures/",
            "reports": "results/reports/"
        },
        
        "processing": [
            {"load_data": {}},
            {"filter": {"highpass": 0.1, "lowpass": 40, "notch": 50}},
            {"detect_bad_channels": {"threshold": 2.0}},
            {"rereference": {"method": "average"}},
            {"remove_artifacts": {"method": "ica"}},
            # Alternative ASR processing:
            # {"remove_artifacts_asr": {"cutoff": 20, "method": "euclid"}},
            {"create_epochs": {"time_window": [-0.2, 1.0], "baseline": [-0.2, 0]}},
            {"time_frequency_analysis": {"method": "morlet", "frequencies": [1, 40]}}
        ],
        
        "output": {
            "save_intermediate": True,
            "generate_plots": True,
            "quality_report": True,
            "file_format": "fif"
        }
    }