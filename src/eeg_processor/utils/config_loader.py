from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import yaml
from .exceptions import ConfigurationError, ValidationError
from ..state_management.participant_handler import Participant

@dataclass
class PipelineConfig:
    raw_data_dir: Path
    results_dir: Path
    file_extension: str
    participants: Union[Dict[str, str], Dict[str, Dict[str, Any]]]  # Support both simple and detailed participant formats
    stages: List[Dict[str, Any]]  # Processing pipeline steps
    conditions: List[Dict[str, Any]]
    study_info: Dict[str, Any]
    output_config: Dict[str, Any]
    dataset_name: Optional[str] = None
    
    @property
    def processed_dir(self) -> Path:
        """Backward compatibility property - returns results_dir/processed"""
        return self.results_dir / "processed"
    
    @property
    def figures_dir(self) -> Path:
        """Backward compatibility property - returns results_dir/figures"""
        return self.results_dir / "figures"
    
    @property
    def reports_dir(self) -> Path:
        """Backward compatibility property - returns results_dir/quality"""
        return self.results_dir / "quality"


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
    
    # Validate required top-level keys for new structure
    required_keys = ['study', 'participants', 'paths', 'processing']
    missing_keys = [key for key in required_keys if key not in raw_config]
    if missing_keys:
        raise ValidationError(f"Missing required configuration keys: {missing_keys}")
    
    paths = raw_config.get('paths', {})
    if not isinstance(paths, dict):
        raise ValidationError("'paths' must be a dictionary")

    # Validate required path keys
    required_path_keys = ['raw_data', 'results']
    missing_path_keys = [key for key in required_path_keys if key not in paths]
    if missing_path_keys:
        raise ValidationError(f"Missing required path keys: {missing_path_keys}")
    
    # Support legacy format for backward compatibility
    if 'processed' in paths and 'results' not in paths:
        logger.warning("Using legacy 'processed' path. Please update config to use 'results' path instead.")
        paths['results'] = paths['processed']

    # Convert paths to absolute
    try:
        abs_paths = {
            k: (config_base / v).resolve() if isinstance(v, str) else v
            for k, v in paths.items()
        }
    except Exception as e:
        raise ValidationError(f"Error resolving paths: {e}")

    # Get raw data directory
    raw_data_dir = abs_paths.get('raw_data')
    if not raw_data_dir.exists():
        raise ValidationError(f"Raw data directory does not exist: {raw_data_dir}")

    # Validate participants structure - support both simple and detailed formats
    participants_data = raw_config.get('participants', {})
    if not isinstance(participants_data, dict) or not participants_data:
        raise ValidationError("'participants' must be a non-empty dictionary")
    
    # Validate each participant entry - handle both formats
    for participant_id, participant_info in participants_data.items():
        if isinstance(participant_info, str):
            # Simple format: participant_id: "filename.ext"
            continue
        elif isinstance(participant_info, dict):
            # Detailed format: participant_id: {file: "filename.ext", age: 25, ...}
            if 'file' not in participant_info:
                raise ValidationError(f"Participant '{participant_id}' missing required 'file' property")
        else:
            raise ValidationError(f"Participant '{participant_id}' must be either a filename string or dictionary with metadata")

    # Extract study information and dataset name
    study_info = raw_config.get('study', {})
    if not isinstance(study_info, dict):
        raise ValidationError("'study' must be a dictionary")
    
    dataset_name = study_info.get('dataset')
    
    # Get results directory
    results_dir = abs_paths.get('results')
    if not results_dir:
        raise ValidationError("Results directory path is required")

    # Validate conditions
    conditions = raw_config.get('conditions', [])
    if not isinstance(conditions, list):
        raise ValidationError("'conditions' must be a list")
    
    for i, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            raise ValidationError(f"Condition {i} must be a dictionary")
        
        if 'name' not in cond:
            raise ValidationError(f"Condition {i} missing required 'name' key")
        
        # Validate triggers structure if present
        if 'triggers' in cond:
            triggers = cond['triggers']
            if not isinstance(triggers, dict):
                raise ValidationError(f"Condition '{cond['name']}': triggers must be a dictionary")
        
        # Validate markers if present
        if 'markers' in cond:
            markers = cond['markers']
            # Allow markers to be None, empty list, list of markers, or dict with trigger->marker_list mapping
            if markers is not None:
                if isinstance(markers, list):
                    # Original format: list of markers
                    for j, marker in enumerate(markers):
                        if not isinstance(marker, (str, int)):
                            raise ValidationError(f"Condition '{cond['name']}': marker {j} must be string or integer, got {type(marker)}")
                elif isinstance(markers, dict):
                    # New format: dict with trigger->marker_list mapping
                    for trigger_key, marker_list in markers.items():
                        if not isinstance(trigger_key, (str, int)):
                            raise ValidationError(f"Condition '{cond['name']}': trigger key '{trigger_key}' must be string or integer")
                        if not isinstance(marker_list, list):
                            raise ValidationError(f"Condition '{cond['name']}': marker list for trigger '{trigger_key}' must be a list")
                        for j, marker in enumerate(marker_list):
                            if not isinstance(marker, (str, int)):
                                raise ValidationError(f"Condition '{cond['name']}': marker {j} in trigger '{trigger_key}' must be string or integer, got {type(marker)}")
                else:
                    raise ValidationError(f"Condition '{cond['name']}': markers must be a list, dict, or None")

    # Validate processing pipeline
    processing_stages = raw_config.get('processing', [])
    if not isinstance(processing_stages, list):
        raise ValidationError("'processing' must be a list")
    
    # Validate output configuration
    output_config = raw_config.get('output', {})
    if not isinstance(output_config, dict):
        raise ValidationError("'output' must be a dictionary")

    return PipelineConfig(
        raw_data_dir=raw_data_dir,
        results_dir=results_dir,
        file_extension=paths.get('file_extension', '.vhdr'),
        participants=participants_data,
        stages=processing_stages,
        conditions=conditions,
        study_info=study_info,
        output_config=output_config,
        dataset_name=dataset_name
    )


def _deep_merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def get_participant_by_id(config: PipelineConfig, participant_id: str) -> Optional[Participant]:
    """Get participant by ID from config"""
    for participant in config.participants:
        if participant.id == participant_id:
            return participant
    return None
