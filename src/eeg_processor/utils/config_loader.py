from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import yaml
from ..state_management.participant_handler import Participant

@dataclass
class PipelineConfig:
    raw_data_dir: Path
    interim_dir: Path
    results_dir: Path
    figures_dir: Path
    file_extension: str
    participants: Union[Dict[str, str], List[str]]  # Raw participants data
    stages: List[Union[str, Dict]]
    conditions: List[Dict]


def load_config(config_path: str) -> PipelineConfig:
    """Load and validate YAML config file."""
    config_path = Path(config_path).resolve()
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)

    return validate_config(raw_config, config_path.parent)


def load_config_from_dict(config_data: Dict, base_path: Path = None) -> PipelineConfig:
    """Load and validate config from dictionary data."""
    if base_path is None:
        base_path = Path.cwd()
    
    return validate_config(config_data, base_path)


def validate_config(raw_config: Dict, config_base: Path) -> PipelineConfig:
    """Validate and convert raw config dict to PipelineConfig."""
    paths = raw_config.get('paths', {})

    # Convert paths to absolute
    abs_paths = {
        k: (config_base / v).resolve() if isinstance(v, str) else v
        for k, v in paths.items()
        if k != 'participants'  # Handle participants separately
    }

    # Handle participants - pass raw data to ParticipantHandler for processing
    participants_data = None

    # Get the correct raw data path key
    raw_data_dir = abs_paths.get('raw_data_dir')
    if not raw_data_dir:
        raise ValueError("Config must specify 'raw_data_dir' path")

    if 'participants' in paths:
        participants_data = paths['participants']
    else:
        # Auto-discovery case - return list of filenames for ParticipantHandler
        ext = paths.get('file_extension', '.vhdr')
        file_paths = sorted(list(raw_data_dir.glob(f"*{ext}")) +
                            list(raw_data_dir.glob(f"**/*{ext}")))
        participants_data = [fp.name for fp in file_paths]  # Just pass filenames

    if not participants_data:
        raise ValueError("No participants found. Check participants specification or raw_data path.")

    # Validate conditions if present
    conditions = raw_config.get('conditions', [])
    for cond in conditions:
        if not all(k in cond for k in ('name',)):  # Only name is required
            raise ValueError(f"Condition {cond.get('name', 'unnamed')} missing required 'name' key")

        # Validate condition markers if present
        if 'condition_markers' in cond:
            markers = cond['condition_markers']
            if not isinstance(markers, list) or len(markers) != 2:
                raise ValueError(
                    f"Condition {cond['name']}: condition_markers must be a list of 2 elements [start, end]")

    return PipelineConfig(
        raw_data_dir=raw_data_dir,  # Use the corrected variable
        interim_dir=abs_paths.get('interim_dir'),
        results_dir=abs_paths.get('results_dir'),
        figures_dir=abs_paths.get('figures_dir'),
        file_extension=paths.get('file_extension', '.vhdr'),
        participants=participants_data,  # Pass raw data to ParticipantHandler
        stages=raw_config.get('stages', []),
        conditions=conditions
    )


def get_participant_by_id(config: PipelineConfig, participant_id: str) -> Optional[Participant]:
    """Get participant by ID from config"""
    for participant in config.participants:
        if participant.id == participant_id:
            return participant
    return None


def get_participant_ids_union(config_paths: List[str]) -> List[str]:
    """
    Get union of all participant IDs across multiple configs
    Useful for session-based processing
    """
    from ..state_management.participant_handler import ParticipantHandler

    all_participant_ids = set()

    for config_path in config_paths:
        config = load_config(config_path)
        # Create temporary participant handler to get IDs
        temp_handler = ParticipantHandler(config)
        all_participant_ids.update(temp_handler.get_participant_ids())

    return sorted(list(all_participant_ids))


def validate_session_configs(config_paths: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Validate and return participant mapping across multiple configs
    Returns: {participant_id: {config_name: filename}}
    """
    from ..state_management.participant_handler import ParticipantHandler

    participant_mapping = {}

    for config_path in config_paths:
        config_path_obj = Path(config_path)
        config_name = config_path_obj.stem.replace("_processing_params", "").replace("RIEEG_", "")
        config = load_config(config_path)

        # Create temporary participant handler to resolve participants
        temp_handler = ParticipantHandler(config)

        for participant in temp_handler.participants:
            if participant.id not in participant_mapping:
                participant_mapping[participant.id] = {}
            participant_mapping[participant.id][config_name] = participant.file_path.name

    return participant_mapping