from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import yaml


@dataclass
class PipelineConfig:
    raw_data_path: Path
    interim_dir: Path
    results_dir: Path
    figures_dir: Path
    file_extension: str
    participants: List[Path]
    stages: List[Union[str, Dict]]
    conditions: List[Dict]


def load_config(config_path: str) -> PipelineConfig:
    """Load and validate YAML config file."""
    config_path = Path(config_path).resolve()
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    return validate_config(raw_config, config_path.parent)


def validate_config(raw_config: Dict, config_base: Path) -> PipelineConfig:
    """Validate and convert raw config dict to PipelineConfig."""
    paths = raw_config.get('paths', {})

    # Convert paths to absolute
    abs_paths = {
        k: (config_base / v).resolve() if isinstance(v, str) else v
        for k, v in paths.items()
    }

    # Handle participants list
    if 'participants' in paths:
        participants = [(config_base / p).resolve() for p in paths['participants']]
    else:
        ext = paths.get('file_extension', '.vhdr')
        base_dir = abs_paths['raw_data']
        participants = sorted(list(base_dir.glob(f"*{ext}")) + list(base_dir.glob(f"**/*{ext}")))

    # Validate conditions if present
    conditions = raw_config.get('conditions', [])
    for cond in conditions:
        if not all(k in cond for k in ('name', 'condition_markers')):
            raise ValueError(f"Condition {cond.get('name')} missing required keys")

    return PipelineConfig(
        raw_data_path=abs_paths['raw_data'],
        interim_dir=abs_paths.get('interim_dir'),
        results_dir=abs_paths.get('results_dir'),
        figures_dir=abs_paths.get('figures_dir'),
        file_extension=paths.get('file_extension', '.vhdr'),
        participants=participants,
        stages=raw_config.get('stages', []),
        conditions=conditions
    )