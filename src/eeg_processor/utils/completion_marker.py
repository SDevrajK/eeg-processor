"""
Completion marker system for crash-resilient pipeline resumption.

Each participant writes a small .done file after successful processing.
The file contains a hash of the processing config (stages + conditions),
so that if the config changes the participant is automatically reprocessed.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


def _compute_config_hash(stages: List[Any], conditions: List[Any]) -> str:
    """Hash the processing-relevant config fields.

    Only stages and conditions affect the output — metadata fields like
    study name or researcher are excluded intentionally.
    """
    hashable = {"stages": stages, "conditions": conditions}
    serialized = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


def _marker_path(results_dir: Path, participant_id: str) -> Path:
    return results_dir / "processed" / f"{participant_id}.done"


def write_marker(results_dir: Path, participant_id: str, stages: List[Any], conditions: List[Any]) -> None:
    """Write a .done marker file for a successfully processed participant."""
    marker = {
        "completed": True,
        "participant": participant_id,
        "config_hash": _compute_config_hash(stages, conditions),
        "timestamp": datetime.now().isoformat(),
    }
    path = _marker_path(results_dir, participant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(marker, f, indent=2)
    logger.debug(f"Wrote completion marker: {path.name}")


def is_already_processed(results_dir: Path, participant_id: str, stages: List[Any], conditions: List[Any]) -> bool:
    """Return True if participant has a valid .done file matching the current config hash."""
    path = _marker_path(results_dir, participant_id)
    if not path.exists():
        return False

    try:
        with open(path) as f:
            marker = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read completion marker for {participant_id} ({e}) — will reprocess")
        return False

    if not marker.get("completed"):
        return False

    current_hash = _compute_config_hash(stages, conditions)
    stored_hash = marker.get("config_hash", "")

    if current_hash != stored_hash:
        logger.info(
            f"Config changed since {participant_id} was last processed "
            f"(stored hash: {stored_hash[:8]}…, current: {current_hash[:8]}…) — will reprocess"
        )
        return False

    logger.info(f"Skipping {participant_id} — already processed with current config (hash: {current_hash[:8]}…)")
    return True
