import mne
from typing import Dict


def convert_to_numeric_event_id(raw: mne.io.BaseRaw, event_mapping: Dict[str, str]) -> Dict[str, int]:
    """
    Convert string event markers to numeric codes for MNE.

    Args:
        raw: Loaded raw EEG data
        event_mapping: Dictionary from config (e.g., {"Standard": "S  1"})

    Returns:
        Dictionary with numeric event codes (e.g., {"Standard": 1})

    Raises:
        ValueError: If any markers are not found in the data
    """
    _, event_id_map = mne.events_from_annotations(raw)
    numeric_id = {}

    for desc, marker in event_mapping.items():
        found = False
        for code, name in event_id_map.items():
            if name == marker:
                numeric_id[desc] = code
                found = True
                break

        if not found:
            available = list(event_id_map.values())
            raise ValueError(
                f"Event '{marker}' not found in data. "
                f"Available markers: {available}"
            )

    return numeric_id