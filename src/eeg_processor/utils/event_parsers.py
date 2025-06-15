from typing import Union, List, Tuple
import numpy as np
from mne import events_from_annotations
import re
from mne.io import BaseRaw, read_raw_brainvision, read_raw_curry


def _detect_raw_format(raw: BaseRaw) -> str:
    """Detect format directly from Raw object properties with Neuroscan support"""
    # Method 1: Check object type string
    raw_type = str(type(raw))
    if 'BrainVision' in raw_type:
        return 'brainvision'
    elif 'Curry' in raw_type:
        return 'curry'
    elif 'Neuroscan' in raw_type or 'CNT' in raw_type:
        return 'neuroscan'

    # Method 2: Parse annotations (fallback)
    if hasattr(raw, 'annotations') and raw.annotations:
        first_desc = raw.annotations.description[0]
        if isinstance(first_desc, str):
            if re.match(r'Stimulus/S\d+', first_desc) or re.match(r'Response/R\d+', first_desc):
                return 'brainvision'
            elif first_desc.isdigit():  # Neuroscan often uses pure numbers
                return 'neuroscan'
        elif isinstance(first_desc, (int, np.integer)):
            return 'curry'

    # Method 3: Filename in raw.info (last resort)
    if 'filename' in raw.info:
        filename = raw.info['filename'].lower()
        if filename.endswith('.vhdr'):
            return 'brainvision'
        elif filename.endswith('.dat'):
            return 'curry'
        elif filename.endswith('.cnt'):
            return 'neuroscan'

    raise ValueError(
        "Could not detect file format from Raw object. "
        f"Type: {type(raw)}, Annotations: {getattr(raw, 'annotations', None)}, "
        f"Filename: {raw.info.get('filename')}"
    )


def get_normalized_event_code(raw: BaseRaw, event_code: Union[str, int]) -> str:
    """
    Handles all format-specific normalization rules.
    Returns consistent string representation for matching.
    """
    # First extract digits from any input format
    if isinstance(event_code, (int, float)):
        digits = str(int(event_code))
    else:
        digits = ''.join(filter(str.isdigit, str(event_code))) or '0'

    # Detect format and apply appropriate formatting
    raw_type = str(type(raw))
    if 'BrainVision' in raw_type:
        num = int(digits)
        if num < 10:
            return f"Stimulus/S  {num}"
        elif num <= 99:
            return f"Stimulus/S {num}"
        return f"Stimulus/S{num}"
    elif 'Curry' in raw_type:
        return digits  # Curry uses plain numbers
    elif 'Neuroscan' in raw_type:
        return digits  # Neuroscan uses plain numbers
    else:
        # Fallback for unknown formats
        return digits


def find_matching_events(raw: BaseRaw, event_code: Union[str, int]) -> np.ndarray:
    """
    Universal event finder that relies entirely on get_normalized_event_code
    for format-specific handling
    """
    # Get consistent normalized code
    norm_code = get_normalized_event_code(raw, event_code)
    from mne import events_from_annotations
    events, event_dict = events_from_annotations(raw)

    # Find all matching event IDs (exact string matching)
    matching_ids = [
        event_id
        for desc, event_id in event_dict.items()
        if str(desc).strip() == str(norm_code).strip()
    ]

    if not matching_ids:
        available = list(event_dict.keys())
        raise ValueError(
            f"Event '{event_code}' (normalized: '{norm_code}') not found.\n"
            f"Available events: {available}"
        )

    return np.concatenate([events[events[:, 2] == id, 0] for id in matching_ids])