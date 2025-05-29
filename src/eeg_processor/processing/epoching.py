# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import yaml
import mne
from mne import Epochs
from mne.io import BaseRaw


def create_epochs(raw: BaseRaw,
                  condition: dict,
                  tmin: float,
                  tmax: float,
                  baseline: tuple,
                  reject_bad: bool = True,
                  reject: Optional[Dict[str, float]] = None,
                  flat: Optional[Dict[str, float]] = None,
                  inplace: bool = False,  # Parameter for consistency, but ignored
                  **kwargs) -> Epochs:
    """
    Create epochs from Raw data - always returns new Epochs object

    Args:
        raw: Raw EEG data
        condition: Dictionary containing epoch events (from YAML)
        tmin: Start time before event (seconds)
        tmax: End time after event (seconds)
        baseline: Baseline correction period (tuple)
        reject_bad: Whether to reject bad epochs
        reject: Rejection thresholds (e.g., {'eeg': 100e-6})
        flat: Flat signal thresholds (e.g., {'eeg': 5e-6})
        inplace: Ignored - epoching always creates new object
        **kwargs: Additional epoching parameters

    Returns:
        Epochs object with artifact rejection applied

    Note: The inplace parameter is accepted for API consistency but ignored
          since epoching necessarily creates a new Epochs object from Raw data.
    """
    if inplace:
        logger.warning("inplace=True ignored for epoching - always creates new Epochs object")

    # Set default rejection thresholds if not provided
    default_reject = {
        'eeg': 100e-6,
        'Fp1': 1,  # High threshold (effectively ignored)
        'Fp2': 1  # High threshold (effectively ignored)
    }
    default_flat = {'eeg': 5e-6}

    reject_params = reject if reject is not None else (default_reject if reject_bad else None)
    flat_params = flat if flat is not None else (default_flat if reject_bad else None)

    if reject_params:
        reject_params = {key: float(value) for key, value in reject_params.items()}
    if flat_params:
        flat_params = {key: float(value) for key, value in flat_params.items()}

    # Create event dictionary from condition
    from mne import events_from_annotations
    events, event_dict = events_from_annotations(raw)
    event_id = {}

    for key, value in condition['epoch_events'].items():
        if isinstance(value, dict):  # Nested case
            for sub_key, sub_value in value.items():
                event_id[f"{sub_key}"] = int(sub_value)
        else:  # Flat case
            event_id[key] = event_dict[str(value)]

    # Create epochs
    epochs = Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject_params,
        flat=flat_params,
        preload=True,
        **kwargs
    )

    logger.info(f"Created {len(epochs)} epochs from {len(event_id)} event types")
    return epochs

def generate_annotations_from_config(raw, condition):
    """
    Dynamically generate MNE annotations based on nested event groups in YAML.

    Args:
        raw: MNE Raw object containing EEG data.
        condition: Dictionary defining condition with nested event groups.

    Returns:
        MNE Raw object with annotations added (if applicable).
    """
    annotations_list = []

    # Check if the event groups are nested
    if any(isinstance(events, dict) for events in condition.get("epoch_events", {}).values()):
        for event_type, events in condition["epoch_events"].items():
            if isinstance(events, dict):  # Only process nested dictionaries
                for event_label, event_code in events.items():
                    from mne import Epochs, events_from_annotations
                    event_times, _ = events_from_annotations(raw)
                    for time in event_times[:, 0]:
                        annotations_list.append((time, 0.1, f"{event_type}_{event_label}"))  # Mark event type

        # Convert annotations list to MNE Annotations object
        annotations = mne.Annotations(
            onset=[entry[0] for entry in annotations_list],
            duration=[entry[1] for entry in annotations_list],
            description=[entry[2] for entry in annotations_list]
        )
        raw.set_annotations(annotations)

    return raw  # Return modified raw data (with annotations if nested)


import pandas as pd


def extract_event_metadata(epochs, condition):
    """
    Extract metadata from epoch events while preserving original order.

    Args:
        epochs: MNE Epochs object to process
        condition: Dictionary containing event definitions

    Returns:
        pd.DataFrame: Metadata DataFrame matching epoch order
    """
    # Create mapping from event codes to their labels and types
    code_map = {}
    for event_type, events in condition.get('epoch_events', {}).items():
        if isinstance(events, dict):
            # Nested structure (e.g., Standards: {A: 1, B: 2})
            for label, code in events.items():
                code_map[int(code)] = (event_type, label)
        else:
            # Flat structure (e.g., Targets: 3)
            code_map[int(events)] = (event_type, str(events))

    metadata_entries = []

    # Process events in original order
    for event_code in epochs.events[:, 2]:
        try:
            event_type, event_label = code_map[event_code]
        except KeyError:
            raise ValueError(f"Event code {event_code} not found in condition definition")

        metadata_entries.append({
            'event_type': event_type,
            'event_label': event_label
        })

    return pd.DataFrame(metadata_entries)