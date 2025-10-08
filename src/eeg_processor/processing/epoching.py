# Standard library
from loguru import logger
from typing import Dict, List, Optional, Union
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
                  preload: bool = True,  # Control memory usage
                  picks: Optional[Union[str, List[str]]] = None,  # Channel selection
                  **kwargs) -> Epochs:
    """
    Create epochs from Raw data following MNE best practices.

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
        preload: Whether to load epoch data into memory immediately
        picks: Channel selection (None for all channels)
        **kwargs: Additional epoching parameters

    Returns:
        Epochs object with artifact rejection applied

    Note: The inplace parameter is accepted for API consistency but ignored
          since epoching necessarily creates a new Epochs object from Raw data.
    """
    if inplace:
        logger.warning("inplace=True ignored for epoching - always creates new Epochs object")

    # Set default rejection thresholds only for channel types present in the data
    channel_types_in_data = set(ch['kind'] for ch in raw.info['chs'])
    ch_type_mapping = {2: 'eeg', 202: 'eog', 102: 'emg'}  # FIFF channel type codes
    
    default_reject = {}
    default_flat = {}
    
    # Only add rejection criteria for channel types that exist
    for ch_type_code, ch_type_name in ch_type_mapping.items():
        if ch_type_code in channel_types_in_data:
            if ch_type_name == 'eeg':
                default_reject[ch_type_name] = 100e-6  # 100 µV
                default_flat[ch_type_name] = 1e-6      # 1 µV
            elif ch_type_name == 'eog':
                default_reject[ch_type_name] = 150e-6  # 150 µV
                default_flat[ch_type_name] = 1e-6      # 1 µV

    reject_params = reject if reject is not None else (default_reject if reject_bad else None)
    flat_params = flat if flat is not None else (default_flat if reject_bad else None)

    # Convert to proper data types
    if reject_params:
        reject_params = {key: float(value) for key, value in reject_params.items()}
    if flat_params:
        flat_params = {key: float(value) for key, value in flat_params.items()}

    # Extract events from raw data
    try:
        events, event_dict = mne.events_from_annotations(raw)
        if len(events) == 0:
            logger.warning("No events found in raw data annotations")
            # Try to find events in the data itself
            events = mne.find_events(raw, shortest_event=1)
            if len(events) == 0:
                raise ValueError("No events found in raw data")
    except Exception as e:
        logger.error(f"Error extracting events: {e}")
        raise

    # Build event_id dictionary from condition
    event_id = {}
    triggers = condition.get('triggers', condition.get('epoch_events', {}))
    
    for key, value in triggers.items():
        if isinstance(value, dict):  # Nested case
            for sub_key, sub_value in value.items():
                event_id[f"{sub_key}"] = int(sub_value)
        else:  # Flat case
            if isinstance(value, str) and value in event_dict:
                event_id[key] = event_dict[value]
            else:
                event_id[key] = int(value)

    # Validate event_id
    if not event_id:
        raise ValueError("No valid event IDs found in condition")

    # Check if events exist in the data
    available_events = set(events[:, 2])
    requested_events = set(event_id.values())
    missing_events = requested_events - available_events
    
    if missing_events:
        logger.warning(f"Events not found in data: {missing_events}")
        # Filter out missing events
        event_id = {k: v for k, v in event_id.items() if v in available_events}
        
    if not event_id:
        raise ValueError("No requested events found in the data")

    # Create epochs using MNE constructor
    try:
        epochs = Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            reject=reject_params,
            flat=flat_params,
            preload=preload,
            verbose=False,  # Control verbosity
            **kwargs
        )
        
        logger.info(f"Created {len(epochs)} epochs from {len(event_id)} event types")
        logger.info(f"Event types: {list(event_id.keys())}")
        
        # Add participant metadata to epochs if available in raw data
        _add_participant_metadata_to_epochs(epochs, raw)
        
        return epochs
        
    except Exception as e:
        logger.error(f"Error creating epochs: {e}")
        raise


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
    triggers = condition.get('triggers', {}) or condition.get('epoch_events', {})
    for event_type, events in triggers.items():
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


def _add_participant_metadata_to_epochs(epochs, raw):
    """
    Add participant metadata from raw data to epochs.metadata DataFrame.
    
    Args:
        epochs: MNE Epochs object to add metadata to
        raw: Raw object containing participant metadata in subject_info
    """
    try:
        # Extract participant metadata from raw subject_info
        subject_info = raw.info.get('subject_info')
        if not subject_info:
            return
            
        his_id = subject_info.get('his_id')
        if not his_id or '|' not in his_id:
            return
            
        # Parse metadata from his_id
        parts = his_id.split('|')
        participant_id = parts[0]
        participant_metadata = {}
        
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                participant_metadata[key] = value
        
        if not participant_metadata:
            return
            
        # Create metadata DataFrame for epochs
        n_epochs = len(epochs)
        epochs_metadata = pd.DataFrame({
            'participant_id': [participant_id] * n_epochs,
            **{key: [value] * n_epochs for key, value in participant_metadata.items()},
            'epoch_number': range(n_epochs)
        })
        
        # Add to epochs
        epochs.metadata = epochs_metadata
        
        logger.debug(f"Added participant metadata to {n_epochs} epochs")
        logger.debug(f"  Participant: {participant_id}")
        logger.debug(f"  Metadata keys: {list(participant_metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to epochs: {e}")
        # Don't raise - this is non-critical functionality