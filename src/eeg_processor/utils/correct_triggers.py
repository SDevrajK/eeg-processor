"""
Trigger correction utilities for EEG processing pipeline.

Handles cases where triggers are incorrectly coded (e.g., changed to 255)
in experimental paradigms with known patterns.
"""

import numpy as np
from mne.io import BaseRaw
from mne import events_from_annotations
from loguru import logger
from typing import Dict, Any, List, Union, Optional


def correct_triggers(raw: BaseRaw,
                    condition: Dict[str, Any],
                    method: str = "alternating",
                    corrupted_codes: Optional[List[int]] = None,
                    auto_detect_corrupted: bool = True,
                    inplace: bool = False,
                    **kwargs) -> BaseRaw:
    """
    Correct incorrectly coded triggers in EEG data.

    Args:
        raw: MNE Raw object (should be segmented to single condition)
        condition: Current condition dict with epoch_events info
        method: Correction method ('alternating' for now, extensible)
        corrupted_codes: List of specific corrupted codes to fix (e.g., [255, 127])
        auto_detect_corrupted: If True, treat any code outside expected codes as corrupted
        inplace: Whether to modify raw object in place
        **kwargs: Additional parameters for specific methods

    Returns:
        Raw object with corrected triggers
    """
    if not inplace:
        raw = raw.copy()

    if method == "alternating":
        _correct_alternating_triggers(raw, condition, corrupted_codes, auto_detect_corrupted, **kwargs)
    else:
        raise ValueError(f"Unknown trigger correction method: {method}")

    return raw


def _correct_alternating_triggers(raw: BaseRaw,
                                 condition: Dict[str, Any],
                                 corrupted_codes: Optional[List[int]] = None,
                                 auto_detect_corrupted: bool = True,
                                 **kwargs) -> None:
    """
    Correct triggers in alternating pattern paradigms.

    Handles cases where triggers are incorrectly changed to corrupted codes
    in paradigms where events alternate between two codes.

    Args:
        raw: MNE Raw object (modified in place)
        condition: Condition dict containing epoch_events
        corrupted_codes: Specific codes to treat as corrupted (e.g., [255, 127])
        auto_detect_corrupted: If True, treat any code outside expected codes as corrupted
    """
    # Get events from annotations using MNE function
    events, event_id = events_from_annotations(raw)

    if len(events) == 0:
        logger.warning("No events found in raw data - skipping trigger correction")
        return

    # Extract epoch event codes from condition (check new 'triggers' format first)
    epoch_events = condition.get("triggers", {}) or condition.get("epoch_events", {})
    if len(epoch_events) == 0 or len(epoch_events) > 2:
        raise ValueError(f"Alternating correction requires 1 or 2 epoch events, got {len(epoch_events)}")

    # Convert condition codes to actual event IDs using event_id mapping
    def get_event_id(code):
        """Convert condition code to actual event ID from event_id mapping"""
        if isinstance(code, int):
            # If it's already an integer, check if it exists in event_id values
            if code in event_id.values():
                return code
            # If not found, look for string representation
            str_code = str(code)
            if str_code in event_id:
                return event_id[str_code]
        elif isinstance(code, str):
            if code in event_id:
                return event_id[code]

        # Last resort: return the code as-is and log warning
        logger.warning(f"Could not find event ID for code {code}, using as-is")
        return code

    # Get the alternating codes
    condition_codes = list(epoch_events.values())
    onset_code = get_event_id(condition_codes[0])  # First event should be onset

    if len(condition_codes) == 1:
        # Infer the other code from the event sequence
        offset_code = _infer_alternating_code(events, onset_code)
        if offset_code is None:
            raise ValueError(f"Could not infer alternating code for provided code {onset_code}")
        logger.info(f"Inferred alternating code: {onset_code} <-> {offset_code}")
    else:
        offset_code = get_event_id(condition_codes[1])  # Second event should be offset

    expected_codes = {onset_code, offset_code}

    # Also include condition markers as valid (they shouldn't be corrected)  
    condition_markers = condition.get("markers", []) or condition.get("condition_markers", [])
    if condition_markers:
        for marker in condition_markers:
            marker_id = get_event_id(marker)
            expected_codes.add(marker_id)

    # Determine which codes to treat as corrupted
    if auto_detect_corrupted:
        # Find any codes that aren't in our expected set
        all_codes = set(events[:, 2])
        corrupted_set = all_codes - expected_codes
        logger.info(f"Auto-detected corrupted codes: {sorted(corrupted_set)}")
    else:
        corrupted_set = set(corrupted_codes) if corrupted_codes else {255}

    logger.info(f"Correcting alternating triggers: {onset_code} <-> {offset_code}")
    logger.info(f"Expected codes: {sorted(expected_codes)}")
    logger.info(f"Treating as corrupted: {sorted(corrupted_set)}")

    # Find corrupted events (only among epoch events, not condition markers)
    corrupted_indices = []
    for i, event in enumerate(events):
        event_code = event[2]
        if event_code in corrupted_set:
            corrupted_indices.append(i)

    if len(corrupted_indices) == 0:
        logger.info("No corrupted triggers found - no correction needed")
        return

    logger.info(f"Found {len(corrupted_indices)} corrupted triggers at samples: {events[corrupted_indices, 0]}")

    # Correct each corrupted event
    corrections_made = 0
    for corrupt_idx in corrupted_indices:
        expected_code = _determine_expected_code(events, corrupt_idx, onset_code, offset_code, expected_codes)

        if expected_code is not None:
            original_code = events[corrupt_idx, 2]
            sample_time = events[corrupt_idx, 0] / raw.info['sfreq']  # Convert to seconds
            logger.info(f"Corrected trigger at sample {events[corrupt_idx, 0]} ({sample_time:.3f}s): {original_code} -> {expected_code}")
            events[corrupt_idx, 2] = expected_code
            corrections_made += 1
        else:
            logger.warning(f"Could not determine expected code for corrupted event at sample {events[corrupt_idx, 0]}")

    logger.info(f"Successfully corrected {corrections_made}/{len(corrupted_indices)} corrupted triggers")

    # Create new annotations from corrected events
    _update_raw_annotations(raw, events, event_id)


def _infer_alternating_code(events: np.ndarray, known_code: int) -> Union[int, None]:
    """
    Infer the alternating code from event sequence.

    Looks for a code that appears to alternate with the known_code by finding
    which code most frequently appears adjacent to it in the sequence.

    Args:
        events: Events array [n_events, 3] with [sample, 0, event_id]
        known_code: The code we know about

    Returns:
        The inferred alternating code, or None if cannot be determined
    """
    # Find all unique codes in the events
    unique_codes = set(events[:, 2])
    unique_codes.discard(known_code)

    if len(unique_codes) == 0:
        return None

    # Find which code appears to alternate with known_code
    # by counting adjacency (how often codes appear next to known_code)
    adjacency_counts = {}
    for i in range(len(events) - 1):
        current_code = events[i, 2]
        next_code = events[i + 1, 2]

        if current_code == known_code and next_code != known_code:
            adjacency_counts[next_code] = adjacency_counts.get(next_code, 0) + 1
        elif next_code == known_code and current_code != known_code:
            adjacency_counts[current_code] = adjacency_counts.get(current_code, 0) + 1

    if not adjacency_counts:
        return None

    # Return the code with highest adjacency count
    alternating_code = max(adjacency_counts, key=adjacency_counts.get)
    logger.debug(f"Adjacency counts: {adjacency_counts}, selected: {alternating_code}")
    return alternating_code


def _determine_expected_code(events: np.ndarray,
                           corrupt_idx: int,
                           onset_code: int,
                           offset_code: int,
                           expected_codes: set) -> Union[int, None]:
    """
    Determine the expected trigger code for a corrupted event based on alternating pattern.

    Args:
        events: Events array [n_events, 3] with [sample, 0, event_id]
        corrupt_idx: Index of the corrupted event
        onset_code: Expected onset trigger code
        offset_code: Expected offset trigger code
        expected_codes: Set of all valid codes (including condition markers)

    Returns:
        Expected trigger code or None if cannot be determined
    """
    # Strategy: Look at neighboring valid epoch events to determine pattern

    # First, try to look at the previous valid epoch event (not condition markers)
    prev_code = _find_previous_valid_code(events, corrupt_idx, onset_code, offset_code)
    if prev_code is not None:
        # Alternate from previous code
        return offset_code if prev_code == onset_code else onset_code

    # If no valid previous event, try next valid epoch event
    next_code = _find_next_valid_code(events, corrupt_idx, onset_code, offset_code)
    if next_code is not None:
        # Alternate from next code (reverse logic)
        return offset_code if next_code == onset_code else onset_code

    # If we can't find any pattern context, determine position relative to start
    # Count epoch events before this position to determine if we should be onset or offset
    epoch_count = _count_epoch_events_before(events, corrupt_idx, onset_code, offset_code)

    # Alternating pattern: onset (0), offset (1), onset (2), offset (3), ...
    # Even positions = onset, odd positions = offset
    expected_code = onset_code if epoch_count % 2 == 0 else offset_code

    logger.debug(f"No pattern context found for event at index {corrupt_idx}, "
                f"epoch position {epoch_count} -> {expected_code}")
    return expected_code


def _find_previous_valid_code(events: np.ndarray,
                            corrupt_idx: int,
                            onset_code: int,
                            offset_code: int) -> Union[int, None]:
    """Find the most recent valid (non-corrupted) event code before corrupt_idx."""
    for i in range(corrupt_idx - 1, -1, -1):
        event_code = events[i, 2]
        if event_code in [onset_code, offset_code]:
            return event_code
    return None


def _find_next_valid_code(events: np.ndarray,
                        corrupt_idx: int,
                        onset_code: int,
                        offset_code: int) -> Union[int, None]:
    """Find the next valid (non-corrupted) event code after corrupt_idx."""
    for i in range(corrupt_idx + 1, len(events)):
        event_code = events[i, 2]
        if event_code in [onset_code, offset_code]:
            return event_code
    return None


def _update_raw_annotations(raw: BaseRaw, events: np.ndarray, event_id: dict) -> None:
    """
    Update the raw object's annotations based on corrected events.

    Args:
        raw: Raw object to update
        events: Corrected events array
        event_id: Event ID mapping from events_from_annotations
    """
    from mne import annotations_from_events

    # Create reverse mapping for annotations_from_events
    event_desc = {v: k for k, v in event_id.items()}

    # Convert events back to annotations
    annotations = annotations_from_events(
        events=events,
        event_desc=event_desc,
        sfreq=raw.info['sfreq'],
        orig_time=raw.info['meas_date']
    )

    # Replace annotations in raw object
    raw.set_annotations(annotations)


def _count_epoch_events_before(events: np.ndarray,
                              target_idx: int,
                              onset_code: int,
                              offset_code: int) -> int:
    """
    Count how many epoch events (onset or offset) appear before target_idx.

    This helps determine the expected position in the alternating sequence
    when we can't find neighboring context.

    Args:
        events: Events array
        target_idx: Index we're trying to correct
        onset_code: Onset event code
        offset_code: Offset event code

    Returns:
        Number of epoch events before target_idx
    """
    count = 0
    for i in range(target_idx):
        event_code = events[i, 2]
        if event_code in [onset_code, offset_code]:
            count += 1
    return count