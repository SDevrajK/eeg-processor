# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import yaml

import numpy as np
from mne.io import BaseRaw
from .event_parsers import find_matching_events


def _replace_raw_content(original_raw: BaseRaw, new_raw: BaseRaw) -> BaseRaw:
    """
    Replace the content of original_raw with new_raw while preserving object identity

    Used for structural operations like crop and segment_condition where
    in-place operation means replacing the entire object content.
    """
    # Core data and info
    original_raw.__dict__.clear()
    original_raw.__dict__.update(new_raw.__dict__.copy())

    return original_raw

def _get_event_times(raw, event_code):
    """Now just a thin wrapper"""
    try:
        event_samples = find_matching_events(raw, event_code)
        return event_samples / raw.info['sfreq']
    except ValueError as e:
        # Augment error with crop-specific context
        raise ValueError(f"Crop failed: {str(e)}") from e


def crop_data(
        raw: BaseRaw,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        inplace: bool = False,
        crop_before: Optional[Union[str, int]] = None,
        crop_after: Optional[Union[str, int]] = None,
        show: bool = None
) -> BaseRaw:
    """
    Crop raw EEG data using either absolute times or event markers.

    Parameters
    ----------
    crop_before : str | int | None
        Event code (e.g., 51, 'Stimulus/S 51', 'Stimulus/S  1')
    crop_after : str | int | None
        Event code to crop after last occurrence
    """
    current_raw = raw.copy()

    current_start, current_end = current_raw.times[0], current_raw.times[-1]

    # --- Absolute time cropping (priority) ---
    if t_min is not None or t_max is not None:
        start = 0.0 if t_min is None else t_min
        end = current_end if t_max is None else t_max

    # --- Event-based cropping ---
    elif crop_before is not None or crop_after is not None:
        start, end = current_start, current_end

        if crop_before is not None:
            times = _get_event_times(current_raw, crop_before)
            if len(times) > 0:
                start = times[0] - 10

        if crop_after is not None:
            times = _get_event_times(current_raw, crop_after)
            if len(times) > 0:
                end = times[-1] + 10

        if start >= end:
            raise ValueError(
                f"Invalid crop: [{start:.2f}s, {end:.2f}s]\n"
                f"crop_before='{crop_before}' → {start:.2f}s\n"
                f"crop_after='{crop_after}' → {end:.2f}s"
            )

    else:
        raise ValueError("Must provide either (t_min/t_max) or (crop_before/crop_after)")

    # Clamp to recording duration
    clamped_start = max(start, current_start)
    clamped_end = min(end, current_end)
    current_raw.crop(tmin=clamped_start, tmax=clamped_end, include_tmax=True)

    if inplace:
        # Replace original object content with cropped data
        return _replace_raw_content(raw, current_raw)
    else:
        return current_raw


def adjust_event_times(raw: BaseRaw,
                       shift_ms: float,
                       target_events: Optional[List[str]] = None,
                       protect_events: Optional[List[str]] = None,
                       inplace: bool = False,
                       **kwargs) -> BaseRaw:
    """
    Adjust event times with optional in-place operation

    Args:
        raw: Raw EEG data with annotations
        shift_ms: Time shift in milliseconds
        target_events: Specific events to shift (None = all events)
        protect_events: Events to exclude from shifting
        inplace: Whether to modify input object or create copy
        **kwargs: Additional parameters

    Returns:
        Raw object with adjusted event times
    """
    if not raw.annotations:
        return raw if inplace else raw.copy()

    if inplace:
        result = raw
    else:
        result = raw.copy()

    # Convert shift to seconds
    shift_sec = shift_ms / 1000
    annotations = result.annotations

    # Create modification mask
    modify_mask = np.ones(len(annotations), dtype=bool)
    for i, desc in enumerate(annotations.description):
        if protect_events and desc in protect_events:
            modify_mask[i] = False
        elif target_events and desc not in target_events:
            modify_mask[i] = False

    # Apply changes
    new_onset = annotations.onset.copy()
    new_onset[modify_mask] += shift_sec

    # Create new annotations
    new_annot = annotations.__class__(
        onset=new_onset,
        duration=annotations.duration,
        description=annotations.description,
        orig_time=annotations.orig_time
    )

    result.set_annotations(new_annot)
    return result


def segment_by_condition_markers(raw: BaseRaw,
                                 condition: dict,
                                 padding: float = 5.0,
                                 inplace: bool = False,
                                 **kwargs) -> BaseRaw:
    """
    Segment Raw data based on condition markers with optional in-place operation

    Args:
        raw: Raw EEG data
        condition: Dictionary containing condition_markers [start, end]
        padding: Padding around segments in seconds
        inplace: Whether to modify input object or create copy
        **kwargs: Additional parameters

    Returns:
        Raw object containing only the segmented data

    Note: Due to the nature of segmentation (cropping and concatenating),
          in-place operation replaces the original object's content entirely.
    """
    markers = condition.get('condition_markers')
    if not markers or len(markers) != 2:
        raise ValueError("Condition must have exactly 2 markers [start, end]")

    from mne import concatenate_raws

    # Always work on a copy first since we need to analyze events
    working_data = raw.copy()
    sfreq = working_data.info['sfreq']

    # Use the same event finding logic as crop_data
    try:
        start_times = _get_event_times(working_data, markers[0])
        end_times = _get_event_times(working_data, markers[1])
    except ValueError as e:
        logger.error(f"Segmentation failed: {str(e)}")
        return raw if inplace else raw.copy()

    if len(start_times) == 0 or len(end_times) == 0:
        logger.warning(f"No segments found for markers {markers}")
        return raw if inplace else raw.copy()

    # Validate and process segments
    segments = []
    for start_time, end_time in zip(start_times, end_times):
        tmin = max(start_time - padding, working_data.times[0])
        tmax = min(end_time + padding, working_data.times[-1])

        if tmax > tmin:  # Valid segment
            segment = working_data.copy().crop(tmin=tmin, tmax=tmax)
            segments.append(segment)

    if not segments:
        logger.warning("No valid segments found")
        return raw if inplace else raw.copy()

    # Create segmented data
    if len(segments) > 1:
        segmented_raw = concatenate_raws(segments)
    else:
        segmented_raw = segments[0]

    if inplace:
        # Replace original object's content with segmented data
        logger.info(f"Segmented data applied in-place: {len(segments)} segments")
        return _replace_raw_content(raw, segmented_raw)  # Fixed: was using working_data
    else:
        logger.info(f"Segmentation completed: {len(segments)} segments")
        return segmented_raw