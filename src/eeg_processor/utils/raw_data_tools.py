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

        #adjust event_samples for cropped object
        event_samples -= raw.first_samp

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
        segment_start: Optional[Union[str, int]] = None,
        segment_end: Optional[Union[str, int]] = None,
        show: bool = None,
        padded: bool = True,
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
    elif crop_before is not None or crop_after is not None or segment_start is not None or segment_end is not None:
        start, end = current_start, current_end

        if crop_before is not None:
            times = _get_event_times(current_raw, crop_before)
            if len(times) > 0:
                start = times[0] - 10 if padded else times[0]

        if crop_after is not None:
            times = _get_event_times(current_raw, crop_after)
            if len(times) > 0:
                end = times[-1] + 10 if padded else times[-1]

        if segment_start is not None:
            times = _get_event_times(current_raw, segment_start)
            if len(times) > 0:
                start = times[0] - 10 if padded else times[0]

        if segment_end is not None:
            times = _get_event_times(current_raw, segment_end)
            if len(times) > 0:
                end = times[-1] + 10 if padded else times[-1]

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


def _pair_last_start_before_end(start_times, end_times):
    """
    For each end event, find the most recent start event that precedes it.
    Handles restarts by using the last start before each end.

    Args:
        start_times: List of start event times
        end_times: List of end event times

    Returns:
        List of (start_time, end_time) tuples
    """
    start_times = np.array(sorted(start_times))
    end_times = np.array(sorted(end_times))

    pairs = []
    used_starts = set()

    for end_time in end_times:
        # Find all starts that come before this end
        valid_start_indices = np.where(start_times < end_time)[0]

        # Of those, find the most recent one that hasn't been used
        best_start_idx = None
        for idx in reversed(valid_start_indices):  # Search backwards (most recent first)
            if idx not in used_starts:
                best_start_idx = idx
                break

        if best_start_idx is not None:
            start_time = start_times[best_start_idx]
            pairs.append((start_time, end_time))
            used_starts.add(best_start_idx)

    return pairs

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

    Raises:
        ValueError: If condition_markers are invalid or missing
        RuntimeError: If segmentation fails due to data issues

    Note: Due to the nature of segmentation (cropping and concatenating),
          in-place operation replaces the original object's content entirely.
    """
    # Critical validation - let these bubble up
    # Check for both legacy 'condition_markers' and new 'markers' format
    markers = condition.get('condition_markers') or condition.get('markers')
    
    if not markers:
        raise ValueError("Condition must have 'markers' or 'condition_markers' defined")
    
    from mne import concatenate_raws

    # Always work on a copy first since we need to analyze events
    working_data = raw.copy()
    sfreq = working_data.info['sfreq']

    # Handle different marker formats
    if isinstance(markers, dict):
        # New dict format: {trigger_id: [start_marker, end_marker], ...}
        # Use ALL trigger sets to collect events
        if not markers:
            raise ValueError("Markers dictionary cannot be empty")
        
        logger.info(f"Processing {len(markers)} trigger sets: {list(markers.keys())}")
        
        all_start_times = []
        all_end_times = []
        trigger_info = []  # Track which trigger each event pair belongs to
        
        for trigger_id, marker_list in markers.items():
            if not isinstance(marker_list, list) or len(marker_list) != 2:
                raise ValueError(f"Each marker list must have exactly 2 markers [start, end], got {marker_list} for trigger '{trigger_id}'")
            
            try:
                start_times = _get_event_times(working_data, marker_list[0])
                end_times = _get_event_times(working_data, marker_list[1])
                
                logger.info(f"Trigger '{trigger_id}' markers {marker_list}: {len(start_times)} starts, {len(end_times)} ends")
                
                # Handle mismatched events for this trigger set
                if len(start_times) != len(end_times):
                    logger.warning(f"Trigger '{trigger_id}': Mismatched events ({len(start_times)} starts, {len(end_times)} ends). Using pairing strategy.")
                    trigger_pairs = _pair_last_start_before_end(start_times, end_times)
                    
                    if trigger_pairs:
                        logger.info(f"Trigger '{trigger_id}': Paired {len(trigger_pairs)} segments")
                        for start_time, end_time in trigger_pairs:
                            all_start_times.append(start_time)
                            all_end_times.append(end_time)
                            trigger_info.append(trigger_id)
                    else:
                        logger.warning(f"Trigger '{trigger_id}': No valid pairs found")
                else:
                    # Perfect match for this trigger set
                    logger.info(f"Trigger '{trigger_id}': Perfect match with {len(start_times)} segments")
                    for start_time, end_time in zip(start_times, end_times):
                        all_start_times.append(start_time)
                        all_end_times.append(end_time)
                        trigger_info.append(trigger_id)
                        
            except ValueError as e:
                logger.warning(f"Event detection failed for trigger '{trigger_id}' with markers {marker_list}: {str(e)}")
                continue  # Skip this trigger set but continue with others
        
        # Critical validation - no events found across all trigger sets
        if len(all_start_times) == 0 or len(all_end_times) == 0:
            error_msg = f"No events found across any trigger sets {list(markers.keys())} - data may be corrupt or incorrect condition"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Create event pairs with trigger information
        event_pairs = list(zip(all_start_times, all_end_times))
        logger.info(f"Combined all trigger sets: {len(event_pairs)} total segments")
        
        # Log summary by trigger
        trigger_counts = {}
        for trigger_id in trigger_info:
            trigger_counts[trigger_id] = trigger_counts.get(trigger_id, 0) + 1
        for trigger_id, count in trigger_counts.items():
            logger.info(f"  Trigger '{trigger_id}': {count} segments")
    
    elif isinstance(markers, list):
        # Handle legacy list format: [start_marker, end_marker]
        if len(markers) != 2:
            raise ValueError("Condition must have exactly 2 markers [start, end]")
        
        # Critical failure - event finding errors should bubble up
        try:
            start_times = _get_event_times(working_data, markers[0])
            end_times = _get_event_times(working_data, markers[1])
        except ValueError as e:
            logger.error(f"Event detection failed for markers {markers}: {str(e)}")
            raise RuntimeError(f"Segmentation failed - cannot find events: {str(e)}") from e

        # Critical validation - no events found means data is wrong
        if len(start_times) == 0 or len(end_times) == 0:
            error_msg = f"No events found for markers {markers} - data may be corrupt or incorrect condition"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Handle mismatched events with intelligent pairing
        if len(start_times) != len(end_times):
            logger.warning(f"Mismatched events: {len(start_times)} starts, {len(end_times)} ends. "
                           f"Using last-start-before-end pairing strategy.")

            event_pairs = _pair_last_start_before_end(start_times, end_times)

            if not event_pairs:
                raise RuntimeError("No valid start-end pairs found using pairing strategy")

            logger.info(f"Paired {len(event_pairs)} segments from mismatched events")

        else:
            # Perfect match - use existing logic
            event_pairs = list(zip(start_times, end_times))
            logger.info(f"Perfect event match: {len(event_pairs)} segments")
    
    else:
        raise ValueError("Markers must be a list [start, end] or dict {trigger: [start, end]}")

    # Process segments
    segments = []
    for i, (start_time, end_time) in enumerate(event_pairs):
        try:
            tmin = max(start_time - padding, working_data.times[0])
            tmax = min(end_time + padding, working_data.times[-1])

            if tmax <= tmin:
                logger.warning(f"Segment {i} has invalid timing: {tmin:.2f}s to {tmax:.2f}s - skipping")
                continue

            segment = working_data.copy().crop(tmin=tmin, tmax=tmax)
            segments.append(segment)

        except Exception as e:
            error_msg = f"Failed to process segment {i} ({start_time:.2f}s-{end_time:.2f}s): {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Critical failure - no valid segments at all
    if not segments:
        error_msg = f"No valid segments created from {len(start_times)} potential segments"

        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Create final segmented data
    try:
        if len(segments) > 1:
            segmented_raw = concatenate_raws(segments)
        else:
            segmented_raw = segments[0]
    except Exception as e:
        # Concatenation failure is critical
        logger.error(f"Failed to concatenate segments: {e}")
        raise RuntimeError(f"Segment concatenation failed: {e}") from e

    if inplace:
        logger.info(f"Segmented data applied in-place: {len(segments)} segments")
        return _replace_raw_content(raw, segmented_raw)
    else:
        logger.info(f"Segmentation completed: {len(segments)} segments")
        return segmented_raw