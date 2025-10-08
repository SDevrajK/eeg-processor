"""
Data segmentation functionality for EEG processing pipeline.

This module handles segmentation of raw EEG data based on condition markers,
providing utilities for both single and multi-condition data processing.
"""

from typing import Dict, List, Optional, Union
from loguru import logger
from mne.io import BaseRaw
from mne import concatenate_raws
import numpy as np

from ..utils.event_parsers import find_matching_events


# Default segmentation padding in seconds
DEFAULT_SEGMENTATION_PADDING = 5.0


def segment_raw_by_conditions(raw: BaseRaw, conditions: List[dict]) -> Dict[str, BaseRaw]:
    """
    Pre-segment raw data by condition markers based on markers field.
    
    Segmentation is performed when conditions have markers defined,
    regardless of the number of conditions.
    
    Args:
        raw: The loaded and preloaded raw data
        conditions: List of condition configurations
        
    Returns:
        Dict mapping condition name to raw data (segmented or original)
    """
    # Check if any conditions have markers that require segmentation
    conditions_with_markers = [c for c in conditions if 'markers' in c and c['markers']]
    
    if not conditions_with_markers:
        logger.info("No conditions have markers - skipping segmentation")
        return {condition['name']: raw for condition in conditions}
    
    logger.info(f"Segmenting data for {len(conditions_with_markers)} conditions with markers")
    segmented_data = {}
    
    for condition in conditions:
        condition_name = condition['name']
        segmented_data[condition_name] = segment_single_condition(raw, condition)
        
    return segmented_data


def segment_single_condition(raw: BaseRaw, condition: dict) -> BaseRaw:
    """
    Segment a single condition or return original raw data.
    
    Args:
        raw: Raw EEG data
        condition: Condition configuration dictionary
        
    Returns:
        Segmented raw data or original raw data if no segmentation needed
    """
    condition_name = condition['name']
    
    if 'markers' in condition:
        return perform_segmentation(raw, condition)
    else:
        logger.warning(f"Condition {condition_name} has no markers - using full data")
        return raw


def perform_segmentation(raw: BaseRaw, condition: dict) -> BaseRaw:
    """
    Perform actual segmentation for a condition with markers.

    Args:
        raw: Raw EEG data
        condition: Condition configuration with markers

    Returns:
        Segmented raw data

    Raises:
        RuntimeError: If segmentation fails
    """
    condition_name = condition['name']
    markers = condition['markers']
    padding = condition.get('padding', DEFAULT_SEGMENTATION_PADDING)
    t_min = condition.get('t_min', None)
    t_max = condition.get('t_max', None)

    time_window_str = ""
    if t_min is not None or t_max is not None:
        time_window_str = f", time window: [{t_min or 'start'}s - {t_max or 'end'}s]"

    logger.info(f"Segmenting {condition_name} using 'markers' field")
    logger.debug(f"Segmenting with markers: {markers}, padding: {padding}s{time_window_str}")

    try:
        segmented = segment_by_condition_markers(
            raw,
            condition={'name': condition_name, 'markers': markers, 't_min': t_min, 't_max': t_max},
            padding=padding
        )
        # Calculate actual duration of segmented data
        total_duration = segmented.times[-1] - segmented.times[0]
        n_samples = segmented.n_times
        actual_duration = n_samples / segmented.info['sfreq']
        logger.info(f"Segmented {condition_name}: {actual_duration:.1f}s of actual data ({n_samples} samples @ {segmented.info['sfreq']}Hz)")
        logger.debug(f"Time range in concatenated object: {segmented.times[0]:.1f}s to {segmented.times[-1]:.1f}s")
        return segmented
    except Exception as e:
        logger.error(f"Failed to segment condition {condition_name}: {e}")
        raise RuntimeError(f"Segmentation failed for {condition_name}: {e}")


# Helper functions moved from raw_data_tools.py

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


def _get_event_times(raw: BaseRaw, event_code: Union[str, int],
                     t_min: Optional[float] = None, t_max: Optional[float] = None) -> np.ndarray:
    """Get event times from raw data for a specific event code, optionally filtered by time window"""
    try:
        event_samples = find_matching_events(raw, event_code)
        # Adjust event_samples for cropped object
        event_samples -= raw.first_samp
        event_times = event_samples / raw.info['sfreq']

        # Apply time window filtering if specified
        if t_min is not None or t_max is not None:
            valid_mask = np.ones(len(event_times), dtype=bool)

            if t_min is not None:
                valid_mask &= (event_times >= t_min)

            if t_max is not None:
                valid_mask &= (event_times <= t_max)

            event_times = event_times[valid_mask]
            logger.debug(f"Filtered {np.sum(~valid_mask)} events outside time window [{t_min}-{t_max}], kept {len(event_times)}")

        return event_times
    except ValueError as e:
        # Augment error with crop-specific context
        raise ValueError(f"Event detection failed: {str(e)}") from e


def _pair_last_start_before_end(start_times: List[float], end_times: List[float]) -> List[tuple]:
    """
    Pair each start event with the next immediate end event that follows it.
    Each start and end can only be used once.

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
    used_ends = set()

    # For each start, find the closest end that comes after it
    for i, start_time in enumerate(start_times):
        if i in used_starts:
            continue
            
        # Find the first end time after this start that hasn't been used
        valid_end_indices = np.where((end_times > start_time))[0]
        
        best_end_idx = None
        for idx in valid_end_indices:  # Search forward for next available end
            if idx not in used_ends:
                best_end_idx = idx
                break

        if best_end_idx is not None:
            end_time = end_times[best_end_idx]
            pairs.append((start_time, end_time))
            used_starts.add(i)
            used_ends.add(best_end_idx)
            logger.debug(f"Paired start {start_time:.1f}s with end {end_time:.1f}s")
        else:
            logger.warning(f"No matching end found for start at {start_time:.1f}s")

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
        condition: Dictionary containing markers [start, end] or {block: [start, end]}
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
    markers = condition.get('markers')
    t_min = condition.get('t_min', None)
    t_max = condition.get('t_max', None)

    if not markers:
        raise ValueError("Condition must have 'markers' defined")

    # Always work on a copy first since we need to analyze events
    working_data = raw.copy()

    # Handle different marker formats
    if isinstance(markers, dict):
        # New dict format: {block_id: [start_marker, end_marker], ...}
        # Use ALL block sets to collect events
        if not markers:
            raise ValueError("Markers dictionary cannot be empty")
        
        logger.info(f"Processing {len(markers)} block sets: {list(markers.keys())}")
        
        # Process blocks sequentially - each block gets one segment
        event_pairs = []
        used_event_indices = {'start': set(), 'end': set()}
        
        # First, collect all events with their indices
        all_start_events = []
        all_end_events = []
        
        # Get unique marker codes
        start_markers = set()
        end_markers = set()
        for marker_list in markers.values():
            start_markers.add(marker_list[0])
            end_markers.add(marker_list[1])
        
        # Collect all start events
        for marker in start_markers:
            try:
                times = _get_event_times(working_data, marker, t_min, t_max)
                for i, t in enumerate(times):
                    all_start_events.append({'time': t, 'marker': marker, 'index': len(all_start_events)})
            except ValueError:
                pass

        # Collect all end events
        for marker in end_markers:
            try:
                times = _get_event_times(working_data, marker, t_min, t_max)
                for i, t in enumerate(times):
                    all_end_events.append({'time': t, 'marker': marker, 'index': len(all_end_events)})
            except ValueError:
                pass
        
        # Sort by time
        all_start_events.sort(key=lambda x: x['time'])
        all_end_events.sort(key=lambda x: x['time'])
        
        logger.debug(f"Found {len(all_start_events)} total start events, {len(all_end_events)} total end events")
        
        # Process each block in order
        for block_id, marker_list in markers.items():
            if not isinstance(marker_list, list) or len(marker_list) != 2:
                raise ValueError(f"Each marker list must have exactly 2 markers [start, end], got {marker_list} for block '{block_id}'")
            
            start_marker, end_marker = marker_list
            
            # Find next available start event with the right marker
            start_event = None
            for event in all_start_events:
                if (event['marker'] == start_marker and 
                    event['index'] not in used_event_indices['start']):
                    start_event = event
                    break
            
            if not start_event:
                logger.warning(f"Block '{block_id}': no available start marker {start_marker}")
                continue
                
            # Find the next end event after this start with the right marker
            end_event = None
            for event in all_end_events:
                if (event['time'] > start_event['time'] and 
                    event['marker'] == end_marker and
                    event['index'] not in used_event_indices['end']):
                    end_event = event
                    break
                    
            if not end_event:
                logger.warning(f"Block '{block_id}': no available end marker {end_marker} after start at {start_event['time']:.1f}s")
                continue
                
            # Create the pair
            event_pairs.append((start_event['time'], end_event['time']))
            used_event_indices['start'].add(start_event['index'])
            used_event_indices['end'].add(end_event['index'])
            
            logger.info(f"Block '{block_id}': paired start {start_event['time']:.1f}s (marker {start_marker}) with end {end_event['time']:.1f}s (marker {end_marker})")
        
        # Critical validation - no valid pairs found
        if not event_pairs:
            error_msg = f"No valid event pairs found across any block sets {list(markers.keys())}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        logger.info(f"Combined all block sets: {len(event_pairs)} total segments")
    
    elif isinstance(markers, list):
        # Handle legacy list format: [start_marker, end_marker]
        if len(markers) != 2:
            raise ValueError("Condition must have exactly 2 markers [start, end]")
        
        # Critical failure - event finding errors should bubble up
        try:
            start_times = _get_event_times(working_data, markers[0], t_min, t_max)
            end_times = _get_event_times(working_data, markers[1], t_min, t_max)
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
    logger.info(f"Processing {len(event_pairs)} event pairs with padding={padding}s")
    for i, (start_time, end_time) in enumerate(event_pairs):
        try:
            # Log raw event times first
            logger.debug(f"Event pair {i+1}: start={start_time:.1f}s, end={end_time:.1f}s (raw duration: {end_time - start_time:.1f}s)")
            
            tmin = max(start_time - padding, working_data.times[0])
            tmax = min(end_time + padding, working_data.times[-1])

            if tmax <= tmin:
                logger.warning(f"Segment {i} has invalid timing: {tmin:.2f}s to {tmax:.2f}s - skipping")
                continue

            segment = working_data.copy().crop(tmin=tmin, tmax=tmax)
            segment_duration = tmax - tmin
            segments.append(segment)
            logger.info(f"Segment {i+1}: {tmin:.1f}s to {tmax:.1f}s (duration: {segment_duration:.1f}s, with padding)")

        except Exception as e:
            error_msg = f"Failed to process segment {i} ({start_time:.2f}s-{end_time:.2f}s): {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    # Critical failure - no valid segments at all
    if not segments:
        error_msg = f"No valid segments created from {len(event_pairs)} potential segments"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Create final segmented data
    try:
        if len(segments) > 1:
            # Log timing info before concatenation
            logger.debug("Segment timing before concatenation:")
            for i, seg in enumerate(segments):
                logger.debug(f"  Segment {i+1}: first_samp={seg.first_samp}, times[0]={seg.times[0]:.1f}s, times[-1]={seg.times[-1]:.1f}s, duration={seg.times[-1]-seg.times[0]:.1f}s")
            
            segmented_raw = concatenate_raws(segments)
            
            # Log timing after concatenation
            logger.debug(f"After concatenation: first_samp={segmented_raw.first_samp}, times[0]={segmented_raw.times[0]:.1f}s, times[-1]={segmented_raw.times[-1]:.1f}s")
        else:
            segmented_raw = segments[0]
    except Exception as e:
        # Concatenation failure is critical
        logger.error(f"Failed to concatenate segments: {e}")
        raise RuntimeError(f"Segment concatenation failed: {e}") from e

    # Calculate total duration of all segments
    total_segment_duration = sum(seg.times[-1] - seg.times[0] for seg in segments)
    
    if inplace:
        logger.info(f"Segmented data applied in-place: {len(segments)} segments, total duration: {total_segment_duration:.1f}s")
        return _replace_raw_content(raw, segmented_raw)
    else:
        logger.info(f"Segmentation completed: {len(segments)} segments, total duration: {total_segment_duration:.1f}s")
        return segmented_raw