"""
Bad channel detection and interpolation for EEG data.

This module provides robust bad channel detection using statistical methods,
with optional segment-wise processing and automatic interpolation.
"""

from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
import numpy as np
from mne.io import BaseRaw
from mne import concatenate_raws
import matplotlib.pyplot as plt


def detect_bad_channels(
        raw: BaseRaw,
        flat_threshold: float = 1e-15,
        noisy_threshold: float = 3.0,
        segment_wise: bool = False,
        segment_length: float = 10.0,
        interpolate: bool = True,
        inplace: bool = False,
        verbose: bool = False,
        show_plot: bool = False,
        plot_duration: float = 2.0,
        plot_start: float = 5.0
) -> BaseRaw:
    """
    Detect and optionally interpolate bad channels using statistical methods.

    Uses variance-based detection to identify flat and noisy channels, with
    robust statistics (MAD) for outlier detection.

    Args:
        raw: MNE Raw object
        flat_threshold: Threshold for flat channels (variance in V²)
        noisy_threshold: Threshold for noisy channels (MAD multiplier)
        segment_wise: Process data in segments (recommended for long recordings)
        segment_length: Segment duration in seconds
        interpolate: Whether to interpolate detected bad channels
        inplace: Modify input object (if False, returns copy)
        verbose: Enable detailed logging
        show_plot: Show before/after comparison plot
        plot_duration: Duration of plot window in seconds
        plot_start: Start time for plot in seconds

    Returns:
        Raw object with bad channels detected and optionally interpolated

    Notes:
        - Stores detailed metrics in raw._bad_channel_metrics for quality tracking
        - Excludes EOG and stimulus channels from detection
        - Uses Median Absolute Deviation (MAD) for robust noise detection
    """
    if not inplace:
        raw = raw.copy()

    original_bads = set(raw.info['bads'])

    logger.info(f"Starting bad channel detection (interpolate={interpolate}, segment_wise={segment_wise})")
    if original_bads:
        logger.debug(f"Pre-existing bad channels: {sorted(original_bads)}")

    if segment_wise:
        detected_bads = _detect_bad_channels_segmented(
            raw, flat_threshold, noisy_threshold, segment_length, interpolate, verbose
        )
    else:
        detected_bads = _detect_bad_channels_global(
            raw, flat_threshold, noisy_threshold, interpolate, verbose
        )

    # Store comprehensive metrics for quality tracking
    final_bads = set(raw.info['bads'])
    raw._bad_channel_metrics = {
        'original_bads': sorted(original_bads),
        'detected_bads': sorted(detected_bads),
        'final_bads': sorted(final_bads),
        'n_original': len(original_bads),
        'n_detected': len(detected_bads),
        'n_final': len(final_bads),
        'interpolation_attempted': interpolate,
        'interpolation_successful': interpolate and len(final_bads) < len(original_bads | detected_bads),
        'method': 'segment_wise' if segment_wise else 'global',
        'parameters': {
            'flat_threshold': flat_threshold,
            'noisy_threshold': noisy_threshold,
            'segment_length': segment_length if segment_wise else None
        }
    }

    # Summary logging
    if final_bads:
        logger.info(f"Final bad channel count: {len(final_bads)}")

    if show_plot and (detected_bads or original_bads):
        _plot_bad_channels_comparison(raw, original_bads, detected_bads,
                                      plot_start, plot_duration, interpolate)

    return raw


def _detect_bad_channels_global(
        raw: BaseRaw,
        flat_threshold: float,
        noisy_threshold: float,
        interpolate: bool,
        verbose: bool
) -> set:
    """Global bad channel detection across entire recording."""
    detected_bads = _find_bad_channels_in_segment(raw, flat_threshold, noisy_threshold)

    if detected_bads:
        # Combine with existing bad channels
        all_bads = list(set(raw.info['bads']) | set(detected_bads))
        raw.info['bads'] = all_bads

        logger.info(f"Detected bad channels: {sorted(detected_bads)}")

        if interpolate:
            _interpolate_bad_channels(raw, verbose)

    return set(detected_bads)


def _detect_bad_channels_segmented(
        raw: BaseRaw,
        flat_threshold: float,
        noisy_threshold: float,
        segment_length: float,
        interpolate: bool,
        verbose: bool
) -> set:
    """Segment-wise bad channel detection for long recordings."""
    n_segments = int(np.ceil(raw.times[-1] / segment_length))
    all_detected_bads = set()

    logger.info(f"Processing {n_segments} segments of {segment_length}s each")

    segments = []
    for i in range(n_segments):
        tmin = i * segment_length
        tmax = min((i + 1) * segment_length, raw.times[-1])
        segment = raw.copy().crop(tmin=tmin, tmax=tmax)

        # Detect bad channels in this segment
        segment_bads = _find_bad_channels_in_segment(segment, flat_threshold, noisy_threshold)

        if segment_bads:
            logger.debug(f"Segment {i + 1}/{n_segments} bad channels: {sorted(segment_bads)}")
            all_detected_bads.update(segment_bads)

            # Apply to segment
            segment.info['bads'] = list(set(segment.info['bads']) | set(segment_bads))

            if interpolate:
                _interpolate_bad_channels(segment, verbose)

        segments.append(segment)

    # Reconstruct continuous data from segments
    if len(segments) > 1:
        raw_reconstructed = concatenate_raws(segments)
        raw.crop(tmin=0, tmax=0)  # Clear original data
        raw = raw_reconstructed

    # Update global bad channels list
    all_bads = list(set(raw.info['bads']) | all_detected_bads)
    raw.info['bads'] = all_bads

    logger.success(f"Segment-wise processing complete. Total bad channels: {len(all_bads)}")

    return all_detected_bads


def _find_bad_channels_in_segment(
        raw_segment: BaseRaw,
        flat_threshold: float,
        noisy_threshold: float
) -> List[str]:
    """
    Core detection algorithm using statistical methods.

    Uses variance for flat channel detection and MAD for noise detection.
    """
    data = raw_segment.get_data()
    variances = np.var(data, axis=1)

    # Exclude non-EEG channels from detection
    exclude_patterns = ['EOG', 'HEOG', 'VEOG', 'STIM', 'TRIGGER', 'ECG']
    exclude_channels = [
        ch for ch in raw_segment.ch_names
        if any(pattern in ch.upper() for pattern in exclude_patterns)
    ]

    # Flat channel detection
    flat_channels = [
        raw_segment.ch_names[i] for i, var in enumerate(variances)
        if var < flat_threshold and raw_segment.ch_names[i] not in exclude_channels
    ]

    # Noisy channel detection using MAD (Median Absolute Deviation)
    eeg_variances = [
        var for i, var in enumerate(variances)
        if raw_segment.ch_names[i] not in exclude_channels
    ]

    if len(eeg_variances) > 0:
        median_var = np.median(eeg_variances)
        mad = 1.4826 * np.median(np.abs(eeg_variances - median_var))

        noisy_channels = [
            raw_segment.ch_names[i] for i, var in enumerate(variances)
            if (abs(var - median_var) > noisy_threshold * mad and
                raw_segment.ch_names[i] not in exclude_channels)
        ]
    else:
        noisy_channels = []

    # Combine detections, excluding already marked bad channels
    detected_bads = list(set(flat_channels + noisy_channels) - set(raw_segment.info['bads']))

    if flat_channels:
        logger.debug(f"Flat channels (var < {flat_threshold:.1e}): {sorted(flat_channels)}")
    if noisy_channels:
        logger.debug(f"Noisy channels ({noisy_threshold}×MAD): {sorted(noisy_channels)}")

    return detected_bads


def _interpolate_bad_channels(raw: BaseRaw, verbose: bool) -> None:
    """
    Interpolate bad channels with error handling and success tracking.
    """
    bads_before = raw.info['bads'].copy()

    if not bads_before:
        return

    try:
        raw.interpolate_bads(reset_bads=True, verbose=verbose)

        bads_after = raw.info['bads']
        successfully_interpolated = [ch for ch in bads_before if ch not in bads_after]

        if successfully_interpolated:
            logger.success(f"Successfully interpolated: {sorted(successfully_interpolated)}")
        if bads_after:
            logger.warning(f"Interpolation failed for: {sorted(bads_after)} - keeping as bad")

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raw.info['bads'] = bads_before  # Restore original bad channels list


def _plot_bad_channels_comparison(
        raw: BaseRaw,
        original_bads: set,
        detected_bads: set,
        plot_start: float,
        plot_duration: float,
        interpolate: bool
) -> None:
    """
    Create before/after comparison plot for bad channels.
    """
    start_idx = int(plot_start * raw.info['sfreq'])
    end_idx = start_idx + int(plot_duration * raw.info['sfreq'])
    time = raw.times[start_idx:end_idx]

    n_plots = (1 if original_bads else 0) + (1 if detected_bads else 0)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    plot_idx = 0

    # Plot original bad channels
    if original_bads:
        ax = axes[plot_idx]
        for ch in sorted(original_bads):
            if ch in raw.ch_names:
                ch_data = raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0] * 1e6
                ax.plot(time, ch_data, label=ch, alpha=0.8)

        ax.set_title(f'Pre-existing bad channels: {sorted(original_bads)}')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot newly detected bad channels
    if detected_bads:
        ax = axes[plot_idx]
        for ch in sorted(detected_bads):
            if ch in raw.ch_names:
                ch_data = raw.get_data(picks=[ch], start=start_idx, stop=end_idx)[0] * 1e6
                ax.plot(time, ch_data, label=f'{ch} (detected)', alpha=0.8)

        ax.set_title(f'Newly detected bad channels: {sorted(detected_bads)}')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def get_channel_quality_summary(raw: BaseRaw) -> Dict:
    """
    Extract channel quality summary from processed raw data.

    Args:
        raw: Processed raw data with _bad_channel_metrics attribute

    Returns:
        Dictionary with channel quality metrics
    """
    if hasattr(raw, '_bad_channel_metrics'):
        return raw._bad_channel_metrics.copy()
    else:
        # Fallback for raw data without detailed metrics
        return {
            'final_bads': raw.info['bads'],
            'n_final': len(raw.info['bads']),
            'interpolation_attempted': False,
            'method': 'unknown'
        }