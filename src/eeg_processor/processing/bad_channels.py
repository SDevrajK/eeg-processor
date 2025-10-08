"""
Bad channel detection and interpolation for EEG data.

This module provides robust bad channel detection using MNE's Local Outlier Factor (LOF) method,
with LOF-based interpolation validation.
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
        n_neighbors: int = 10,
        threshold: float = 1.5,
        interpolate: bool = True,
        inplace: bool = False,
        verbose: bool = False,
        show_plot: bool = False,
        plot_duration: float = 2.0,
        plot_start: float = 5.0
) -> BaseRaw:
    """
    Detect and optionally interpolate bad channels using MNE's LOF method.

    Uses Local Outlier Factor to identify channels that are outliers compared
    to their spatial neighbors, avoiding false positives from physiological
    artifacts like eyeblinks.

    Args:
        raw: MNE Raw object
        n_neighbors: Number of neighboring channels to consider for LOF (8-12 good for 32-ch systems)
        threshold: LOF threshold for outlier detection (higher = more conservative)
        interpolate: Whether to interpolate detected bad channels
        inplace: Modify input object (if False, returns copy)
        verbose: Enable detailed logging
        show_plot: Show before/after comparison plot
        plot_duration: Duration of plot window in seconds
        plot_start: Start time for plot in seconds

    Returns:
        Raw object with bad channels detected and optionally interpolated

    Notes:
        - LOF method is robust against physiological artifacts like eyeblinks
        - Automatically excludes non-EEG channels from detection
        - Stores detailed metrics in raw._bad_channel_metrics for quality tracking
        - Uses LOF re-detection to validate interpolation success
    """
    if not inplace:
        raw = raw.copy()

    original_bads = set(raw.info['bads'])

    logger.info(f"Starting LOF bad channel detection (interpolate={interpolate})")
    if original_bads:
        logger.debug(f"Pre-existing bad channels: {sorted(original_bads)}")

    # Use MNE's LOF method for bad channel detection
    try:
        from mne.preprocessing import find_bad_channels_lof
        detected_bads = find_bad_channels_lof(
            raw,
            n_neighbors=n_neighbors,
            threshold=threshold,
            verbose=verbose,
            picks='eeg'
        )

        if detected_bads:
            # Combine with existing bad channels
            all_bads = list(set(raw.info['bads']) | set(detected_bads))
            raw.info['bads'] = all_bads

            logger.info(f"LOF detected bad channels: {sorted(detected_bads)}")

            if interpolate:
                _interpolate_bad_channels(raw, n_neighbors, threshold, verbose)
        else:
            logger.info("No bad channels detected by LOF method")

    except Exception as e:
        logger.error(f"LOF detection failed: {e}")
        detected_bads = []

    # Store comprehensive metrics for quality tracking
    final_bads = set(raw.info['bads'])
    interpolation_details = getattr(raw, '_interpolation_details', {})

    raw._bad_channel_metrics = {
        'original_bads': sorted(original_bads),
        'detected_bads': sorted(detected_bads),
        'final_bads': sorted(final_bads),
        'n_original': len(original_bads),
        'n_detected': len(detected_bads),
        'n_final': len(final_bads),
        'interpolation_attempted': interpolate,
        'interpolation_successful': interpolate and (len(detected_bads) > 0) and (final_bads != set(detected_bads)),
        'interpolation_details': interpolation_details,
        'method': 'lof_global',
        'parameters': {
            'n_neighbors': n_neighbors,
            'threshold': threshold
        }
    }

    # Summary logging
    if final_bads:
        logger.info(f"Final bad channel count: {len(final_bads)}")

    if show_plot and (detected_bads or original_bads):
        _plot_bad_channels_comparison(raw, original_bads, set(detected_bads),
                                      plot_start, plot_duration, interpolate)

    return raw


def _interpolate_bad_channels(raw: BaseRaw, n_neighbors: int, threshold: float, verbose: bool) -> None:
    """
    Interpolate bad channels with LOF-based validation.
    """
    bads_before = raw.info['bads'].copy()

    if not bads_before:
        return

    n_bads = len(bads_before)
    n_total = len([ch for ch in raw.ch_names if not any(pattern in ch.upper() for pattern in ['EOG', 'HEOG', 'VEOG', 'STIM', 'TRIGGER', 'ECG'])])
    bad_percentage = (n_bads / n_total) * 100

    logger.info(f"Attempting interpolation of {n_bads} bad channels ({bad_percentage:.1f}% of {n_total} EEG channels)")

    # Warn if too many bad channels for reliable interpolation
    if bad_percentage > 30:
        logger.warning(f"High percentage of bad channels ({bad_percentage:.1f}%) - interpolation may be unreliable")

    try:
        # Perform interpolation with reset_bads=False to keep track of what was interpolated
        raw.interpolate_bads(reset_bads=False, verbose=verbose)

        # Validate interpolation success using LOF re-detection
        validation_results = _validate_interpolation_with_lof(raw, bads_before, n_neighbors, threshold, verbose)

        # Update bad channels list based on validation
        raw.info['bads'] = validation_results['still_noisy']

        # Store detailed interpolation metrics
        raw._interpolation_details = {
            'attempted': sorted(bads_before),
            'successfully_interpolated': sorted(validation_results['successfully_interpolated']),
            'still_noisy': sorted(validation_results['still_noisy']),
            'success_rate': validation_results['success_rate'],
            'bad_percentage_before': bad_percentage,
            'interpolation_reliable': bad_percentage <= 30,
            'validation_method': 'lof_redetection'
        }

        # Log results
        if validation_results['successfully_interpolated']:
            logger.success(f"Successfully interpolated {len(validation_results['successfully_interpolated'])} channels: {sorted(validation_results['successfully_interpolated'])}")

        if validation_results['still_noisy']:
            logger.warning(f"{len(validation_results['still_noisy'])} channels still noisy after interpolation: {sorted(validation_results['still_noisy'])}")

        logger.info(f"Interpolation success rate: {validation_results['success_rate']:.1%}")

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raw.info['bads'] = bads_before  # Restore original bad channels list
        raw._interpolation_details = {
            'attempted': sorted(bads_before),
            'successfully_interpolated': [],
            'still_noisy': sorted(bads_before),
            'success_rate': 0.0,
            'bad_percentage_before': bad_percentage,
            'interpolation_reliable': False,
            'validation_method': 'lof_redetection',
            'error': str(e)
        }


def _validate_interpolation_with_lof(raw: BaseRaw, original_bads: List[str], n_neighbors: int, threshold: float, verbose: bool) -> Dict:
    """
    Validate interpolation success by re-running LOF detection.

    If interpolation worked, channels shouldn't be detected as bad anymore.

    Args:
        raw: Raw data with interpolated channels
        original_bads: List of originally bad channels that were interpolated
        verbose: Enable detailed logging

    Returns:
        Dictionary with validation results
    """
    from mne.preprocessing import find_bad_channels_lof

    if verbose:
        logger.info("Validating interpolation success using LOF re-detection...")

    try:
        # Re-run LOF detection with same parameters on interpolated data
        # Temporarily clear bads to let LOF detect all outliers fresh
        temp_bads = raw.info['bads'].copy()
        raw.info['bads'] = []

        # Run LOF detection
        still_bad_lof = find_bad_channels_lof(
            raw,
            n_neighbors=n_neighbors,
            threshold=threshold,
            verbose=verbose,
            picks='eeg'
        )

        # Restore the bads list
        raw.info['bads'] = temp_bads

        # Analyze which originally bad channels are still detected by LOF
        still_noisy = [ch for ch in original_bads if ch in still_bad_lof]
        successfully_interpolated = [ch for ch in original_bads if ch not in still_bad_lof]

        if verbose:
            if still_bad_lof:
                logger.debug(f"LOF re-detection found {len(still_bad_lof)} outlier channels: {sorted(still_bad_lof)}")
            else:
                logger.debug("LOF re-detection found no outlier channels")

            logger.debug(f"Of {len(original_bads)} interpolated channels:")
            logger.debug(f"  - {len(successfully_interpolated)} no longer detected as outliers")
            logger.debug(f"  - {len(still_noisy)} still detected as outliers")

        return {
            'still_noisy': still_noisy,
            'successfully_interpolated': successfully_interpolated,
            'success_rate': len(successfully_interpolated) / len(original_bads) if original_bads else 1.0,
            'all_detected_outliers': still_bad_lof
        }

    except Exception as e:
        logger.error(f"LOF validation failed: {e}")
        # Fallback - assume all interpolation failed
        return {
            'still_noisy': original_bads,
            'successfully_interpolated': [],
            'success_rate': 0.0,
            'all_detected_outliers': [],
            'validation_error': str(e)
        }


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
                ax.plot(time, ch_data, label=f'{ch} (LOF detected)', alpha=0.8)

        title = f'LOF detected bad channels: {sorted(detected_bads)}'
        if interpolate:
            title += ' (interpolated)' if ch in raw.ch_names else ' (interpolation failed)'
        ax.set_title(title)
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


def print_interpolation_report(raw: BaseRaw) -> None:
    """
    Print a detailed report of interpolation results.

    Args:
        raw: Processed raw data with interpolation details
    """
    if not hasattr(raw, '_bad_channel_metrics'):
        print("No bad channel metrics available")
        return

    metrics = raw._bad_channel_metrics
    interp_details = metrics.get('interpolation_details', {})

    print("\n" + "="*60)
    print("BAD CHANNEL INTERPOLATION REPORT")
    print("="*60)

    print(f"Original bad channels: {metrics.get('n_original', 0)} ({metrics.get('original_bads', [])})")
    print(f"Detected bad channels: {metrics.get('n_detected', 0)} ({metrics.get('detected_bads', [])})")
    print(f"Final bad channels: {metrics.get('n_final', 0)} ({metrics.get('final_bads', [])})")

    if interp_details:
        print(f"\nInterpolation Details:")
        print(f"  Attempted: {len(interp_details.get('attempted', []))} channels")
        print(f"  Successfully interpolated: {len(interp_details.get('successfully_interpolated', []))} channels")
        print(f"  Still noisy after interpolation: {len(interp_details.get('still_noisy', []))} channels")
        print(f"  Success rate: {interp_details.get('success_rate', 0):.1%}")
        print(f"  Bad channel percentage: {interp_details.get('bad_percentage_before', 0):.1f}%")
        print(f"  Interpolation reliable: {interp_details.get('interpolation_reliable', False)}")
        print(f"  Validation method: {interp_details.get('validation_method', 'unknown')}")

        if interp_details.get('still_noisy'):
            print(f"  Still noisy channels: {sorted(interp_details['still_noisy'])}")

        if interp_details.get('error'):
            print(f"  Error: {interp_details['error']}")

    print("="*60)


def check_interpolation_quality(raw: BaseRaw,
                               plot_noisy_channels: bool = True,
                               duration: float = 5.0) -> Dict:
    """
    Check the quality of interpolated channels and optionally plot them.

    Args:
        raw: Processed raw data
        plot_noisy_channels: Whether to plot channels that are still noisy
        duration: Duration of data to plot (seconds)

    Returns:
        Dictionary with quality assessment
    """
    if not hasattr(raw, '_bad_channel_metrics'):
        logger.warning("No bad channel metrics available for quality check")
        return {}

    metrics = raw._bad_channel_metrics
    interp_details = metrics.get('interpolation_details', {})

    # Print summary
    print_interpolation_report(raw)

    # Plot still-noisy channels if requested
    if plot_noisy_channels and interp_details.get('still_noisy'):
        still_noisy = interp_details['still_noisy']

        if len(still_noisy) <= 10:  # Don't overwhelm with too many plots
            logger.info(f"Plotting {len(still_noisy)} channels that are still noisy after interpolation...")

            # Get data for plotting
            start_time = 5.0  # Skip first few seconds
            start_idx = int(start_time * raw.info['sfreq'])
            end_idx = start_idx + int(duration * raw.info['sfreq'])

            fig, axes = plt.subplots(len(still_noisy), 1, figsize=(12, 2*len(still_noisy)),
                                   sharex=True)
            if len(still_noisy) == 1:
                axes = [axes]

            time = raw.times[start_idx:end_idx]

            for i, ch in enumerate(still_noisy):
                if ch in raw.ch_names:
                    ch_idx = raw.ch_names.index(ch)
                    ch_data = raw.get_data(picks=[ch_idx], start=start_idx, stop=end_idx)[0] * 1e6

                    axes[i].plot(time, ch_data, 'r-', alpha=0.8)
                    axes[i].set_title(f'Still Noisy: {ch}')
                    axes[i].set_ylabel('Amplitude (µV)')
                    axes[i].grid(True, alpha=0.3)

            axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.show()
        else:
            logger.info(f"Too many noisy channels ({len(still_noisy)}) to plot individually")

    # Return quality assessment
    quality_assessment = {
        'interpolation_successful': interp_details.get('success_rate', 0) > 0.8,
        'high_bad_percentage': interp_details.get('bad_percentage_before', 0) > 30,
        'channels_still_noisy': len(interp_details.get('still_noisy', [])),
        'recommendation': 'good' if interp_details.get('success_rate', 0) > 0.8 else 'review_data'
    }

    if quality_assessment['high_bad_percentage']:
        logger.warning("High percentage of bad channels detected - consider reviewing data quality")

    if quality_assessment['channels_still_noisy'] > 5:
        logger.warning(f"{quality_assessment['channels_still_noisy']} channels still noisy - interpolation may be unreliable")

    return quality_assessment