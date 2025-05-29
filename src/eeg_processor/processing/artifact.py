# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union, Tuple
import yaml

import numpy as np
from mne import Epochs, concatenate_raws, pick_types, pick_channels_regexp
from mne.io import BaseRaw
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import random

## Bad Channels ##
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
        plot_duration: float = 2.0,  # seconds of data to show
        plot_start: float = 5.0  # Start time for plot (seconds)
) -> BaseRaw:
    """
    Core bad channel detection and interpolation logic with failure handling.

    Args:
        raw: MNE Raw object
        flat_threshold: Threshold for flat channels (V)
        noisy_threshold: Threshold for noisy channels (SD)
        segment_wise: Enable segment-wise processing
        segment_length: Segment duration (seconds)
        interpolate: Whether to interpolate bad channels
        verbose: Show detailed logging

    Returns:
        Processed Raw object with updated bad channels
    """
    if inplace:
        raw_processed = raw
    else:
        raw_processed = raw.copy()

    original_bads = set(raw_processed.info['bads'])

    logger.info(f"Starting bad channel detection (interpolate={interpolate}, segment_wise={segment_wise})")
    logger.debug(f"Original bad channels: {original_bads or 'None'}")

    if segment_wise:
        segments = []
        all_detected_bads = set()
        n_segments = int(np.ceil(raw_processed.times[-1] / segment_length))
        logger.info(f"Processing data in {n_segments} segments of {segment_length}s")

        for i in range(n_segments):
            tmin = i * segment_length
            tmax = min((i + 1) * segment_length, raw_processed.times[-1])
            segment = raw_processed.copy().crop(tmin=tmin, tmax=tmax)

            bads = _find_bad_channels_in_segment(
                segment,
                flat_threshold,
                noisy_threshold
            )

            if bads:
                logger.info(f"Segment {i + 1}/{n_segments} bad channels: {bads}")
                all_detected_bads.update(bads)

                if interpolate:
                    bads_before_interp = bads.copy()
                    segment.info['bads'] = bads

                    try:
                        segment.interpolate_bads(reset_bads=True, verbose=verbose)

                        # Check interpolation success
                        still_bad = segment.info['bads']
                        successfully_interpolated = [ch for ch in bads_before_interp if ch not in still_bad]

                        if successfully_interpolated:
                            logger.debug(f"Segment {i + 1}: Successfully interpolated {successfully_interpolated}")
                        if still_bad:
                            logger.warning(f"Segment {i + 1}: Interpolation failed for {still_bad}")

                    except Exception as e:
                        logger.error(f"Segment {i + 1}: Interpolation failed: {e}")
                        segment.info['bads'] = bads_before_interp

            segments.append(segment)

        raw_processed = concatenate_raws(segments)

        # Update global bads list
        all_bads_combined = list(original_bads | all_detected_bads)
        if all_bads_combined:
            raw_processed.info['bads'] = all_bads_combined

        logger.success(f"Processed all segments. Total bad channels: {len(all_bads_combined)}")

    else:
        # Global processing
        detected_bads = _find_bad_channels_in_segment(
            raw_processed,
            flat_threshold,
            noisy_threshold
        )

        # Combine original and detected bad channels
        all_bads = list(original_bads | set(detected_bads))
        raw_processed.info['bads'] = all_bads

        if detected_bads:
            logger.info(f"New bad channels detected: {detected_bads}")

            if interpolate:
                bads_before_interp = all_bads.copy()

                try:
                    raw_processed.interpolate_bads(reset_bads=True, verbose=verbose)

                    # Check interpolation success
                    still_bad = raw_processed.info['bads']
                    successfully_interpolated = [ch for ch in bads_before_interp if ch not in still_bad]

                    if successfully_interpolated:
                        logger.success(f"Successfully interpolated: {successfully_interpolated}")
                    if still_bad:
                        logger.warning(f"Interpolation failed for: {still_bad} - these will be excluded from analysis")

                except Exception as e:
                    logger.error(f"Interpolation failed: {e}")
                    # Keep all channels as bad if interpolation completely fails
                    raw_processed.info['bads'] = bads_before_interp

        else:
            logger.info("No new bad channels detected")

    # Final reporting
    final_bads = set(raw_processed.info['bads'])
    new_bads = final_bads - original_bads

    if new_bads:
        logger.info(f"Summary - New bad channels: {new_bads}")
    else:
        logger.info("Summary - No additional bad channels found")

    # Store metrics for quality tracking (accessible via the function's result)
    raw_processed._bad_channel_metrics = {
        'original_bads': list(original_bads),
        'detected_bads': list(new_bads) if not segment_wise else list(all_detected_bads),
        'final_bads': list(final_bads),
        'n_original': len(original_bads),
        'n_detected': len(new_bads) if not segment_wise else len(all_detected_bads),
        'n_final': len(final_bads),
        'interpolation_attempted': interpolate
    }

    if show_plot and (new_bads or original_bads):
        import matplotlib.pyplot as plt

        # Prepare data
        start_idx = int(plot_start * raw.info['sfreq'])
        end_idx = start_idx + int(plot_duration * raw.info['sfreq'])
        time = raw.times[start_idx:end_idx]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                       sharex=True, sharey=True)

        # Plot original bad channels
        if original_bads:
            orig_data = raw.get_data(picks=list(original_bads),
                                     start=start_idx, stop=end_idx) * 1e6  # Convert to µV
            for i, ch in enumerate(original_bads):
                ax1.plot(time, orig_data[i].T,
                         label=ch, alpha=0.8)
            ax1.set_title(f'Original bad channels: {original_bads}')
            ax1.legend(loc='upper right')
            ax1.set_ylabel('Amplitude (µV)')
            ax1.grid(True)

        # Plot newly detected bad channels
        if new_bads:
            # Get pre-interpolation data
            new_data = raw.get_data(picks=list(new_bads),
                                    start=start_idx, stop=end_idx) * 1e6

            # Plot original bad channels
            for i, ch in enumerate(new_bads):
                ax2.plot(time, new_data[i].T,
                         color='tab:orange', alpha=0.8, label=f'{ch} (original)')

            # Plot interpolated versions if enabled
            if interpolate:
                interp_data = raw_processed.get_data(picks=list(new_bads),
                                                     start=start_idx, stop=end_idx) * 1e6
                for i, ch in enumerate(new_bads):
                    ax2.plot(time, interp_data[i].T,
                             color='tab:green', alpha=0.8, linestyle='--',
                             label=f'{ch} (interpolated)')

            ax2.set_title(f'New bad channels: {new_bads}')
            ax2.legend(loc='upper right')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude (µV)')
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return raw_processed


def _find_bad_channels_in_segment(
        raw_segment: BaseRaw,
        flat_threshold: float,
        noisy_threshold: float
) -> List[str]:
    """Internal helper for channel quality metrics"""
    data = raw_segment.get_data()
    variances = np.var(data, axis=1)

    eog_channels = ['HEOG', 'VEOG', 'EOG']
    stim_channels = ['STIM', 'TRIGGER']
    exclude_channels = eog_channels + stim_channels

    # Flat channel detection
    flat_chans = [
        raw_segment.ch_names[i]
        for i, v in enumerate(variances)
        if v < flat_threshold
    ]

    flat_chans = [chan for chan in flat_chans if chan not in exclude_channels]

    if flat_chans:
        logger.debug(f"Flat channels (var < {flat_threshold:.1e}): {flat_chans}")

    # Noisy channel detection (MAD-based)
    median = np.median(variances)
    mad = 1.4826 * np.median(np.abs(variances - median))
    noisy_chans = [
        raw_segment.ch_names[i]
        for i, v in enumerate(variances)
        if abs(v - median) > noisy_threshold * mad
    ]

    noisy_chans = [chan for chan in noisy_chans if chan not in exclude_channels]

    if noisy_chans:
        logger.debug(f"Noisy channels ({noisy_threshold}×MAD): {noisy_chans}")

    bads = list(set(flat_chans + noisy_chans) - set(raw_segment.info['bads']))


    return bads

## Regression Blink Correction ##
def remove_blinks_regression(
        raw: BaseRaw,
        eog_channels: List[str] = ['HEOG', 'VEOG'],
        show_plot: bool = True,
        plot_duration: float = 3.0,
        verbose: bool = False
) -> BaseRaw:
    """
    Remove blinks using linear regression with temporary average reference.
    Restores original reference after correction.
    """
    # Validate EOG channels exist
    missing_eogs = [ch for ch in eog_channels if ch not in raw.ch_names]
    if missing_eogs:
        raise ValueError(f"Missing EOG channels: {missing_eogs}")

    # Create safety copy
    original_raw = raw.copy()

    #Store original reference info
    orig_ref = {
        # Store which channels were EEG channels
        'eeg_ch_names': [ch for i, ch in enumerate(original_raw.ch_names)
                         if i in pick_types(original_raw.info, eeg=True, meg=False)],
        # Store reference description
        'description': original_raw.info.get('description', None),
        # Store bad channels
        'bads': original_raw.info['bads'].copy()
    }

    # Apply temporary average reference
    original_raw.set_eeg_reference(ref_channels='average', projection=False)

    # Detect blink events
    from mne.preprocessing import EOGRegression, find_eog_events
    eog_events = find_eog_events(original_raw, ch_name=eog_channels[1])
    if verbose:
        logger.info(f"Found {len(eog_events)} blink events")
    if len(eog_events) == 0:
        raise ValueError("No blink events detected in EOG channels!")

    # Add diagnostic plot before correction
    if show_plot:
        plot_eog_with_blinks(raw.copy(), eog_channels)

    # Fit regression model to blinks
    eog_indices = [original_raw.ch_names.index(ch) for ch in eog_channels]
    model = EOGRegression(
        picks=pick_types(original_raw.info, eeg=True),
        picks_artifact=eog_indices
    ).fit(original_raw)  # <-- Fix: Remove `eog_events` here

    cleaned_raw = model.apply(original_raw)

    # Restore original reference if needed
    if orig_ref['description'] != 'average':
        if verbose:
            logger.info("Restoring original reference scheme")

        # Case 1: Specific reference channels (like ['M1', 'M2'])
        if isinstance(orig_ref['description'], list):
            available_refs = [ch for ch in orig_ref['description']
                              if ch in cleaned_raw.ch_names]
            if available_refs:
                cleaned_raw.set_eeg_reference(ref_channels=available_refs, projection=False)
            elif verbose:
                logger.warning(f"Original reference channels {orig_ref['description']} not available")

        # Case 2: Other reference schemes
        elif orig_ref['description']:
            try:
                cleaned_raw.set_eeg_reference(ref_channels=orig_ref['description'], projection=False)
            except Exception as e:
                if verbose:
                    logger.warning(f"Could not restore original reference: {str(e)}")

    # Restore bad channel info
    cleaned_raw.info['bads'] = orig_ref['bads']

    if show_plot:
        # Select a random EEG channel
        eeg_chs = pick_types(raw.info, eeg=True)
        #chan_idx = random.choice(eeg_chs)
        chan_idx = 1
        chan_name = raw.ch_names[chan_idx]

        # Prepare data for plotting
        n_samples = int(plot_duration * raw.info['sfreq'])
        time = np.arange(n_samples) / raw.info['sfreq']
        orig_data = original_raw.get_data(picks=chan_idx)[0, :n_samples] * 1e6  # μV
        clean_data = cleaned_raw.get_data(picks=chan_idx)[0, :n_samples] * 1e6

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, orig_data, 'b', label='Original')
        ax.plot(time, clean_data, 'r', alpha=0.7, label='Cleaned')
        ax.set_title(f"Blink Regression: {chan_name}\n(Reference: {original_raw.info['description'] or 'Original'})")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return cleaned_raw


def plot_eog_with_blinks(raw: BaseRaw, eog_channels: List[str], plot_duration: float = 30.0):
    """
    Plot EOG channels with annotated blink events.

    Args:
        raw: Raw EEG data with EOG channels
        eog_channels: List of EOG channel names (e.g., ['HEOG', 'VEOG'])
        plot_duration: Time window to plot (seconds)
    """
    from mne import Annotations
    from mne.preprocessing import find_eog_events

    # Detect blink events
    eog_events = find_eog_events(raw, ch_name=eog_channels[1])
    print(f"Detected {len(eog_events)} blink events")

    # Create annotations from events
    blink_annot = Annotations(
        onset=eog_events[:, 0] / raw.info['sfreq'],  # Convert samples to seconds
        duration=0.3,  # Short duration for blink markers
        description="blink"
    )

    # Plot with annotations
    fig = raw.copy().set_annotations(blink_annot).plot(
        picks=eog_channels,
        duration=plot_duration,
        title=f"EOG Channels | Detected Blinks: {len(eog_events)}",
        scalings={"eog": 100e-6},  # Scale EOG traces for better visibility
        block=True
    )

    return fig



## ICA Functions ##
def remove_blinks_with_ica(raw: BaseRaw,
                           n_components: Union[float, int] = 0.99,
                           plot_components: bool = True,
                           eog_ch: Optional[Union[str, List[str]]] = ['VEOG', 'HEOG'],
                           ecg_ch: Optional[str] = None,
                           decim: int = None,
                           enable_manual: bool = False,
                           auto_classify: bool = True,
                           muscle_threshold: float = 0.8,
                           eye_threshold: float = 0.8,
                           inplace: bool = False,
                           verbose: Optional[Union[bool, str]] = None,
                           **kwargs) -> BaseRaw:
    """
    Complete ICA-based artifact removal with multi-method agreement analysis

    Args:
        raw: Raw EEG data
        n_components: Number of ICA components
        plot_components: Whether to plot component analysis
        eog_ch: EOG channel(s) for blink detection
        ecg_ch: ECG channel for cardiac detection
        decim: Decimation factor for ICA fitting
        enable_manual: Allow manual component selection
        auto_classify: Use ICALabel for automatic classification
        muscle_threshold: ICALabel confidence threshold for muscle artifacts
        eye_threshold: ICALabel confidence threshold for eye artifacts
        require_agreement: Only exclude components found by multiple methods
        agreement_method: "conservative" (intersection) or "liberal" (union)
        inplace: Ignored - ICA always creates new object
        verbose: Verbosity level
        **kwargs: Additional ICA parameters

    Returns:
        Raw object with artifacts removed via ICA
    """
    import numpy as np
    from loguru import logger

    if inplace:
        logger.info("inplace=True ignored for ICA - always creates new object")

    logger.info(f"Starting ICA artifact removal (n_components={n_components})")

    # Work with input data directly for ICA fitting
    current_data = raw

    # Fit ICA
    ica = ICA(
        n_components=n_components,
        method='infomax',
        fit_params=dict(extended=True),
        random_state=42,
        max_iter=1000,
        verbose=verbose
    )

    logger.info("Fitting ICA...")
    ica.fit(current_data, decim=decim)
    logger.success(f"ICA fitted with {ica.n_components_} components")

    # Initialize detection results storage
    detection_results = {
        'icalabel_eye': [],
        'icalabel_muscle': [],
        'icalabel_heart': [],
        'icalabel_line': [],
        'icalabel_noise': [],
        'eog_detection': [],
        'ecg_detection': []
    }

    # === ICALabel Detection ===
    if auto_classify:
        try:
            from mne_icalabel import label_components

            logger.info("Running ICALabel classification...")
            ic_labels = label_components(raw, ica, method='iclabel')

            # Extract labels and probabilities
            labels = ic_labels['labels']
            probabilities = ic_labels['y_pred_proba']

            for i, (label, probs) in enumerate(zip(labels, probabilities)):
                max_prob = np.max(probs)

                # Store detections by type
                if label == 'muscle artifact' and max_prob >= muscle_threshold:
                    detection_results['icalabel_muscle'].append(i)
                    logger.info(f"ICALabel muscle: ICA{i:03d} (confidence: {max_prob:.2f})")

                elif label == 'eye blink' and max_prob >= eye_threshold:
                    detection_results['icalabel_eye'].append(i)
                    logger.info(f"ICALabel eye: ICA{i:03d} (confidence: {max_prob:.2f})")

                elif label == 'heart beat' and max_prob >= 0.8:
                    detection_results['icalabel_heart'].append(i)
                    logger.info(f"ICALabel heart: ICA{i:03d} (confidence: {max_prob:.2f})")

                elif label == 'line noise' and max_prob >= 0.8:
                    detection_results['icalabel_line'].append(i)
                    logger.info(f"ICALabel line: ICA{i:03d} (confidence: {max_prob:.2f})")

                elif label == 'channel noise' and max_prob >= 0.8:
                    detection_results['icalabel_noise'].append(i)
                    logger.info(f"ICALabel noise: ICA{i:03d} (confidence: {max_prob:.2f})")

            # Store classification results
            ica.labels_ = ic_labels

        except ImportError:
            logger.error("mne-icalabel not installed. Install with: pip install mne-icalabel")
            auto_classify = False
        except Exception as e:
            logger.error(f"ICALabel classification failed: {str(e)}")
            auto_classify = False

    # === EOG-based Detection ===
    if eog_ch:
        logger.info("=== EOG-based Component Detection ===")
        if isinstance(eog_ch, list):
            for eog_channel in eog_ch:
                if eog_channel in current_data.ch_names:
                    eog_components = ica.find_bads_eog(current_data, eog_channel, verbose=False)[0]
                    logger.info(f"{eog_channel} detection ({eog_channel}): {np.array(eog_components)}")
                    detection_results['eog_detection'].extend(eog_components)
        elif isinstance(eog_ch, str):
            if eog_ch in current_data.ch_names:
                eog_components = ica.find_bads_eog(current_data, eog_ch, verbose=False)[0]
                logger.info(f"EOG detection ({eog_ch}): {np.array(eog_components)}")
                detection_results['eog_detection'].extend(eog_components)

        # Remove duplicates from EOG detection
        detection_results['eog_detection'] = list(set(detection_results['eog_detection']))

    # === ECG Detection ===
    if ecg_ch and ecg_ch in current_data.ch_names:
        # Use provided ECG channel
        logger.info("=== ECG-based Component Detection (with ECG channel) ===")
        ecg_components = ica.find_bads_ecg(current_data, ecg_ch)[0]
        logger.info(f"ECG detection (manual channel {ecg_ch}): {np.array(ecg_components)}")
        detection_results['ecg_detection'] = ecg_components
    else:
        detection_results['ecg_detection'] = []


    # Overall detection summary
    logger.info(f"\nDetection summary:")
    logger.info(f"  ICALabel muscle: {sorted(detection_results['icalabel_muscle'])}")
    logger.info(f"  ICALabel eye: {sorted(detection_results['icalabel_eye'])}")
    logger.info(f"  ICALabel heart: {sorted(detection_results['icalabel_heart'])}")
    logger.info(f"  ICALabel line: {sorted(detection_results['icalabel_line'])}")
    logger.info(f"  ICALabel noise: {sorted(detection_results['icalabel_noise'])}")
    logger.info(f"  EOG detection: {sorted(detection_results['eog_detection'])}")
    logger.info(f"  ECG detection: {sorted(detection_results['ecg_detection'])}")

    final_excludes = list(set(detection_results['icalabel_muscle'] + detection_results['icalabel_eye'] +
                              detection_results['icalabel_heart'] + detection_results['icalabel_line'] +
                              detection_results['icalabel_noise'] + detection_results['eog_detection'] +
                              detection_results['ecg_detection']))

    # Remove duplicates and sort
    final_excludes = sorted(list(set(final_excludes)))
    logger.info(f"Final components to exclude: {final_excludes}")

    # === Plotting and Visualization ===
    if plot_components and final_excludes:
        logger.info("Plotting artifact component properties...")
        ica.plot_properties(
            current_data,
            picks=final_excludes,
            psd_args={'fmax': 30},
        )

    # Plot all components if requested
    if plot_components:
        logger.info("Plotting all ICA components...")
        ica.plot_components()

        ica.plot_sources()

        if final_excludes:
            logger.info("Plotting overlay comparison...")
            ica.plot_overlay(current_data, exclude=final_excludes)

    # === Manual Override ===
    if enable_manual and plot_components:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close('all')

        logger.info("Opening manual component selection interface...")
        ica.plot_sources(current_data, title="Click components to exclude. Close window when done.", block=True)
        manual_excludes = list(ica.exclude)

        # Combine automatic and manual selections
        combined_excludes = list(set(final_excludes + manual_excludes))
        logger.info(f"Manual additions: {sorted(set(manual_excludes) - set(final_excludes))}")
        final_excludes = combined_excludes

    # === Apply ICA Cleaning ===
    logger.info(f"Applying ICA with {len(final_excludes)} excluded components: {final_excludes}")
    cleaned_raw = ica.apply(current_data, exclude=final_excludes)

    logger.success(f"ICA cleaning completed. Excluded {len(final_excludes)} components.")
    return cleaned_raw

