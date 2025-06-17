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

# ASR import (try/except for graceful degradation)
try:
    import asrpy
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    asrpy = None

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
    new_bads = set(detected_bads) - set(original_bads)

    if final_bads:
        logger.info(f"Summary - Final bad channels: {final_bads}")
    else:
        logger.info("Summary - No bad channels found")

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


## ASR (Artifact Subspace Reconstruction) ##
def clean_rawdata_asr(
        raw: BaseRaw,
        cutoff: Union[int, float] = 20,
        method: str = "euclid",
        blocksize: Optional[int] = None,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        calibration_duration: Optional[float] = None,
        show_plot: bool = False,
        plot_duration: float = 10.0,
        plot_start: float = 5.0,
        inplace: bool = False,
        verbose: bool = False
) -> BaseRaw:
    """
    Clean raw EEG data using Artifact Subspace Reconstruction (ASR).
    
    ASR is an intermediate data cleaning step designed to be applied after bad channel 
    detection/interpolation but before ICA. It identifies and corrects brief high-amplitude 
    artifacts by reconstructing corrupted signal subspaces using a calibration-based approach.
    
    **Recommended Pipeline Position:**
    1. Bad channel detection and interpolation
    2. ASR data cleaning (this function) ← Intermediate step  
    3. ICA artifact removal (for remaining component-based artifacts)
    
    Args:
        raw: MNE Raw object to process
        cutoff: Standard deviation cutoff for artifact detection (default: 20)
                Higher values = more conservative (less correction)
                Lower values = more aggressive (more correction)
                Typical range: 5-100, recommend 10-30 for most applications
        method: Distance metric for ASR algorithm ('euclid' or 'riemann')
        blocksize: Block size for processing (samples). If None, auto-determined
        window_length: Sliding window length in seconds (default: 0.5)
        window_overlap: Window overlap fraction (default: 0.66, i.e., 66%)
        max_dropout_fraction: Max fraction of channels to remove per window (default: 0.1)
        min_clean_fraction: Min fraction of calibration data required (default: 0.25)
        calibration_duration: Duration of data to use for inline calibration (seconds). 
                              If None, uses ASRpy defaults (typically 60s or 25% of data)
        show_plot: Display before/after comparison plot
        plot_duration: Duration of plot comparison (seconds)
        plot_start: Start time for plot (seconds)
        inplace: Ignored (ASR always creates new object)
        verbose: Show detailed processing information
        
    Returns:
        Raw object with ASR artifact correction applied
        
    Notes:
        **Why ASR as Intermediate Step:**
        - ASR corrects transient high-amplitude artifacts (motion, electrode pops, etc.)
        - These artifacts can interfere with ICA decomposition quality
        - ASR preserves underlying neural signals better than simple rejection
        - Cleaner data leads to better ICA component separation
        
        **Calibration Approach:**
        - **Automatic inline**: Uses beginning of provided data for calibration
        - **Custom duration**: Specify calibration_duration for explicit calibration length
        - ASRpy automatically identifies clean segments within the specified duration
        
        **Calibration Requirements:**
        - Minimum 30 seconds of relatively clean data
        - Recommended 60+ seconds for optimal performance
        - Data should be high-pass filtered (≥0.5 Hz) before ASR
        - At least min_clean_fraction (25%) must be artifact-free
        
        **Quality Tracking:**
        - Stores detailed metrics in raw._asr_metrics for research QC
        - Includes calibration quality, correlation preservation, variance changes
        - ASR preserves brain signals while removing transient artifacts
        
        **Research Recommendations:**
        - Apply after bad channel detection/interpolation but before ICA
        - Use consistent parameters across subjects in a study
        - Include 1-2 minutes of eyes-closed resting data at recording start
        - Validate results with correlation analysis (should be >0.8)
        
    References:
        - Mullen et al. (2015). Real-time neuroimaging and cognitive monitoring
          using wearable dry EEG. IEEE Trans Biomed Eng, 62(11), 2553-2567.
        - Kothe & Jung (2016). Artifact removal techniques for EEG recordings.
        - Chang et al. (2020). Evaluation of artifact subspace reconstruction for 
          automatic artifact removal in single-trial analysis of ERPs. NeuroImage.
    """
    if not ASR_AVAILABLE:
        raise ImportError(
            "ASRpy is not installed. Please install it with: pip install asrpy"
        )
    
    if inplace:
        logger.info("inplace=True ignored for ASR - always creates new object")
    
    logger.info(f"Starting ASR data cleaning (cutoff={cutoff}, method={method})")
    
    # Store original info for comparison
    original_raw = raw.copy()
    n_channels = len(raw.ch_names)
    duration = raw.times[-1]
    
    # Validate parameters
    if cutoff <= 0:
        raise ValueError("ASR cutoff must be positive")
    if not 0 < window_overlap < 1:
        raise ValueError("Window overlap must be between 0 and 1")
    if not 0 < max_dropout_fraction <= 1:
        raise ValueError("Max dropout fraction must be between 0 and 1")
    if not 0 < min_clean_fraction <= 1:
        raise ValueError("Min clean fraction must be between 0 and 1")
    
    # Validate data length for calibration
    data_duration = raw.times[-1]
    if calibration_duration is not None:
        if calibration_duration > data_duration:
            logger.warning(f"Requested calibration duration ({calibration_duration:.1f}s) "
                          f"exceeds data length ({data_duration:.1f}s). Using full data length.")
            calibration_duration = None
        elif calibration_duration < 30:
            logger.warning(f"Calibration duration ({calibration_duration:.1f}s) is quite short. "
                          f"Consider using at least 30 seconds for robust calibration.")
    
    if verbose:
        if calibration_duration is not None:
            logger.info(f"Using first {calibration_duration:.1f}s of data for ASR calibration")
        else:
            logger.info(f"Using ASRpy automatic calibration on {data_duration:.1f}s of data")
    
    # Determine block size if not specified
    if blocksize is None:
        # Use ASRpy's default behavior or calculate based on data length
        sfreq = raw.info['sfreq']
        # Aim for blocks of ~30 seconds or data length, whichever is smaller
        blocksize = min(int(30 * sfreq), len(raw.times))
        if verbose:
            logger.info(f"Auto-determined blocksize: {blocksize} samples ({blocksize/sfreq:.1f}s)")
    
    try:
        # Initialize ASR
        asr = asrpy.ASR(
            sfreq=raw.info['sfreq'],
            cutoff=cutoff,
            method=method,
            blocksize=blocksize,
            win_len=window_length,
            win_overlap=window_overlap,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction
        )
        
        if verbose:
            logger.info(f"ASR initialized with parameters:")
            logger.info(f"  Sampling frequency: {raw.info['sfreq']:.1f} Hz")
            logger.info(f"  Cutoff: {cutoff}")
            logger.info(f"  Method: {method}")
            logger.info(f"  Block size: {blocksize} samples")
            logger.info(f"  Window length: {window_length}s")
            logger.info(f"  Window overlap: {window_overlap:.1%}")
        
        # Apply ASR with optional calibration duration
        logger.info("Applying ASR correction...")
        if calibration_duration is not None:
            # Use explicit calibration duration by cropping calibration data
            # but apply to full data
            calibration_raw_subset = original_raw.copy().crop(tmax=calibration_duration)
            if verbose:
                logger.info(f"Fitting ASR on first {calibration_duration:.1f}s of data...")
            asr.fit(calibration_raw_subset)
            if verbose:
                logger.info("Applying ASR correction to full dataset...")
            cleaned_raw = asr.transform(original_raw.copy())
        else:
            # Use ASRpy default inline calibration
            asr.fit(original_raw)
            cleaned_raw = asr.transform(original_raw.copy())

        # Validate output
        if not isinstance(cleaned_raw, BaseRaw):
            raise RuntimeError("ASR did not return a valid MNE Raw object")
        
        # Calculate correction statistics
        original_data = raw.get_data()
        cleaned_data = cleaned_raw.get_data()
        
        # Calculate RMS difference for each channel
        rms_original = np.sqrt(np.mean(original_data**2, axis=1))
        rms_cleaned = np.sqrt(np.mean(cleaned_data**2, axis=1))
        rms_change = (rms_cleaned - rms_original) / rms_original * 100
        
        # Calculate correlation between original and cleaned data
        correlations = []
        for ch_idx in range(n_channels):
            corr = np.corrcoef(original_data[ch_idx], cleaned_data[ch_idx])[0, 1]
            correlations.append(corr)
        
        mean_correlation = np.nanmean(correlations)
        
        # Calculate global metrics
        global_var_original = np.var(original_data)
        global_var_cleaned = np.var(cleaned_data)
        variance_change = (global_var_cleaned - global_var_original) / global_var_original * 100
        
        # Store comprehensive metrics for quality tracking
        cleaned_raw._asr_metrics = {
            'cutoff': cutoff,
            'method': method,
            'blocksize': blocksize,
            'window_length': window_length,
            'window_overlap': window_overlap,
            'max_dropout_fraction': max_dropout_fraction,
            'min_clean_fraction': min_clean_fraction,
            'calibration_duration': calibration_duration,
            'data_duration': duration,
            'calibration_approach': 'explicit_duration' if calibration_duration else 'automatic_inline',
            'processing_time': duration,
            'n_channels': n_channels,
            'mean_correlation': float(mean_correlation),
            'variance_change_percent': float(variance_change),
            'channel_rms_changes': rms_change.tolist(),
            'channel_correlations': [float(c) for c in correlations],
            'parameters': {
                'cutoff': cutoff,
                'method': method,
                'blocksize': blocksize,
                'calibration_duration': calibration_duration,
                'window_params': {
                    'length': window_length,
                    'overlap': window_overlap
                },
                'dropout_params': {
                    'max_dropout_fraction': max_dropout_fraction,
                    'min_clean_fraction': min_clean_fraction
                }
            }
        }
        
        # Log summary
        logger.success(f"ASR correction completed")
        logger.info(f"  Calibration: {calibration_duration:.1f}s" if calibration_duration else "  Calibration: automatic inline")
        logger.info(f"  Mean channel correlation: {mean_correlation:.3f}")
        logger.info(f"  Variance change: {variance_change:+.1f}%")
        
        channels_changed = np.sum(np.abs(rms_change) > 5)  # Channels with >5% RMS change
        logger.info(f"  Channels with significant change: {channels_changed}/{n_channels}")
        
        # Enhanced calibration quality assessment
        if mean_correlation < 0.8:
            logger.warning(f"  Low correlation ({mean_correlation:.3f}) suggests poor calibration or excessive correction")
            logger.warning("  Consider: longer calibration data, higher cutoff, or checking data quality")
        
        if channels_changed == 0:
            logger.info("  No significant artifacts detected - minimal correction applied")
        elif channels_changed > n_channels * 0.5:
            logger.warning(f"  High number of channels corrected ({channels_changed}). Consider checking data quality.")
            logger.warning("  Potential issues: insufficient calibration data or very noisy data")
        
        # Optional visualization
        if show_plot:
            _plot_asr_comparison(raw, cleaned_raw, plot_start, plot_duration)
        
        return cleaned_raw
        
    except Exception as e:
        logger.error(f"ASR correction failed: {str(e)}")
        logger.info("Returning original data unchanged")
        
        # Return original data with error metrics
        error_raw = raw.copy()
        error_raw._asr_metrics = {
            'cutoff': cutoff,
            'method': method,
            'calibration_duration': calibration_duration,
            'calibration_approach': 'explicit_duration' if calibration_duration else 'automatic_inline',
            'error': str(e),
            'correction_applied': False,
            'parameters': {
                'cutoff': cutoff,
                'method': method,
                'blocksize': blocksize,
                'calibration_duration': calibration_duration
            }
        }
        
        return error_raw


def _plot_asr_comparison(
        original_raw: BaseRaw, 
        cleaned_raw: BaseRaw, 
        start_time: float, 
        duration: float
) -> None:
    """
    Plot before/after comparison of ASR correction.
    
    Args:
        original_raw: Original raw data
        cleaned_raw: ASR-corrected raw data  
        start_time: Start time for plot (seconds)
        duration: Duration to plot (seconds)
    """
    import matplotlib.pyplot as plt
    
    # Select a subset of EEG channels for plotting (max 6 for clarity)
    eeg_picks = pick_types(original_raw.info, eeg=True, meg=False)[:6]
    
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found for ASR comparison plot")
        return
    
    # Extract data for plotting
    start_idx = int(start_time * original_raw.info['sfreq'])
    end_idx = start_idx + int(duration * original_raw.info['sfreq'])
    
    if end_idx > original_raw.n_times:
        end_idx = original_raw.n_times
        logger.warning(f"Plot duration adjusted to fit data length")
    
    time = original_raw.times[start_idx:end_idx]
    
    original_data = original_raw.get_data(picks=eeg_picks, start=start_idx, stop=end_idx) * 1e6  # µV
    cleaned_data = cleaned_raw.get_data(picks=eeg_picks, start=start_idx, stop=end_idx) * 1e6  # µV
    
    # Create comparison plot
    n_channels = len(eeg_picks)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    # Channel names for plotting
    ch_names = [original_raw.ch_names[pick] for pick in eeg_picks]
    
    for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        # Plot original in blue
        ax.plot(time, original_data[i], 'b-', alpha=0.7, linewidth=1, label='Original')
        
        # Plot cleaned in red
        ax.plot(time, cleaned_data[i], 'r-', alpha=0.8, linewidth=1, label='ASR Corrected')
        
        # Calculate and display correlation
        correlation = np.corrcoef(original_data[i], cleaned_data[i])[0, 1]
        
        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{ch_name} - Correlation: {correlation:.3f}', fontsize=10)
        
        if i == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    
    # Overall title with statistics
    if hasattr(cleaned_raw, '_asr_metrics'):
        metrics = cleaned_raw._asr_metrics
        cutoff = metrics['cutoff']
        mean_corr = metrics['mean_correlation']
        var_change = metrics['variance_change_percent']
        
        fig.suptitle(
            f'ASR Artifact Correction (Cutoff: {cutoff}, Mean Correlation: {mean_corr:.3f}, '
            f'Variance Change: {var_change:+.1f}%)',
            fontsize=12, y=0.98
        )
    else:
        fig.suptitle('ASR Artifact Correction - Before vs After', fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def get_asr_quality_summary(raw: BaseRaw) -> Dict:
    """
    Extract ASR quality summary from processed raw data.
    
    Args:
        raw: Processed raw data with _asr_metrics attribute
        
    Returns:
        Dictionary with ASR quality metrics
    """
    if hasattr(raw, '_asr_metrics'):
        return raw._asr_metrics.copy()
    else:
        # Fallback for raw data without ASR metrics
        return {
            'cutoff': None,
            'method': 'unknown',
            'correction_applied': False,
            'mean_correlation': None,
            'variance_change_percent': None
        }


# Legacy wrapper for backward compatibility
def remove_artifacts_asr(raw: BaseRaw, **kwargs) -> BaseRaw:
    """
    Legacy wrapper for clean_rawdata_asr.
    
    Deprecated: Use clean_rawdata_asr instead.
    ASR should be used as an intermediate cleaning step, not as an alternative to ICA.
    """
    logger.warning("remove_artifacts_asr is deprecated. Use clean_rawdata_asr instead.")
    logger.warning("ASR should be applied as an intermediate cleaning step after bad channel detection but before ICA.")
    return clean_rawdata_asr(raw, **kwargs)

