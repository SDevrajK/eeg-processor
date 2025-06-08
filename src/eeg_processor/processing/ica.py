"""
ICA-based artifact removal for EEG data.

This module provides comprehensive ICA artifact removal with automatic component
classification using ICALabel, ARCI heartbeat detection, and correlation-based detection methods.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import ICA
import matplotlib.pyplot as plt


def remove_artifacts_ica(
        raw: BaseRaw,
        n_components: Union[float, int] = 15,
        method: str = 'infomax',
        eog_channels: Optional[List[str]] = ['VEOG', 'HEOG'],
        ecg_channels: Optional[List[str]] = None,
        auto_classify: bool = False,
        use_arci: bool = False,
        muscle_threshold: float = 0.8,
        eye_threshold: float = 0.8,
        heart_threshold: float = 0.8,
        line_noise_threshold: float = 0.8,
        arci_cardiac_freq_range: Tuple[float, float] = (0.6, 1.7),
        arci_regularity_threshold: float = 0.4,
        plot_components: bool = False,
        enable_manual: bool = False,
        decim: Optional[int] = None,
        random_state: int = 42,
        inplace: bool = False,
        verbose: Optional[Union[bool, str]] = None
) -> BaseRaw:
    """
    Remove artifacts using Independent Component Analysis (ICA).

    Combines automatic classification (ICALabel) with correlation-based detection
    to identify and remove eye blinks, muscle artifacts, cardiac artifacts, and
    line noise components.

    Args:
        raw: Input raw EEG data
        n_components: Number of ICA components (15-20 recommended for 32-ch systems)
        method: ICA algorithm ('infomax', 'fastica', 'picard')
        eog_channels: EOG channel names for blink detection
        ecg_channels: ECG channel names for cardiac detection
        auto_classify: Enable ICALabel automatic classification
        use_arci: Enable ARCI heartbeat detection (alternative to ICLabel heart detection)
        muscle_threshold: Classification confidence threshold for muscle artifacts
        eye_threshold: Classification confidence threshold for eye artifacts
        heart_threshold: Classification confidence threshold for cardiac artifacts
        line_noise_threshold: Classification confidence threshold for line noise
        arci_cardiac_freq_range: ARCI cardiac frequency range in Hz (default: 0.6-1.7)
        arci_regularity_threshold: ARCI regularity threshold for heart rhythm (default: 0.4)
        plot_components: Show component analysis plots
        enable_manual: Allow manual component selection interface
        decim: Decimation factor for ICA fitting (speeds up computation)
        random_state: Random seed for reproducibility
        inplace: Ignored (ICA always creates new object)
        verbose: Verbosity level

    Returns:
        Raw object with artifacts removed

    Notes:
        - Stores detailed metrics in raw._ica_metrics for quality tracking
        - Combines multiple detection methods for robust artifact identification
        - Supports both automatic and manual component selection
        - ARCI method requires proper filtering (notch + 0.1-30 Hz recommended)
        - Prints detailed ICLabel probability table for all components (if auto_classify=True)
    """
    if inplace:
        logger.info("inplace=True ignored for ICA - always creates new object")

    logger.info(f"Starting ICA artifact removal (n_components={n_components}, method={method})")

    # Fit ICA
    ica = _fit_ica(raw, n_components, method, decim, random_state, verbose)

    # Detect artifact components using multiple methods
    detection_results = _detect_artifact_components(
        raw, ica, eog_channels, ecg_channels, auto_classify, use_arci,
        muscle_threshold, eye_threshold, heart_threshold, line_noise_threshold,
        arci_cardiac_freq_range, arci_regularity_threshold
    )

    # Determine final components to exclude
    final_excludes = _determine_final_excludes(detection_results)

    # Manual override if requested
    if enable_manual and plot_components:
        final_excludes = _manual_component_selection(raw, ica, final_excludes)

    # Visualization
    if plot_components:
        _plot_ica_analysis(raw, ica, final_excludes, detection_results)

    # Apply ICA cleaning
    logger.info(f"Applying ICA with {len(final_excludes)} excluded components: {sorted(final_excludes)}")
    cleaned_raw = ica.apply(raw, exclude=final_excludes)

    # Store comprehensive metrics for quality tracking
    cleaned_raw._ica_metrics = {
        'n_components_fitted': ica.n_components_,
        'n_components_excluded': len(final_excludes),
        'excluded_components': sorted(final_excludes),
        'detection_methods': {
            'icalabel_muscle': detection_results.get('icalabel_muscle', []),
            'icalabel_eye': detection_results.get('icalabel_eye', []),
            'icalabel_heart': detection_results.get('icalabel_heart', []),
            'icalabel_line': detection_results.get('icalabel_line', []),
            'icalabel_noise': detection_results.get('icalabel_noise', []),
            'eog_correlation': detection_results.get('eog_detection', []),
            'ecg_correlation': detection_results.get('ecg_detection', []),
            'arci_heartbeat': detection_results.get('arci_heartbeat', [])
        },
        'explained_variance': getattr(ica, 'pca_explained_variance_', None),
        'method': method,
        'parameters': {
            'n_components': n_components,
            'muscle_threshold': muscle_threshold,
            'eye_threshold': eye_threshold,
            'heart_threshold': heart_threshold,
            'use_arci': use_arci,
            'arci_params': {
                'cardiac_freq_range': arci_cardiac_freq_range,
                'regularity_threshold': arci_regularity_threshold
            } if use_arci else None,
            'random_state': random_state
        }
    }

    logger.success(f"ICA cleaning completed. Excluded {len(final_excludes)} components.")
    return cleaned_raw


def _fit_ica(
        raw: BaseRaw,
        n_components: Union[float, int],
        method: str,
        decim: Optional[int],
        random_state: int,
        verbose: Optional[Union[bool, str]]
) -> ICA:
    """Fit ICA to the raw data."""
    ica = ICA(
        n_components=n_components,
        method=method,
        fit_params=dict(extended=True) if method == 'infomax' else dict(),
        random_state=random_state,
        max_iter=500,
        verbose=verbose
    )

    logger.info("Fitting ICA...")
    ica.fit(raw, decim=decim)
    logger.success(f"ICA fitted with {ica.n_components_} components")

    return ica


def _detect_artifact_components(
        raw: BaseRaw,
        ica: ICA,
        eog_channels: Optional[List[str]],
        ecg_channels: Optional[List[str]],
        auto_classify: bool,
        use_arci: bool,
        muscle_threshold: float,
        eye_threshold: float,
        heart_threshold: float,
        line_noise_threshold: float,
        arci_cardiac_freq_range: Tuple[float, float],
        arci_regularity_threshold: float
) -> Dict[str, List[int]]:
    """
    Detect artifact components using multiple methods.
    """
    detection_results = {
        'icalabel_muscle': [],
        'icalabel_eye': [],
        'icalabel_heart': [],
        'icalabel_line': [],
        'icalabel_noise': [],
        'eog_detection': [],
        'ecg_detection': [],
        'arci_heartbeat': []  # NEW: ARCI detection results
    }

    # ICALabel automatic classification
    if auto_classify:
        try:
            detection_results.update(_run_icalabel_classification(
                raw, ica, muscle_threshold, eye_threshold, heart_threshold, line_noise_threshold
            ))
        except Exception as e:
            logger.error(f"ICALabel classification failed: {e}")

    # EOG-based detection
    if eog_channels:
        detection_results['eog_detection'] = _detect_eog_components(raw, ica, eog_channels)

    # ECG-based detection
    if ecg_channels:
        detection_results['ecg_detection'] = _detect_ecg_components(raw, ica, ecg_channels)

    # ARCI heartbeat detection
    if use_arci:
        try:
            # Import ARCI function (you'll need to save the ARCI code in a separate file or add it here)
            arci_components, arci_detailed = detect_heartbeat_components_arci(
                ica, raw,
                cardiac_freq_range=arci_cardiac_freq_range,
                regularity_threshold=arci_regularity_threshold,
                verbose=True
            )
            detection_results['arci_heartbeat'] = arci_components
            logger.info(f"ARCI heartbeat detection: {arci_components}")
        except NameError:
            logger.warning("ARCI function not available. Please ensure detect_heartbeat_components_arci is imported.")
        except Exception as e:
            logger.error(f"ARCI heartbeat detection failed: {e}")

    # Log detection summary
    _log_detection_summary(detection_results)

    return detection_results


def _run_icalabel_classification(
        raw: BaseRaw,
        ica: ICA,
        muscle_threshold: float,
        eye_threshold: float,
        heart_threshold: float,
        line_noise_threshold: float
) -> Dict[str, List[int]]:
    """Run ICALabel automatic component classification."""
    from mne_icalabel import label_components
    from mne_icalabel.iclabel import iclabel_label_components

    logger.info("Running ICALabel classification...")

    # Get FULL probability matrix (n_components, 7_classes)
    full_probabilities = iclabel_label_components(raw, ica)

    # Get standard output for compatibility
    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels['labels']

    # Print formatted classification table with FULL probabilities
    _print_icalabel_table(labels, full_probabilities)

    classification_results = {
        'icalabel_muscle': [],
        'icalabel_eye': [],
        'icalabel_heart': [],
        'icalabel_line': [],
        'icalabel_noise': []
    }

    # Classification mapping using full probabilities
    # Categories: ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
    thresholds = {
        1: ('icalabel_muscle', muscle_threshold),      # muscle artifact
        2: ('icalabel_eye', eye_threshold),           # eye blink
        3: ('icalabel_heart', heart_threshold),       # heart beat
        4: ('icalabel_line', line_noise_threshold),   # line noise
        5: ('icalabel_noise', 0.8)                    # channel noise
    }

    for i, (label, probs) in enumerate(zip(labels, full_probabilities)):
        # Check each artifact category
        for class_idx, (result_key, threshold) in thresholds.items():
            if probs[class_idx] >= threshold:
                classification_results[result_key].append(i)
                logger.info(f"ICALabel {result_key.replace('icalabel_', '')}: ICA{i:03d} (confidence: {probs[class_idx]:.2f})")

    # Store full classification results on ICA object
    ica.labels_ = ic_labels
    ica.labels_probabilities_ = full_probabilities  # Store full matrix for later use

    return classification_results


def _print_icalabel_table(labels: List[str], full_probabilities: np.ndarray) -> None:
    """Print a formatted table showing ICLabel classification results with full probability matrix."""

    # ICLabel category order and short names
    categories = ['Brain', 'Muscle', 'Eye', 'Heart', 'Line', 'Noise', 'Other']

    print("\n" + "="*85)
    print("ICLabel Component Classification Results (Full Probability Matrix)")
    print("="*85)

    # Header
    header = f"{'Comp':>4} {'Label':>12} {'Conf':>5} "
    for cat in categories:
        header += f"{cat:>7}"
    print(header)
    print("-"*85)

    # Component rows
    for i, (label, probs) in enumerate(zip(labels, full_probabilities)):
        # Get confidence (max probability for this component)
        max_prob = np.max(probs)

        # Clean up label for display
        display_label = label.replace(' artifact', '').replace(' beat', '').replace(' blink', '').replace(' noise', '')

        # Format row
        row = f"ICA{i:03d} {display_label:>12} {max_prob:>5.2f} "

        # Add all 7 probability values
        for prob in probs:
            row += f"{prob:>7.2f}"

        print(row)

    print("="*85)

    # Summary statistics
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nClassification Summary:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} components")

    # Identify potential issues with heart components specifically
    heart_analysis = []
    low_confidence_brain = []
    very_low_confidence = []

    for i, (label, probs) in enumerate(zip(labels, full_probabilities)):
        max_prob = np.max(probs)
        heart_prob = probs[3]  # Heart is index 3

        # Flag components with notable heart probability
        if heart_prob > 0.1:  # Any significant heart probability
            heart_analysis.append(f"ICA{i:03d} (heart: {heart_prob:.3f}, {label}: {max_prob:.3f})")

        # Flag low confidence brain components (potential single-electrode artifacts)
        if label == 'brain' and max_prob < 0.9:
            low_confidence_brain.append(f"ICA{i:03d} ({max_prob:.3f})")

        # Flag very low confidence overall
        if max_prob < 0.6:
            very_low_confidence.append(f"ICA{i:03d} ({label}, {max_prob:.3f})")

    if heart_analysis:
        print(f"\n💓 Components with notable heart probability:")
        for item in heart_analysis:
            print(f"   {item}")

    if low_confidence_brain:
        print(f"\n⚠️  Low-confidence 'brain' components (check for single-electrode artifacts):")
        print(f"   {', '.join(low_confidence_brain)}")

    if very_low_confidence:
        print(f"\n⚠️  Very low confidence components (potential noise):")
        print(f"   {', '.join(very_low_confidence)}")

    print()


def _detect_eog_components(raw: BaseRaw, ica: ICA, eog_channels: List[str]) -> List[int]:
    """Detect eye artifact components using EOG correlation."""
    eog_components = []

    logger.info("Detecting EOG-correlated components...")

    for eog_ch in eog_channels:
        if eog_ch in raw.ch_names:
            try:
                components, scores = ica.find_bads_eog(raw, eog_ch, verbose=False)
                eog_components.extend(components)
                logger.info(f"EOG detection ({eog_ch}): components {components}")
            except Exception as e:
                logger.warning(f"EOG detection failed for {eog_ch}: {e}")

    return list(set(eog_components))  # Remove duplicates


def _detect_ecg_components(raw: BaseRaw, ica: ICA, ecg_channels: List[str]) -> List[int]:
    """Detect cardiac artifact components using ECG correlation."""
    ecg_components = []

    logger.info("Detecting ECG-correlated components...")

    for ecg_ch in ecg_channels:
        if ecg_ch in raw.ch_names:
            try:
                components, scores = ica.find_bads_ecg(raw, ecg_ch, verbose=False)
                ecg_components.extend(components)
                logger.info(f"ECG detection ({ecg_ch}): components {components}")
            except Exception as e:
                logger.warning(f"ECG detection failed for {ecg_ch}: {e}")

    return list(set(ecg_components))


# ARCI function
def detect_heartbeat_components_arci(
    ica: ICA,
    raw: BaseRaw,
    cardiac_freq_range: Tuple[float, float] = (0.6, 1.7),
    regularity_threshold: float = 0.4,
    min_peaks: int = 10,
    peak_height_std: float = 1.5,
    verbose: bool = True
) -> Tuple[List[int], Dict]:
    """
    Detect heartbeat components using the ARCI method.
    Based on the published ARCI algorithm from Frontiers in Neuroscience, 2019.
    """
    from scipy import signal

    logger.info("Starting ARCI heartbeat component detection...")

    heartbeat_components = []
    sources = ica.get_sources(raw)
    sfreq = raw.info['sfreq']

    if verbose:
        logger.info(f"Analyzing {ica.n_components_} components")
        logger.info(f"Cardiac frequency range: {cardiac_freq_range[0]:.1f}-{cardiac_freq_range[1]:.1f} Hz")
        logger.info(f"Corresponding BPM range: {cardiac_freq_range[0]*60:.0f}-{cardiac_freq_range[1]*60:.0f}")

    detailed_results = {}

    for i in range(ica.n_components_):
        component_data = sources.get_data()[i]

        # Stage 1: Frequency domain analysis
        freqs, psd = signal.welch(
            component_data,
            sfreq,
            nperseg=int(min(sfreq*4, len(component_data)//4))
        )

        # Exclude edge effects as in ARCI paper
        valid_mask = (freqs >= 0.4) & (freqs <= 99.0)
        freqs_valid = freqs[valid_mask]
        psd_valid = psd[valid_mask]

        if len(psd_valid) == 0:
            continue

        # Find maximum power peak
        max_peak_idx = np.argmax(psd_valid)
        max_peak_freq = freqs_valid[max_peak_idx]
        max_peak_power = psd_valid[max_peak_idx]
        total_power = np.sum(psd_valid)
        power_ratio = max_peak_power / total_power if total_power > 0 else 0

        # Store detailed analysis
        detailed_results[i] = {
            'max_freq': max_peak_freq,
            'max_power_ratio': power_ratio,
            'in_cardiac_range': False,
            'peak_count': 0,
            'regularity_cv': np.inf,
            'mean_bpm': 0,
            'is_heartbeat': False
        }

        # Stage 2: Check if peak frequency is in cardiac range
        in_cardiac_range = cardiac_freq_range[0] <= max_peak_freq <= cardiac_freq_range[1]
        detailed_results[i]['in_cardiac_range'] = in_cardiac_range

        if in_cardiac_range:
            if verbose:
                logger.debug(f"ICA{i:03d}: Peak at {max_peak_freq:.3f} Hz (cardiac range)")

            # Stage 3: Time domain validation - Peak detection for regularity
            abs_component = np.abs(component_data)
            threshold = np.mean(abs_component) + peak_height_std * np.std(abs_component)

            # Find peaks with minimum distance constraint
            min_distance = int(0.5 * sfreq)  # 0.5 seconds minimum (120 BPM max)
            peaks, properties = signal.find_peaks(
                abs_component,
                height=threshold,
                distance=min_distance
            )

            detailed_results[i]['peak_count'] = len(peaks)

            if len(peaks) >= min_peaks:
                # Calculate inter-peak intervals
                intervals = np.diff(peaks) / sfreq  # Convert to seconds

                if len(intervals) > 0:
                    # Check regularity using coefficient of variation
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    cv = std_interval / mean_interval if mean_interval > 0 else np.inf
                    mean_bpm = 60 / mean_interval if mean_interval > 0 else 0

                    detailed_results[i]['regularity_cv'] = cv
                    detailed_results[i]['mean_bpm'] = mean_bpm

                    # ARCI criteria: regular rhythm within valid heart rate range
                    if (cv < regularity_threshold and
                        cardiac_freq_range[0]*60 <= mean_bpm <= cardiac_freq_range[1]*60):

                        heartbeat_components.append(i)
                        detailed_results[i]['is_heartbeat'] = True

                        if verbose:
                            logger.success(f"ICA{i:03d}: HEARTBEAT DETECTED!")
                            logger.info(f"  Peak frequency: {max_peak_freq:.3f} Hz")
                            logger.info(f"  Mean BPM: {mean_bpm:.1f}")
                            logger.info(f"  Regularity CV: {cv:.3f}")
                            logger.info(f"  Peak count: {len(peaks)}")
                    elif verbose:
                        logger.debug(f"ICA{i:03d}: In cardiac range but failed regularity test")
                        logger.debug(f"  CV: {cv:.3f} (threshold: {regularity_threshold})")
                        logger.debug(f"  BPM: {mean_bpm:.1f}")
            elif verbose:
                logger.debug(f"ICA{i:03d}: Insufficient peaks ({len(peaks)} < {min_peaks})")

    # Summary logging
    if verbose:
        logger.info(f"ARCI Detection Summary:")
        logger.info(f"  Total components analyzed: {ica.n_components_}")
        logger.info(f"  Components in cardiac frequency range: {sum(1 for r in detailed_results.values() if r['in_cardiac_range'])}")
        logger.info(f"  Heartbeat components detected: {len(heartbeat_components)}")
        if heartbeat_components:
            logger.success(f"  Detected heartbeat components: {heartbeat_components}")
        else:
            logger.warning("  No heartbeat components detected")

    return heartbeat_components, detailed_results


def _determine_final_excludes(detection_results: Dict[str, List[int]]) -> List[int]:
    """Combine detection results to determine final components to exclude."""
    all_components = []

    for method, components in detection_results.items():
        all_components.extend(components)

    return sorted(list(set(all_components)))


def _manual_component_selection(raw: BaseRaw, ica: ICA, auto_excludes: List[int]) -> List[int]:
    """Allow manual component selection through interactive interface."""
    logger.info("Opening manual component selection interface...")

    # Show all components
    fig = ica.plot_components()

    # Show sources for interactive selection
    ica.plot_sources(raw, title="Click components to exclude. Close window when done.", block=True)

    # Get manual selections
    manual_excludes = list(ica.exclude)

    # Combine automatic and manual selections
    combined_excludes = sorted(list(set(auto_excludes + manual_excludes)))

    manual_additions = set(manual_excludes) - set(auto_excludes)
    if manual_additions:
        logger.info(f"Manual additions: {sorted(manual_additions)}")

    plt.close('all')  # Clean up plots

    return combined_excludes


def _plot_ica_analysis(
        raw: BaseRaw,
        ica: ICA,
        final_excludes: List[int],
        detection_results: Dict[str, List[int]]
) -> None:
    """Create comprehensive ICA analysis plots."""
    if not final_excludes:
        logger.info("No components to exclude - skipping detailed plots")
        return

    logger.info("Generating ICA analysis plots...")

    # Plot component properties for excluded components
    if len(final_excludes) <= 10:  # Avoid overwhelming plots
        ica.plot_properties(
            raw,
            picks=final_excludes,
            psd_args={'fmax': 50},
            figsize=(8, 6)
        )

    # Plot all components overview
    ica.plot_components(picks=range(min(20, ica.n_components_)))

    # Plot before/after overlay
    ica.plot_overlay(raw, exclude=final_excludes, picks='eeg')

    # Plot sources
    ica.plot_sources(raw, stop=10.0, title="ICA Sources (first 10 seconds)", block=True)


def _log_detection_summary(detection_results: Dict[str, List[int]]) -> None:
    """Log comprehensive detection summary."""
    logger.info("=== Component Detection Summary ===")

    for method, components in detection_results.items():
        if components:
            logger.info(f"  {method}: {sorted(components)}")
        else:
            logger.debug(f"  {method}: none detected")

    all_detected = set()
    for components in detection_results.values():
        all_detected.update(components)

    logger.info(f"  Total unique components: {sorted(all_detected)}")


def get_ica_quality_summary(raw: BaseRaw) -> Dict:
    """
    Extract ICA quality summary from processed raw data.

    Args:
        raw: Processed raw data with _ica_metrics attribute

    Returns:
        Dictionary with ICA quality metrics
    """
    if hasattr(raw, '_ica_metrics'):
        return raw._ica_metrics.copy()
    else:
        # Fallback for raw data without detailed metrics
        return {
            'n_components_excluded': 0,
            'excluded_components': [],
            'method': 'unknown'
        }


# Legacy wrapper for backward compatibility
def remove_blinks_with_ica(raw: BaseRaw, **kwargs) -> BaseRaw:
    """
    Legacy wrapper for remove_artifacts_ica.

    Deprecated: Use remove_artifacts_ica instead.
    """
    logger.warning("remove_blinks_with_ica is deprecated. Use remove_artifacts_ica instead.")

    # Map old parameters to new ones
    if 'eog_ch' in kwargs:
        kwargs['eog_channels'] = kwargs.pop('eog_ch')
        if isinstance(kwargs['eog_channels'], str):
            kwargs['eog_channels'] = [kwargs['eog_channels']]

    if 'ecg_ch' in kwargs:
        kwargs['ecg_channels'] = kwargs.pop('ecg_ch')
        if isinstance(kwargs['ecg_channels'], str):
            kwargs['ecg_channels'] = [kwargs['ecg_channels']]

    return remove_artifacts_ica(raw, **kwargs)