"""
ICA-based artifact removal for EEG data.

This module provides comprehensive ICA artifact removal with automatic component
classification using ICALabel and correlation-based detection methods.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne.preprocessing import ICA
import matplotlib.pyplot as plt


def remove_artifacts_ica(
        raw: BaseRaw,
        n_components: Union[float, int] = 0.99,
        method: str = 'infomax',
        eog_channels: Optional[List[str]] = ['VEOG', 'HEOG'],
        ecg_channels: Optional[List[str]] = None,
        auto_classify: bool = True,
        muscle_threshold: float = 0.8,
        eye_threshold: float = 0.8,
        heart_threshold: float = 0.8,
        line_noise_threshold: float = 0.8,
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
        n_components: Number of ICA components (float for explained variance, int for exact number)
        method: ICA algorithm ('infomax', 'fastica', 'picard')
        eog_channels: EOG channel names for blink detection
        ecg_channels: ECG channel names for cardiac detection
        auto_classify: Enable ICALabel automatic classification
        muscle_threshold: Classification confidence threshold for muscle artifacts
        eye_threshold: Classification confidence threshold for eye artifacts
        heart_threshold: Classification confidence threshold for cardiac artifacts
        line_noise_threshold: Classification confidence threshold for line noise
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
    """
    if inplace:
        logger.info("inplace=True ignored for ICA - always creates new object")

    logger.info(f"Starting ICA artifact removal (n_components={n_components}, method={method})")

    # Fit ICA
    ica = _fit_ica(raw, n_components, method, decim, random_state, verbose)

    # Detect artifact components using multiple methods
    detection_results = _detect_artifact_components(
        raw, ica, eog_channels, ecg_channels, auto_classify,
        muscle_threshold, eye_threshold, heart_threshold, line_noise_threshold
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
            'ecg_correlation': detection_results.get('ecg_detection', [])
        },
        'explained_variance': getattr(ica, 'pca_explained_variance_', None),
        'method': method,
        'parameters': {
            'n_components': n_components,
            'muscle_threshold': muscle_threshold,
            'eye_threshold': eye_threshold,
            'heart_threshold': heart_threshold,
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
        max_iter=1000,
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
        muscle_threshold: float,
        eye_threshold: float,
        heart_threshold: float,
        line_noise_threshold: float
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
        'ecg_detection': []
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

    logger.info("Running ICALabel classification...")
    ic_labels = label_components(raw, ica, method='iclabel')

    labels = ic_labels['labels']
    probabilities = ic_labels['y_pred_proba']

    classification_results = {
        'icalabel_muscle': [],
        'icalabel_eye': [],
        'icalabel_heart': [],
        'icalabel_line': [],
        'icalabel_noise': []
    }

    # Classification mapping
    thresholds = {
        'muscle artifact': ('icalabel_muscle', muscle_threshold),
        'eye blink': ('icalabel_eye', eye_threshold),
        'heart beat': ('icalabel_heart', heart_threshold),
        'line noise': ('icalabel_line', line_noise_threshold),
        'channel noise': ('icalabel_noise', 0.8)
    }

    for i, (label, probs) in enumerate(zip(labels, probabilities)):
        max_prob = np.max(probs)

        if label in thresholds:
            result_key, threshold = thresholds[label]
            if max_prob >= threshold:
                classification_results[result_key].append(i)
                logger.info(f"ICALabel {label}: ICA{i:03d} (confidence: {max_prob:.2f})")

    # Store full classification results on ICA object
    ica.labels_ = ic_labels

    return classification_results


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
    ica.plot_sources(raw, stop=10.0, title="ICA Sources (first 10 seconds)")


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