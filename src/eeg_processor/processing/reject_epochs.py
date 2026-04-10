"""
Apply artifact rejection to existing Epochs.

This module provides functionality to reject epochs based on amplitude criteria
after epochs have been created. This is particularly useful after applying
artifact correction methods like EOG regression, where you want to re-apply
rejection criteria.

Recommended workflow:
    Epoch → Regression → Baseline → Rejection → Average
"""

from loguru import logger
from typing import Optional, Dict, Union
from mne import BaseEpochs
import numpy as np


def reject_bad_epochs(
    epochs: BaseEpochs,
    reject: Optional[Dict[str, float]] = None,
    flat: Optional[Dict[str, float]] = None,
    check_gradient: bool = True,
    gradient_threshold: Optional[float] = None,
    verbose: Optional[Union[bool, str]] = None
) -> BaseEpochs:
    """
    Apply artifact rejection to existing Epochs.

    This function wraps MNE's Epochs.drop_bad() method and should be used to
    re-apply artifact rejection after artifact removal (e.g., EOG regression).

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs object to apply rejection to.
    reject : dict | None, default=None
        Rejection parameters based on peak-to-peak (PTP) amplitude.
        Keys are channel types, values are amplitude thresholds in V.
        Example: {'eeg': 100e-6} rejects epochs where PTP > 100 µV.

        BrainVision Analyzer equivalent:
        - "Maximal allowed difference": reject={'eeg': 200e-6}

        If None, uses rejection parameters set during epoch creation.
    flat : dict | None, default=None
        Rejection parameters based on minimum PTP amplitude.
        Keys are channel types, values are minimum amplitude thresholds in V.
        Example: {'eeg': 0.5e-6} rejects epochs where PTP < 0.5 µV.

        BrainVision Analyzer equivalent:
        - "Lowest allowed activity": flat={'eeg': 0.5e-6}

        If None, uses flat parameters set during epoch creation.
    check_gradient : bool, default=False
        If True, apply gradient-based rejection (EEG channels only).

        BrainVision Analyzer equivalent:
        - "Maximal allowed voltage step": 50 µV/ms

        This catches sharp transients like electrode pops and disconnections
        that may not exceed peak-to-peak amplitude thresholds.
    gradient_threshold : float | None, default=None
        Maximum allowed voltage step between consecutive samples in µV/ms.
        Only used if check_gradient=True.
        If None, uses default of 50 µV/ms.
    verbose : bool | str | None, default=None
        Control verbosity of the logging output.

    Returns
    -------
    epochs : mne.BaseEpochs
        The Epochs object with bad epochs marked for rejection.

    Notes
    -----
    - Rejection is based on peak-to-peak (PTP) amplitude: max - min in each epoch
    - Bad epochs are marked but not removed until epochs.drop_bad() is called
    - This function calls drop_bad() internally, so bad epochs are removed
    - Gradient rejection runs before amplitude rejection when check_gradient=True
    - After rejection, re-applying baseline correction may be needed

    Examples
    --------
    >>> # Basic rejection with amplitude thresholds
    >>> epochs_clean = reject_bad_epochs(
    ...     epochs,
    ...     reject={'eeg': 100e-6},
    ...     flat={'eeg': 1e-6}
    ... )

    >>> # BrainVision Analyzer-compatible rejection
    >>> epochs_clean = reject_bad_epochs(
    ...     epochs,
    ...     reject={'eeg': 200e-6},  # Max difference: 200 µV
    ...     flat={'eeg': 0.5e-6}     # Min activity: 0.5 µV
    ... )

    >>> # Future: with gradient checking
    >>> epochs_clean = reject_bad_epochs(
    ...     epochs,
    ...     reject={'eeg': 200e-6},
    ...     flat={'eeg': 0.5e-6},
    ...     check_gradient=True,      # Enable gradient check
    ...     gradient_threshold=50.0   # 50 µV/ms
    ... )

    References
    ----------
    BrainVision Analyzer artifact rejection criteria:
    - Maximal allowed voltage step: 50 µV/ms (gradient check)
    - Maximal allowed difference: 200 µV (peak-to-peak amplitude)
    - Lowest allowed activity: 0.5 µV (minimum peak-to-peak)
    """
    if not isinstance(epochs, BaseEpochs):
        raise TypeError(
            f"Expected mne.BaseEpochs object, got {type(epochs).__name__}"
        )

    if not hasattr(epochs, 'preload') or not epochs.preload:
        raise ValueError(
            "Epochs data must be preloaded. Use epochs.load_data() or "
            "set preload=True when creating epochs."
        )

    # Apply gradient-based rejection before amplitude rejection
    if check_gradient:
        _check_gradient_rejection(
            epochs,
            gradient_threshold=gradient_threshold if gradient_threshold is not None else 50.0,
            verbose=verbose
        )

    # Validate and convert reject and flat parameters
    if reject is not None:
        if not isinstance(reject, dict):
            raise TypeError(
                f"reject must be a dict mapping channel types to thresholds, got {type(reject).__name__}. "
                f"Example: reject={{'eeg': 200e-6}}"
            )
        # Convert string values to floats (YAML may load "200e-6" as string)
        reject = {k: float(v) if isinstance(v, str) else v for k, v in reject.items()}

    if flat is not None:
        if not isinstance(flat, dict):
            raise TypeError(
                f"flat must be a dict mapping channel types to thresholds, got {type(flat).__name__}. "
                f"Example: flat={{'eeg': 0.5e-6}}"
            )
        # Convert string values to floats (YAML may load "0.5e-6" as string)
        flat = {k: float(v) if isinstance(v, str) else v for k, v in flat.items()}

    # Log rejection criteria
    n_epochs_before = len(epochs)

    if reject is None and flat is None:
        logger.info("Using existing rejection parameters from epoch creation")
        reject_str = "existing"
        flat_str = "existing"
    else:
        if reject is not None:
            reject_str = ", ".join([f"{k}: {v*1e6:.1f} µV" for k, v in reject.items()])
            logger.info(f"Reject thresholds (max PTP): {reject_str}")
        else:
            reject_str = "existing"

        if flat is not None:
            flat_str = ", ".join([f"{k}: {v*1e6:.1f} µV" for k, v in flat.items()])
            logger.info(f"Flat thresholds (min PTP): {flat_str}")
        else:
            flat_str = "existing"

    # Apply rejection
    try:
        epochs.drop_bad(
            reject=reject if reject is not None else 'existing',
            flat=flat if flat is not None else 'existing',
            verbose=verbose
        )

        n_epochs_after = len(epochs)
        n_rejected = n_epochs_before - n_epochs_after
        rejection_rate = (n_rejected / n_epochs_before * 100) if n_epochs_before > 0 else 0

        logger.success(
            f"Artifact rejection complete: {n_epochs_after}/{n_epochs_before} epochs retained "
            f"({n_rejected} rejected, {rejection_rate:.1f}%)"
        )

        if n_rejected > 0:
            logger.info(f"See epochs.drop_log for details on rejected epochs")

    except Exception as e:
        logger.error(f"Failed to apply artifact rejection: {e}")
        raise

    return epochs


def _check_gradient_rejection(
    epochs: BaseEpochs,
    gradient_threshold: float = 50.0,
    verbose: Optional[Union[bool, str]] = None
) -> BaseEpochs:
    """
    Apply gradient-based rejection to epochs (EEG channels only).

    Catches electrode pops and sharp transients that are physiologically
    impossible but may not exceed peak-to-peak amplitude thresholds.
    Equivalent to BrainVision Analyzer's "Maximal allowed voltage step" (50 µV/ms).

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs to check for gradient violations.
    gradient_threshold : float, default=50.0
        Maximum allowed voltage step between consecutive samples in µV/ms.
    verbose : bool | str | None
        Control verbosity.

    Returns
    -------
    epochs : mne.BaseEpochs
        Epochs with gradient-violating epochs dropped.
    """
    import mne
    eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False, stim=False)
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found for gradient rejection")
        return epochs

    # Shape: (n_epochs, n_eeg_channels, n_times)
    data = epochs.get_data(picks=eeg_picks)

    # Sample-to-sample differences scaled to µV/ms
    dt_ms = 1000.0 / epochs.info['sfreq']
    gradients = np.abs(np.diff(data, axis=2)) * 1e6 / dt_ms

    # Maximum gradient across all EEG channels and timepoints per epoch
    max_gradients = gradients.max(axis=(1, 2))

    bad_indices = np.where(max_gradients > gradient_threshold)[0].tolist()

    if bad_indices:
        epochs.drop(bad_indices, reason='GRADIENT')
        logger.info(
            f"Gradient rejection ({gradient_threshold} µV/ms): "
            f"dropped {len(bad_indices)} epochs"
        )
    else:
        logger.info(f"Gradient rejection ({gradient_threshold} µV/ms): no epochs exceeded threshold")

    return epochs
