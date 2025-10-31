"""
Apply baseline correction to existing Epochs.

This module provides functionality to re-apply baseline correction to Epochs
that have already been created. This is particularly useful after applying
artifact correction methods like EOG regression, where MNE-Python recommends
re-baselining the data.

Recommended workflow:
    Epoch → Regression → Baseline → Average
"""

from loguru import logger
from typing import Optional, Tuple, Union
from mne import BaseEpochs


def apply_baseline_correction(
    epochs: BaseEpochs,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0),
    verbose: Optional[Union[bool, str]] = None
) -> BaseEpochs:
    """
    Apply baseline correction to existing Epochs.

    This function wraps MNE's Epochs.apply_baseline() method and should be used
    to re-apply baseline correction after artifact removal (e.g., EOG regression).

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs object to baseline correct.
    baseline : None | tuple of length 2, default=(None, 0)
        The time interval to apply as baseline. If None, no baseline correction
        is applied. If a tuple (a, b), the interval is between a and b (in seconds),
        including the endpoints.
        - If a is None, the beginning of the data is used
        - If b is None, the end of the data is used
        - If (None, None), the entire time interval is used
        Default is (None, 0), i.e., from beginning until time point zero.
    verbose : bool | str | None, default=None
        Control verbosity of the logging output.

    Returns
    -------
    epochs : mne.BaseEpochs
        The baseline-corrected Epochs object (modified in-place but also returned).

    Notes
    -----
    - Baseline correction calculates the mean signal during the baseline period
      and subtracts it from the entire epoch for each channel independently.
    - This operation is applied in-place to the Epochs object.
    - Baseline correction can be applied multiple times, but cannot be reverted
      once the data has been loaded.
    - After artifact removal (e.g., EOG regression), re-applying baseline
      correction is recommended per MNE-Python documentation.

    Examples
    --------
    >>> # Basic usage with default baseline (beginning to time 0)
    >>> epochs_clean = apply_baseline_correction(epochs)

    >>> # Custom baseline period from -0.2 to 0 seconds
    >>> epochs_clean = apply_baseline_correction(epochs, baseline=(-0.2, 0))

    >>> # No baseline correction
    >>> epochs_no_baseline = apply_baseline_correction(epochs, baseline=None)

    References
    ----------
    MNE-Python documentation recommends re-applying baseline after regression:
    https://mne.tools/stable/auto_tutorials/preprocessing/35_artifact_correction_regression.html
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

    if baseline is None:
        logger.info("baseline=None: Skipping baseline correction")
        return epochs

    # Validate baseline tuple/list and convert to tuple if needed
    if isinstance(baseline, (list, tuple)):
        if len(baseline) != 2:
            raise ValueError(
                f"baseline must be None or a tuple/list of length 2, got {baseline}"
            )
        baseline = tuple(baseline)  # Convert list to tuple for consistency
    else:
        raise ValueError(
            f"baseline must be None, tuple, or list of length 2, got {type(baseline).__name__}: {baseline}"
        )

    # Log baseline correction details
    tmin, tmax = baseline
    if tmin is None and tmax is None:
        logger.info("Applying baseline correction using entire epoch")
    elif tmin is None:
        logger.info(f"Applying baseline correction from start to {tmax}s")
    elif tmax is None:
        logger.info(f"Applying baseline correction from {tmin}s to end")
    else:
        logger.info(f"Applying baseline correction from {tmin}s to {tmax}s")

    # Apply baseline correction (modifies in-place)
    try:
        epochs.apply_baseline(baseline=baseline, verbose=verbose)
        logger.success(
            f"Baseline correction applied successfully to {len(epochs)} epochs"
        )
    except Exception as e:
        logger.error(f"Failed to apply baseline correction: {e}")
        raise

    return epochs
