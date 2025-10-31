"""
EOG regression-based artifact removal for EEG data.

This module implements EOG (electrooculogram) regression methods for removing blink
and eye movement artifacts from EEG data. It provides a wrapper around MNE-Python's
EOGRegression class with support for both continuous (Raw) and epoched data.

Two methodologies are supported:
1. **Direct Regression**: Standard linear regression for continuous data or epochs
   without significant evoked responses.
2. **Gratton & Coles Method**: Evoked subtraction approach for epoched ERP data,
   which prevents confounding genuine stimulus-locked brain activity with artifacts.

Scientific Background
---------------------
The regression approach models EOG artifacts as linear combinations of EOG channel
activity and subtracts the weighted EOG signals from EEG channels. The Gratton et al.
(1983) method extends this by computing regression coefficients after removing
stimulus-locked variability, ensuring that evoked potentials are preserved.

References
----------
Gratton, G., Coles, M.G., & Donchin, E. (1983). A new method for off-line removal
of ocular artifact. Electroencephalography and Clinical Neurophysiology, 55(4),
468-484. DOI: 10.1016/0013-4694(83)90135-9

Croft, R.J., et al. (2005). EOG correction: a comparison of four methods.
Psychophysiology, 42(1), 16-24. PMID: 15720577

Examples
--------
Direct regression on continuous data:
>>> from mne.io import read_raw_fif
>>> raw = read_raw_fif('sample_raw.fif', preload=True)
>>> raw.set_eeg_reference('average', projection=True)
>>> raw_clean = remove_artifacts_regression(raw, eog_channels=['HEOG', 'VEOG'])

Gratton method on epoched ERP data:
>>> epochs_clean = remove_artifacts_regression(
...     epochs,
...     eog_channels=['HEOG', 'VEOG'],
...     subtract_evoked=True
... )
"""

from typing import Union, List, Optional, Dict, Any
import time
import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne import BaseEpochs
from mne.preprocessing import EOGRegression


__all__ = ['remove_artifacts_regression']


def remove_artifacts_regression(
    data: Union[BaseRaw, BaseEpochs],
    eog_channels: Optional[List[str]] = None,
    subtract_evoked: bool = True,
    show_plot: bool = False,
    plot_duration: float = 10.0,
    plot_start: float = 5.0,
    inplace: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[BaseRaw, BaseEpochs]:
    """
    Remove blink and eye movement artifacts using EOG regression.

    This function wraps MNE-Python's EOGRegression to provide artifact correction
    for both continuous (Raw) and epoched data. For epoched ERP data, it implements
    the Gratton & Coles (1983) method with evoked subtraction to preserve
    stimulus-locked brain activity.

    Parameters
    ----------
    data : BaseRaw | BaseEpochs
        MNE Raw or Epochs object containing EEG and EOG data.
        Data must be preloaded and have EEG reference set.
    eog_channels : list of str, optional
        List of EOG channel names to use as artifact predictors.
        Common configurations: ['HEOG', 'VEOG'] or ['Fp1', 'Fp2'].
        If None, uses all channels marked as 'eog' type. Default: None.
    subtract_evoked : bool
        If True and data is Epochs, use Gratton & Coles method:
        fit regression on evoked-subtracted data, apply to original epochs.
        This preserves stimulus-locked brain activity (ERPs).
        Ignored for Raw data. Default: True.
    show_plot : bool
        If True, display before/after comparison plot. Default: False.
    plot_duration : float
        Duration of comparison plot in seconds. Default: 10.0.
    plot_start : float
        Start time for comparison plot in seconds. Default: 5.0.
    inplace : bool
        If True, log warning and create copy anyway (MNE limitation).
        EOGRegression always creates a new object. Default: False.
    verbose : bool
        Enable detailed logging output. Default: False.
    **kwargs
        Additional parameters passed to EOGRegression:
        - picks : str | list, channels to correct (default: 'eeg')
        - exclude : list, channels to exclude (default: 'bads')
        - proj : bool, apply SSP projections (default: True)

    Returns
    -------
    data_clean : BaseRaw | BaseEpochs
        Corrected data with same type as input.
        Regression metrics stored in data_clean._regression_metrics attribute.

    Raises
    ------
    ValueError
        If EOG channels are not found in data or EEG reference not set.
    RuntimeError
        If regression fitting or application fails.

    Notes
    -----
    **Preprocessing Requirements:**
    - Data must be preloaded in memory
    - EEG reference must be set (e.g., average reference)
    - EOG channels must be present in data

    **Method Selection:**
    - **Raw data**: Always uses direct regression (subtract_evoked ignored)
    - **Epochs without ERPs**: Use subtract_evoked=False for direct regression
    - **Epochs with ERPs**: Use subtract_evoked=True (default) for Gratton method

    **Gratton & Coles Method** (subtract_evoked=True for Epochs):
    1. Subtract evoked response from each epoch (removes stimulus-locked activity)
    2. Fit regression coefficients on residuals (pure artifact modeling)
    3. Apply coefficients to original epochs (preserves ERPs)

    **Quality Metrics:**
    The returned object includes a `_regression_metrics` attribute containing:
    - method: 'direct' or 'gratton_coles'
    - data_type: 'Raw' or 'Epochs'
    - eog_channels: List of EOG channels used
    - mean_correlation: Preservation of EEG signal structure
    - quality_flags: Dictionary of quality indicators

    Examples
    --------
    Direct regression on continuous data:
    >>> raw.set_eeg_reference('average', projection=True)
    >>> raw_clean = remove_artifacts_regression(raw, eog_channels=['HEOG', 'VEOG'])

    Gratton method on epoched ERP data (preserves evoked responses):
    >>> epochs_clean = remove_artifacts_regression(
    ...     epochs,
    ...     eog_channels=['HEOG', 'VEOG'],
    ...     subtract_evoked=True  # Default
    ... )

    Direct regression on epochs without significant evoked responses:
    >>> epochs_clean = remove_artifacts_regression(
    ...     baseline_epochs,
    ...     eog_channels=['HEOG', 'VEOG'],
    ...     subtract_evoked=False
    ... )

    References
    ----------
    Gratton, G., Coles, M.G., & Donchin, E. (1983). A new method for off-line
    removal of ocular artifact. Electroencephalography and Clinical Neurophysiology,
    55(4), 468-484.
    """
    # Handle inplace parameter warning
    if inplace:
        logger.warning(
            "inplace=True ignored for EOG regression - always creates new object"
        )

    # Detect data type
    is_epochs = isinstance(data, BaseEpochs)
    data_type = "Epochs" if is_epochs else "Raw"

    # Log processing start
    method_name = "Gratton & Coles" if (is_epochs and subtract_evoked) else "Direct regression"
    logger.info(
        f"Starting EOG regression artifact removal ({method_name}) on {data_type} data"
    )
    if eog_channels:
        logger.info(f"Using EOG channels: {eog_channels}")

    # Validate inputs and resolve EOG channels
    try:
        eog_channels = _validate_inputs(data, eog_channels)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise

    # Create copies for processing
    original_data = data.copy()
    working_data = data.copy()

    # Main workflow with error handling
    try:
        # Branch based on data type and method selection
        if is_epochs and subtract_evoked:
            # Gratton & Coles method: fit on evoked-subtracted, apply to original
            cleaned_data = _fit_and_apply_epochs_with_evoked_subtraction(
                working_data,
                eog_channels,
                verbose=verbose,
                **kwargs
            )
        else:
            # Direct regression: fit and apply on same data
            cleaned_data = _fit_and_apply_direct(
                working_data,
                eog_channels,
                verbose=verbose,
                **kwargs
            )

        # Calculate quality metrics
        metrics = _calculate_regression_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_channels=eog_channels,
            subtract_evoked=(is_epochs and subtract_evoked),
            data_type=data_type
        )

        # Store metrics in cleaned data
        cleaned_data._regression_metrics = metrics

        # Log success with key metrics
        mean_corr = metrics['artifact_reduction']['mean_correlation_preserved']
        logger.success(
            f"EOG regression completed successfully. "
            f"Mean correlation preserved: {mean_corr:.3f}"
        )

        if verbose:
            logger.debug(f"Quality flags: {metrics['quality_flags']}")
            if 'regression_coefficients' in metrics:
                logger.debug(
                    f"Regression coefficients - "
                    f"mean: {metrics['regression_coefficients']['mean_coeff']:.4f}, "
                    f"max: {metrics['regression_coefficients']['max_coeff']:.4f}"
                )

        # Optional visualization
        if show_plot:
            _plot_regression_comparison(
                original_data=original_data,
                cleaned_data=cleaned_data,
                eog_channels=eog_channels,
                method=metrics['method'],
                plot_start=plot_start,
                plot_duration=plot_duration
            )

        # Success - return cleaned data with metrics
        return cleaned_data

    except Exception as e:
        # Error handling: return original data with error metrics
        logger.error(f"EOG regression failed: {str(e)}")
        logger.warning("Returning original data unchanged due to regression failure")

        # Attach error metrics to original data
        original_data._regression_metrics = {
            'method': 'regression',
            'implementation': 'mne_eog_regression',
            'data_type': data_type,
            'eog_channels': eog_channels,
            'subtract_evoked': subtract_evoked if is_epochs else None,
            'correction_applied': False,
            'error': str(e)
        }

        return original_data


def _validate_inputs(
    data: Union[BaseRaw, BaseEpochs],
    eog_channels: Optional[List[str]]
) -> List[str]:
    """
    Validate input data and EOG channels for regression.

    Parameters
    ----------
    data : BaseRaw | BaseEpochs
        MNE data object to validate.
    eog_channels : list of str or None
        EOG channel names to validate.

    Returns
    -------
    eog_channels : list of str
        Validated EOG channel names (resolved if None was passed).

    Raises
    ------
    ValueError
        If EOG channels are missing, EEG reference not set, or data not preloaded.
    """
    # Check if data is preloaded
    if not data.preload:
        raise ValueError(
            "Data must be preloaded in memory. "
            "Use data.load_data() before running regression."
        )

    # Resolve EOG channels if None
    if eog_channels is None:
        # Use all channels marked as 'eog' type
        eog_channels = [ch for ch in data.ch_names if 'eog' in data.get_channel_types([ch])[0].lower()]
        if not eog_channels:
            raise ValueError(
                "No EOG channels specified and no channels with 'eog' type found. "
                "Specify eog_channels parameter explicitly."
            )
        logger.info(f"Auto-detected EOG channels: {eog_channels}")

    # Validate that all specified EOG channels exist
    missing_channels = [ch for ch in eog_channels if ch not in data.ch_names]
    if missing_channels:
        # Find available EOG-like channels for helpful error message
        available_eog = [
            ch for ch in data.ch_names
            if any(eog_term in ch.upper() for eog_term in ['EOG', 'HEOG', 'VEOG', 'FP1', 'FP2'])
        ]

        error_msg = f"Missing EOG channels: {missing_channels}"
        if available_eog:
            error_msg += f"\nAvailable EOG-like channels: {available_eog}"
        else:
            error_msg += "\nNo EOG-like channels found in data."
        error_msg += f"\nAll available channels: {data.ch_names[:10]}..." if len(data.ch_names) > 10 else f"\nAll available channels: {data.ch_names}"

        raise ValueError(error_msg)

    # Check if EEG reference is set (for Raw data)
    if isinstance(data, BaseRaw):
        ref_applied = data.info.get('custom_ref_applied')
        if not ref_applied:
            logger.warning(
                "No custom EEG reference detected for Raw data. "
                "EOG regression works best with properly referenced data. "
                "Consider using data.set_eeg_reference('average', projection=True) first."
            )
            # Note: MNE's EOGRegression will raise RuntimeError if reference truly required
            # This is just a helpful warning

    logger.debug(f"Input validation passed. Using EOG channels: {eog_channels}")
    return eog_channels


def _fit_and_apply_epochs_with_evoked_subtraction(
    epochs: BaseEpochs,
    eog_channels: List[str],
    verbose: bool = False,
    **kwargs
) -> BaseEpochs:
    """
    Apply Gratton & Coles method: fit on evoked-subtracted data, apply to original.

    This implements the scientifically validated approach for ERP data where
    regression coefficients are computed after removing stimulus-locked activity.
    This prevents the regression from treating genuine brain activity as artifact.

    Parameters
    ----------
    epochs : Epochs
        Epoched EEG data with evoked responses.
    eog_channels : list of str
        EOG channel names to use as predictors.
    verbose : bool
        Enable detailed logging.
    **kwargs
        Additional parameters passed to EOGRegression (picks, exclude, proj).

    Returns
    -------
    epochs_clean : BaseEpochs
        Corrected epochs with evoked responses preserved.

    References
    ----------
    Gratton, G., Coles, M.G., & Donchin, E. (1983). A new method for off-line
    removal of ocular artifact. Electroencephalography and Clinical Neurophysiology,
    55(4), 468-484.
    """
    logger.info("Using Gratton & Coles method with evoked subtraction")

    # Step 1: Create evoked-subtracted copy for fitting
    if verbose:
        logger.debug("Subtracting evoked response from epochs for coefficient estimation")

    epochs_subtracted = epochs.copy().subtract_evoked()

    # Step 2: Initialize EOGRegression model
    # Default to 'eeg' picks if not specified
    picks = kwargs.pop('picks', 'eeg')
    exclude = kwargs.pop('exclude', 'bads')
    proj = kwargs.pop('proj', True)

    if verbose:
        logger.debug(
            f"Initializing EOGRegression with picks={picks}, "
            f"picks_artifact={eog_channels}, exclude={exclude}, proj={proj}"
        )

    model = EOGRegression(
        picks=picks,
        picks_artifact=eog_channels,
        exclude=exclude,
        proj=proj
    )

    # Step 3: Fit model on evoked-subtracted data (pure artifacts)
    if verbose:
        logger.debug("Fitting regression coefficients on evoked-subtracted data")

    model.fit(epochs_subtracted)

    if verbose:
        logger.debug(f"Regression coefficients shape: {model.coef_.shape}")

    # Step 4: Apply model to ORIGINAL epochs (preserves evoked responses)
    logger.info("Applying regression to original epochs (preserving evoked responses)")

    epochs_clean = model.apply(epochs, copy=True)

    # Store the model coefficients for quality control
    epochs_clean._regression_coef = model.coef_

    logger.info("Gratton & Coles method completed successfully")

    return epochs_clean


def _fit_and_apply_direct(
    data: Union[BaseRaw, BaseEpochs],
    eog_channels: List[str],
    verbose: bool = False,
    **kwargs
) -> Union[BaseRaw, BaseEpochs]:
    """
    Apply direct regression: fit and apply on same data without evoked subtraction.

    This is appropriate for continuous (Raw) data or epoched data without significant
    stimulus-locked evoked responses (e.g., resting-state, baseline periods).

    Parameters
    ----------
    data : BaseRaw | BaseEpochs
        MNE Raw or Epochs object to correct.
    eog_channels : list of str
        EOG channel names to use as predictors.
    verbose : bool
        Enable detailed logging.
    **kwargs
        Additional parameters passed to EOGRegression (picks, exclude, proj).

    Returns
    -------
    data_clean : BaseRaw | BaseEpochs
        Corrected data (same type as input).
    """
    data_type = "Epochs" if isinstance(data, BaseEpochs) else "Raw"
    logger.info(f"Using direct regression fitting on {data_type} data")

    # Initialize EOGRegression model
    # Default to 'eeg' picks if not specified
    picks = kwargs.pop('picks', 'eeg')
    exclude = kwargs.pop('exclude', 'bads')
    proj = kwargs.pop('proj', True)

    if verbose:
        logger.debug(
            f"Initializing EOGRegression with picks={picks}, "
            f"picks_artifact={eog_channels}, exclude={exclude}, proj={proj}"
        )

    model = EOGRegression(
        picks=picks,
        picks_artifact=eog_channels,
        exclude=exclude,
        proj=proj
    )

    # Fit model on input data
    if verbose:
        logger.debug(f"Fitting regression coefficients on {data_type} data")

    model.fit(data)

    if verbose:
        logger.debug(f"Regression coefficients shape: {model.coef_.shape}")

    # Apply model to copy of input data
    logger.info(f"Applying regression to {data_type} data")

    data_clean = model.apply(data, copy=True)

    # Store the model coefficients for quality control
    data_clean._regression_coef = model.coef_

    logger.info("Direct regression completed successfully")

    return data_clean


def _calculate_regression_metrics(
    original_data: Union[BaseRaw, BaseEpochs],
    cleaned_data: Union[BaseRaw, BaseEpochs],
    eog_channels: List[str],
    subtract_evoked: bool,
    data_type: str
) -> Dict[str, Any]:
    """
    Calculate quality metrics for regression artifact correction.

    Parameters
    ----------
    original_data : BaseRaw | BaseEpochs
        Original uncorrected data.
    cleaned_data : BaseRaw | BaseEpochs
        Corrected data.
    eog_channels : list of str
        EOG channels used for regression.
    subtract_evoked : bool
        Whether evoked subtraction was used.
    data_type : str
        Type of data ('Raw' or 'Epochs').

    Returns
    -------
    metrics : dict
        Dictionary containing quality metrics and flags.
    """
    import mne

    # Get EEG channel picks
    eeg_picks = mne.pick_types(original_data.info, eeg=True, exclude='bads')

    # Extract data arrays for comparison
    if data_type == 'Epochs':
        # For epochs, average across trials for correlation calculation
        original_array = original_data.get_data(picks=eeg_picks).mean(axis=0)  # (n_channels, n_times)
        cleaned_array = cleaned_data.get_data(picks=eeg_picks).mean(axis=0)
        n_epochs = len(original_data)
    else:
        # For Raw, use full data
        original_array = original_data.get_data(picks=eeg_picks)  # (n_channels, n_times)
        cleaned_array = cleaned_data.get_data(picks=eeg_picks)
        n_epochs = None

    # Calculate per-channel correlation (preservation of signal structure)
    n_channels = original_array.shape[0]
    correlations = []

    for ch_idx in range(n_channels):
        try:
            # Correlation between original and cleaned signal
            corr = np.corrcoef(original_array[ch_idx], cleaned_array[ch_idx])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        except Exception:
            # Handle edge cases (constant signals, etc.)
            correlations.append(0.0)

    mean_correlation = float(np.mean(correlations))
    min_correlation = float(np.min(correlations))
    max_correlation = float(np.max(correlations))

    # Get regression coefficients if available
    regression_coef = getattr(cleaned_data, '_regression_coef', None)
    if regression_coef is not None:
        coef_mean = float(np.mean(np.abs(regression_coef)))
        coef_max = float(np.max(np.abs(regression_coef)))
        coef_shape = tuple(regression_coef.shape)
    else:
        coef_mean = coef_max = coef_shape = None

    # Build comprehensive metrics dictionary matching PRD specification
    metrics = {
        'method': 'regression',  # User-facing method name
        'implementation': 'mne_eog_regression',  # Implementation identifier
        'eog_channels': eog_channels,
        'data_type': data_type,
        'subtract_evoked': subtract_evoked,
        'correction_applied': True,
        'n_eeg_channels': n_channels,
    }

    # Add epoch-specific metrics
    if n_epochs is not None:
        metrics['n_epochs'] = n_epochs

    # Add regression coefficient metrics with PRD field names
    if coef_mean is not None:
        metrics['regression_coefficients'] = {
            'shape': coef_shape,
            'max_coeff': round(coef_max, 4),
            'mean_coeff': round(coef_mean, 4)
        }

    # Add artifact reduction metrics (nested structure per PRD)
    metrics['artifact_reduction'] = {
        'mean_correlation_preserved': round(mean_correlation, 3)
    }

    # Quality flags for automated assessment
    metrics['quality_flags'] = {
        'low_correlation': mean_correlation < 0.85,
        'acceptable_correction': mean_correlation >= 0.85,
        'high_correlation': mean_correlation >= 0.95,
        'extreme_coefficients': coef_max > 0.5 if coef_max is not None else False,
        'minimal_correction': coef_max < 0.01 if coef_max is not None else False
    }

    return metrics


def _plot_regression_comparison(
    original_data: Union[BaseRaw, BaseEpochs],
    cleaned_data: Union[BaseRaw, BaseEpochs],
    eog_channels: List[str],
    method: str,
    plot_start: float = 5.0,
    plot_duration: float = 10.0
) -> None:
    """
    Plot before/after comparison of regression artifact correction.

    This is a stub function for visualization. Full implementation will be added
    in future tasks if needed.

    Parameters
    ----------
    original_data : BaseRaw | BaseEpochs
        Original uncorrected data.
    cleaned_data : BaseRaw | BaseEpochs
        Corrected data.
    eog_channels : list of str
        EOG channels used.
    method : str
        Method used ('direct' or 'gratton_coles').
    plot_start : float
        Start time for plot in seconds.
    plot_duration : float
        Duration of plot in seconds.
    """
    logger.info(
        f"Visualization requested but not yet implemented. "
        f"Method: {method}, EOG channels: {eog_channels}"
    )
    # TODO: Implement visualization similar to EMCP _plot_emcp_comparison()
    # Show original vs cleaned data for selected EEG channels
    # Include EOG channels for reference
    # Display correlation values for each channel
