"""
Eye Movement Correction Procedures (EMCP) - Blink Artifact Removal

This module implements two methods for removing blink artifacts from EEG data:

1. **EOG Regression Method** - Uses MNE's EOGRegression (requires average reference)
2. **Gratton & Coles Method** - Reference-agnostic linear regression approach

The methods have different reference requirements and are designed to integrate 
seamlessly with the EEG processing pipeline with comprehensive quality tracking.

References:
    - Gratton, G., Coles, M. G. H., & Donchin, E. (1983). A new method for off-line 
      removal of ocular artifact. Electroencephalography and Clinical Neurophysiology, 
      55(4), 468-484.
    - MNE-Python EOGRegression: https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html

Usage:
    # EOG Regression method (standard approach)
    cleaned_raw = remove_blinks_eog_regression(
        raw, 
        eog_channels=['HEOG', 'VEOG'],
        show_plot=False
    )
    
    # Gratton & Coles method (reference-agnostic)
    cleaned_raw = remove_blinks_gratton_coles(
        raw,
        eog_channels=['HEOG', 'VEOG'], 
        subtract_evoked=True,
        show_plot=False
    )

Integration with Pipeline:
    The remove_blinks_emcp stage can be added to processing configurations:
    
    stages:
      - remove_blinks_emcp:
          method: "eog_regression"  # or "gratton_coles"
          eog_channels: ["HEOG", "VEOG"]
          show_plot: false
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

# MNE imports
from mne.io import BaseRaw
from mne import Epochs, Evoked, pick_types
from mne.preprocessing import EOGRegression, find_eog_events


def remove_blinks_eog_regression(
    raw: BaseRaw,
    eog_channels: List[str] = ['HEOG', 'VEOG'],
    show_plot: bool = False,
    plot_duration: float = 10.0,
    plot_start: float = 5.0,
    inplace: bool = False,
    verbose: bool = False,
    **kwargs
) -> BaseRaw:
    """
    Remove blink artifacts using MNE's EOGRegression method.
    
    This method uses MNE's EOGRegression class to identify and remove blink artifacts
    from EEG data. **IMPORTANT: For EEG data, apply the desired reference (typically 
    average reference) before using this method, as recommended by MNE.**
    
    Args:
        raw: MNE Raw object containing EEG and EOG data (should be referenced)
        eog_channels: List of EOG channel names for blink detection
        show_plot: Whether to display before/after comparison plot
        plot_duration: Duration of comparison plot (seconds)
        plot_start: Start time for comparison plot (seconds)
        inplace: Whether to modify data in-place (ignored - always creates copy)
        verbose: Enable detailed logging
        **kwargs: Additional parameters passed to EOGRegression
        
    Returns:
        Raw object with blink artifacts removed
        
    Raises:
        ValueError: If EOG channels are missing or no blink events found
        
    Notes:
        - **Requires proper EEG reference (typically average) before application**
        - Generates _emcp_metrics for quality tracking
        - Should be used after re-referencing stage in processing pipeline
        - Supports both Raw and Epochs data types
    """
    if inplace:
        logger.info("inplace=True ignored for EMCP - always creates new object")
    
    logger.info(f"Starting EOG regression blink correction with channels: {eog_channels}")
    
    # Validate EOG channels
    _validate_eog_channels(raw, eog_channels)
    
    # Create working copy
    working_raw = raw.copy()
    original_raw = raw.copy()
    
    # Store original reference info for metrics
    original_ref = working_raw.info.get('custom_ref_applied', 'unknown')
    
    try:
        # Detect blink events using primary EOG channel (typically VEOG)
        primary_eog = eog_channels[1] if len(eog_channels) > 1 else eog_channels[0]
        eog_events = find_eog_events(working_raw, ch_name=primary_eog, verbose=verbose)
        
        if len(eog_events) == 0:
            logger.warning("No blink events detected - returning original data")
            working_raw._emcp_metrics = {
                'method': 'eog_regression',
                'eog_channels': eog_channels,
                'blink_events_found': 0,
                'correction_applied': False,
                'original_reference': original_ref,
                'error': 'no_blink_events'
            }
            return working_raw
        
        if verbose:
            logger.info(f"Found {len(eog_events)} blink events using {primary_eog}")
        
        # Get EEG and EOG channel indices
        eeg_picks = pick_types(working_raw.info, eeg=True, meg=False)
        eog_picks = [working_raw.ch_names.index(ch) for ch in eog_channels 
                     if ch in working_raw.ch_names]
        
        if len(eog_picks) == 0:
            raise ValueError("No valid EOG channels found in data")
        
        # Initialize EOGRegression with explicit parameters
        eog_regression_params = {
            'picks': eeg_picks,
            'picks_artifact': eog_picks,
            **kwargs  # Allow additional parameters from config
        }
        
        logger.info("Fitting EOG regression model...")
        eog_regressor = EOGRegression(**eog_regression_params)
        eog_regressor.fit(working_raw)
        
        # Apply regression correction
        logger.info("Applying EOG regression correction...")
        cleaned_raw = eog_regressor.apply(working_raw.copy())
        
        # Calculate comprehensive quality metrics
        original_data = original_raw.get_data(picks=eeg_picks)
        cleaned_data = cleaned_raw.get_data(picks=eeg_picks)
        eog_data = original_raw.get_data(picks=eog_picks)
        
        # Calculate regression coefficients from the fitted model
        # EOGRegression stores regression weights in the model
        regression_coeffs = None
        if hasattr(eog_regressor, 'coef_'):
            regression_coeffs = eog_regressor.coef_
        
        metrics = _calculate_emcp_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_data=eog_data,
            method='eog_regression',
            regression_coeffs=regression_coeffs,
            blink_events=len(eog_events),
            eog_channels=eog_channels,
            primary_eog_channel=primary_eog,
            original_reference=original_ref
        )
        
        # Store metrics in cleaned raw object
        cleaned_raw._emcp_metrics = metrics
        
        logger.success(f"EOG regression completed. Corrected {len(eog_events)} blink events")
        logger.info(f"Mean correlation preserved: {metrics.get('mean_correlation', 'N/A'):.3f}")
        
        # Optional visualization
        if show_plot:
            _plot_emcp_comparison(
                raw_original=original_raw,
                raw_cleaned=cleaned_raw,
                eog_channels=eog_channels,
                method='eog_regression',
                plot_start=plot_start,
                plot_duration=plot_duration
            )
        
        return cleaned_raw
        
    except Exception as e:
        logger.error(f"EOG regression failed: {str(e)}")
        
        # Return original data with error metrics
        error_raw = original_raw.copy()
        error_raw._emcp_metrics = {
            'method': 'eog_regression',
            'eog_channels': eog_channels,
            'correction_applied': False,
            'original_reference': original_ref,
            'error': str(e)
        }
        
        return error_raw


def remove_blinks_gratton_coles(
    raw: BaseRaw,
    eog_channels: List[str] = ['HEOG', 'VEOG'],
    subtract_evoked: bool = True,
    show_plot: bool = False,
    plot_duration: float = 10.0,
    plot_start: float = 5.0,
    inplace: bool = False,
    verbose: bool = False,
    **kwargs
) -> BaseRaw:
    """
    Remove blink artifacts using the Gratton & Coles method.
    
    This method implements the original Gratton & Coles (1983) approach using
    direct linear regression. It is reference-agnostic and works with any
    reference scheme without requiring modifications.
    
    Args:
        raw: MNE Raw object containing EEG and EOG data
        eog_channels: List of EOG channel names for blink detection
        subtract_evoked: Whether to subtract evoked blink response
        show_plot: Whether to display before/after comparison plot
        plot_duration: Duration of comparison plot (seconds)
        plot_start: Start time for comparison plot (seconds)
        inplace: Whether to modify data in-place (ignored - always creates copy)
        verbose: Enable detailed logging
        **kwargs: Additional parameters for regression
        
    Returns:
        Raw object with blink artifacts removed
        
    Raises:
        ValueError: If EOG channels are missing or no blink events found
        
    Notes:
        - Reference-agnostic - works with any reference scheme
        - Implements original Gratton & Coles algorithm
        - Supports evoked response subtraction option
        - Calculates and stores regression coefficients
        - Generates quality metrics compatible with pipeline
    """
    if inplace:
        logger.info("inplace=True ignored for EMCP - always creates new object")
    
    logger.info(f"Starting Gratton & Coles blink correction with channels: {eog_channels}")
    
    # Validate EOG channels
    _validate_eog_channels(raw, eog_channels)
    
    # Create working copies
    working_raw = raw.copy()
    original_raw = raw.copy()
    
    # Store original reference info for metrics
    original_ref = working_raw.info.get('custom_ref_applied', 'unknown')
    
    try:
        # Detect blink events using primary EOG channel (typically VEOG)
        primary_eog = eog_channels[1] if len(eog_channels) > 1 else eog_channels[0]
        eog_events = find_eog_events(working_raw, ch_name=primary_eog, verbose=verbose)
        
        if len(eog_events) == 0:
            logger.warning("No blink events detected - returning original data")
            working_raw._emcp_metrics = {
                'method': 'gratton_coles',
                'eog_channels': eog_channels,
                'blink_events_found': 0,
                'correction_applied': False,
                'original_reference': original_ref,
                'error': 'no_blink_events'
            }
            return working_raw
        
        if verbose:
            logger.info(f"Found {len(eog_events)} blink events using {primary_eog}")
        
        # Get EEG and EOG channel indices
        eeg_picks = pick_types(working_raw.info, eeg=True, meg=False)
        eog_picks = [working_raw.ch_names.index(ch) for ch in eog_channels 
                     if ch in working_raw.ch_names]
        
        if len(eog_picks) == 0:
            raise ValueError("No valid EOG channels found in data")
        
        # Extract data for processing
        eeg_data = working_raw.get_data(picks=eeg_picks)  # Shape: (n_eeg_channels, n_times)
        eog_data = working_raw.get_data(picks=eog_picks)  # Shape: (n_eog_channels, n_times)
        
        # Use primary EOG channel for regression
        primary_eog_idx = eog_picks[1] if len(eog_picks) > 1 else eog_picks[0]
        primary_eog_data = working_raw.get_data(picks=[primary_eog_idx])[0]  # Shape: (n_times,)
        
        # Initialize arrays for corrected data and regression coefficients
        corrected_eeg = eeg_data.copy()
        regression_coeffs = np.zeros(len(eeg_picks))
        
        logger.info("Computing Gratton & Coles regression coefficients...")
        
        # Apply Gratton & Coles method to each EEG channel
        for i, eeg_ch_idx in enumerate(eeg_picks):
            eeg_ch_data = eeg_data[i]
            
            # Calculate regression coefficient using least squares
            # b = (sum(EOG * EEG)) / (sum(EOG^2))
            # Following Gratton & Coles (1983) Equation 1
            covariance = np.sum(primary_eog_data * eeg_ch_data)
            eog_variance = np.sum(primary_eog_data * primary_eog_data)
            
            if eog_variance > 0:
                beta = covariance / eog_variance
                regression_coeffs[i] = beta
                
                # Apply correction: Corrected_EEG = Original_EEG - beta * EOG
                corrected_eeg[i] = eeg_ch_data - beta * primary_eog_data
                
                if verbose and i < 5:  # Log first few channels
                    channel_name = working_raw.ch_names[eeg_ch_idx]
                    logger.debug(f"Channel {channel_name}: beta = {beta:.6f}")
            else:
                logger.warning(f"Zero EOG variance for channel {working_raw.ch_names[eeg_ch_idx]}")
                regression_coeffs[i] = 0.0
        
        # Optional: Subtract evoked blink response (Gratton & Coles enhancement)
        if subtract_evoked:
            logger.info("Computing and subtracting evoked blink response...")
            
            # Create epochs around blink events for evoked response calculation
            # Use a wider window for evoked response estimation
            tmin, tmax = -0.2, 0.4  # 200ms before to 400ms after blink
            
            try:
                # Create temporary events array for epoching
                events_array = np.column_stack([
                    eog_events[:, 0],  # Sample indices
                    np.zeros(len(eog_events), dtype=int),  # Previous event (unused)
                    np.ones(len(eog_events), dtype=int)    # Event ID
                ])
                
                # Create epochs around blinks
                from mne import Epochs
                blink_epochs = Epochs(
                    working_raw,
                    events_array,
                    event_id={'blink': 1},
                    tmin=tmin,
                    tmax=tmax,
                    picks=eeg_picks,
                    baseline=None,
                    preload=True,
                    verbose=False
                )
                
                if len(blink_epochs) > 0:
                    # Calculate average evoked response
                    evoked_response = blink_epochs.average().data  # Shape: (n_channels, n_times)
                    
                    # Subtract evoked response from each blink occurrence
                    sfreq = working_raw.info['sfreq']
                    
                    for event in eog_events:
                        blink_sample = event[0]
                        start_sample = int(blink_sample + tmin * sfreq)
                        end_sample = int(blink_sample + tmax * sfreq)
                        
                        # Ensure we don't go out of bounds
                        if start_sample >= 0 and end_sample < corrected_eeg.shape[1]:
                            n_samples = end_sample - start_sample
                            if n_samples == evoked_response.shape[1]:
                                corrected_eeg[:, start_sample:end_sample] -= evoked_response
                
                logger.info(f"Subtracted evoked response from {len(blink_epochs)} epochs")
                
            except Exception as e:
                logger.warning(f"Could not subtract evoked response: {str(e)}")
                subtract_evoked = False  # Mark as not applied for metrics
        
        # Create new Raw object with corrected data
        cleaned_raw = working_raw.copy()
        corrected_data = cleaned_raw.get_data()
        corrected_data[eeg_picks] = corrected_eeg
        
        # Update the data in the raw object
        # Note: We need to use _data if it exists, or the appropriate MNE method
        if hasattr(cleaned_raw, '_data'):
            cleaned_raw._data = corrected_data
        else:
            # Alternative approach for different MNE versions
            cleaned_raw = working_raw.copy()
            for i, pick in enumerate(eeg_picks):
                cleaned_raw._data[pick] = corrected_eeg[i]
        
        # Calculate comprehensive quality metrics
        original_data = original_raw.get_data(picks=eeg_picks)
        cleaned_data = corrected_eeg
        full_eog_data = original_raw.get_data(picks=eog_picks)
        
        metrics = _calculate_emcp_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_data=full_eog_data,
            method='gratton_coles',
            regression_coeffs=regression_coeffs,
            blink_events=len(eog_events),
            eog_channels=eog_channels,
            primary_eog_channel=primary_eog,
            original_reference=original_ref,
            subtract_evoked=subtract_evoked
        )
        
        # Store metrics in cleaned raw object
        cleaned_raw._emcp_metrics = metrics
        
        logger.success(f"Gratton & Coles correction completed. Processed {len(eog_events)} blink events")
        logger.info(f"Mean correlation preserved: {metrics.get('mean_correlation', 'N/A'):.3f}")
        logger.info(f"Mean regression coefficient: {np.mean(np.abs(regression_coeffs)):.6f}")
        
        # Optional visualization
        if show_plot:
            _plot_emcp_comparison(
                raw_original=original_raw,
                raw_cleaned=cleaned_raw,
                eog_channels=eog_channels,
                method='gratton_coles',
                plot_start=plot_start,
                plot_duration=plot_duration
            )
        
        return cleaned_raw
        
    except Exception as e:
        logger.error(f"Gratton & Coles correction failed: {str(e)}")
        
        # Return original data with error metrics
        error_raw = original_raw.copy()
        error_raw._emcp_metrics = {
            'method': 'gratton_coles',
            'eog_channels': eog_channels,
            'correction_applied': False,
            'original_reference': original_ref,
            'error': str(e)
        }
        
        return error_raw


def _validate_eog_channels(raw: BaseRaw, eog_channels: List[str]) -> None:
    """
    Validate that required EOG channels exist in the raw data.
    
    Args:
        raw: MNE Raw object
        eog_channels: List of EOG channel names to validate
        
    Raises:
        ValueError: If any EOG channels are missing
    """
    if not eog_channels:
        raise ValueError("At least one EOG channel must be specified")
    
    missing_channels = [ch for ch in eog_channels if ch not in raw.ch_names]
    
    if missing_channels:
        available_channels = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
        error_msg = f"Missing EOG channels: {missing_channels}"
        if available_channels:
            error_msg += f". Available EOG-like channels: {available_channels}"
        else:
            error_msg += ". No EOG channels found in data."
        raise ValueError(error_msg)
    
    logger.debug(f"EOG channel validation passed: {eog_channels}")


def _calculate_emcp_metrics(
    original_data: np.ndarray,
    cleaned_data: np.ndarray,
    eog_data: np.ndarray,
    method: str,
    regression_coeffs: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate essential quality metrics for EMCP correction.
    
    Focuses only on metrics that researchers actually use for quality assessment:
    - Success/failure status
    - Blink detection count (sanity check)
    - Mean correlation (overcorrection detection)
    - Extreme coefficients flag (for Gratton & Coles)
    
    Args:
        original_data: Original EEG data
        cleaned_data: Cleaned EEG data
        eog_data: EOG channel data
        method: EMCP method used ("eog_regression" or "gratton_coles")
        regression_coeffs: Regression coefficients if available
        **kwargs: Additional method-specific parameters
        
    Returns:
        Dictionary containing essential quality metrics only
    """
    n_channels, n_times = original_data.shape
    
    # Calculate mean correlation (key overcorrection indicator)
    correlations = []
    for ch_idx in range(n_channels):
        try:
            corr = np.corrcoef(original_data[ch_idx], cleaned_data[ch_idx])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        except:
            correlations.append(0.0)
    
    mean_correlation = np.nanmean(correlations)
    
    # Essential metrics only
    metrics = {
        'method': method,
        'correction_applied': True,
        'blink_events': kwargs.get('blink_events', 0),
        'mean_correlation': round(float(mean_correlation), 3),
        'eog_channels': kwargs.get('eog_channels', []),
    }
    
    # Method-specific essential metrics
    if method == 'gratton_coles' and regression_coeffs is not None:
        max_coeff = float(np.max(np.abs(regression_coeffs)))
        metrics.update({
            'max_regression_coefficient': round(max_coeff, 4),
            'extreme_coefficients': max_coeff > 0.5,  # Flag for extreme coefficients
            'subtract_evoked': kwargs.get('subtract_evoked', False)
        })
    
    # Simple quality flags for essential issues only
    metrics['quality_flags'] = {
        'no_blinks_detected': metrics['blink_events'] == 0,
        'low_correlation': mean_correlation < 0.8,  # Possible overcorrection
        'extreme_coefficients': metrics.get('extreme_coefficients', False)
    }
    
    return metrics


def _plot_emcp_comparison(
    raw_original: BaseRaw,
    raw_cleaned: BaseRaw,
    eog_channels: List[str],
    method: str,
    plot_start: float = 5.0,
    plot_duration: float = 10.0,
    n_channels: int = 6
) -> None:
    """
    Create before/after comparison plots for EMCP correction.
    
    Args:
        raw_original: Original raw data
        raw_cleaned: Cleaned raw data
        eog_channels: List of EOG channels used
        method: EMCP method applied
        plot_start: Start time for plot (seconds)
        plot_duration: Duration to plot (seconds)
        n_channels: Maximum number of EEG channels to plot
    """
    # Get EEG channels for plotting
    eeg_picks = pick_types(raw_original.info, eeg=True, meg=False)[:n_channels]
    
    if len(eeg_picks) == 0:
        logger.warning("No EEG channels found for EMCP comparison plot")
        return
    
    # Calculate time indices for plotting
    sfreq = raw_original.info['sfreq']
    start_idx = int(plot_start * sfreq)
    end_idx = start_idx + int(plot_duration * sfreq)
    
    if end_idx > raw_original.n_times:
        end_idx = raw_original.n_times
        logger.warning("Plot duration adjusted to fit data length")
    
    time = raw_original.times[start_idx:end_idx]
    
    # Extract data for plotting
    original_data = raw_original.get_data(picks=eeg_picks, start=start_idx, stop=end_idx) * 1e6  # µV
    cleaned_data = raw_cleaned.get_data(picks=eeg_picks, start=start_idx, stop=end_idx) * 1e6  # µV
    
    # Get EOG data for reference
    eog_data = None
    if eog_channels:
        try:
            primary_eog = eog_channels[1] if len(eog_channels) > 1 else eog_channels[0]
            eog_pick = [raw_original.ch_names.index(primary_eog)]
            eog_data = raw_original.get_data(picks=eog_pick, start=start_idx, stop=end_idx) * 1e6
        except (ValueError, IndexError):
            logger.warning(f"Could not extract EOG data for plotting")
    
    # Create figure with subplots
    n_plots = len(eeg_picks) + (1 if eog_data is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    # Get channel names
    ch_names = [raw_original.ch_names[pick] for pick in eeg_picks]
    
    # Plot EOG channel first if available
    ax_idx = 0
    if eog_data is not None:
        axes[ax_idx].plot(time, eog_data[0], 'k-', linewidth=1, alpha=0.8, label=primary_eog)
        axes[ax_idx].set_ylabel(f'{primary_eog}\n(µV)', fontsize=10)
        axes[ax_idx].set_title(f'EOG Reference Channel: {primary_eog}', fontsize=11)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend(loc='upper right')
        ax_idx += 1
    
    # Plot EEG channels
    for i, ch_name in enumerate(ch_names):
        ax = axes[ax_idx + i]
        
        # Plot original data
        ax.plot(time, original_data[i], 'b-', alpha=0.7, linewidth=1, label='Original')
        
        # Plot cleaned data
        ax.plot(time, cleaned_data[i], 'r-', alpha=0.8, linewidth=1, label='Corrected')
        
        # Calculate correlation for this channel
        correlation = np.corrcoef(original_data[i], cleaned_data[i])[0, 1]
        
        # Calculate RMS reduction
        rms_original = np.sqrt(np.mean(original_data[i]**2))
        rms_cleaned = np.sqrt(np.mean(cleaned_data[i]**2))
        rms_change = (rms_cleaned - rms_original) / rms_original * 100
        
        ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{ch_name} - Correlation: {correlation:.3f}, RMS Change: {rms_change:+.1f}%', 
                    fontsize=10)
        
        if i == 0:
            ax.legend(loc='upper right')
    
    # Set x-axis label for bottom plot
    axes[-1].set_xlabel('Time (s)', fontsize=11)
    
    # Get metrics for overall title
    method_name = method.replace('_', ' ').title()
    if hasattr(raw_cleaned, '_emcp_metrics'):
        metrics = raw_cleaned._emcp_metrics
        mean_corr = metrics.get('mean_correlation', 0)
        n_blinks = metrics.get('blink_events', 0)
        
        fig.suptitle(
            f'{method_name} EMCP Correction - Mean Correlation: {mean_corr:.3f}, '
            f'Blink Events: {n_blinks}',
            fontsize=12, y=0.98
        )
    else:
        fig.suptitle(f'{method_name} EMCP Correction - Before vs After', fontsize=12, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Optional: Create a separate correlation summary plot
    if hasattr(raw_cleaned, '_emcp_metrics'):
        metrics = raw_cleaned._emcp_metrics
        correlations = metrics.get('channel_correlations', [])
        
        if correlations and len(correlations) > 1:
            # Create histogram of correlations
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Correlation histogram
            ax1.hist(correlations, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(correlations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(correlations):.3f}')
            ax1.set_xlabel('Correlation Coefficient')
            ax1.set_ylabel('Number of Channels')
            ax1.set_title('Channel Correlation Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Correction percentage by channel
            correction_pcts = metrics.get('correction_percentages', [])
            if correction_pcts:
                ax2.bar(range(len(correction_pcts)), correction_pcts, alpha=0.7, color='orange')
                ax2.set_xlabel('Channel Index')
                ax2.set_ylabel('Correction Percentage (%)')
                ax2.set_title('Correction Applied per Channel')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()


# Quality metrics and diagnostics
def get_emcp_quality_summary(raw: BaseRaw) -> Dict[str, Any]:
    """
    Extract EMCP quality summary from processed raw data.
    
    Args:
        raw: Processed raw data with _emcp_metrics attribute
        
    Returns:
        Dictionary with EMCP quality metrics or empty dict if not available
    """
    if hasattr(raw, '_emcp_metrics'):
        return raw._emcp_metrics.copy()
    else:
        return {
            'method': 'unknown',
            'correction_applied': False,
            'eog_channels': [],
            'metrics': {}
        }