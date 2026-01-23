"""
Custom wavelet time-frequency analysis.

This module provides experimental/advanced wavelet analyses using custom
wavelet families not available in MNE's built-in methods. Currently supports
Morse wavelets via the clouddrift package.

Uses clouddrift's native wavelet transform pipeline rather than MNE's low-level
cwt() function, providing a complete end-to-end solution for custom wavelets.
"""

import numpy as np
from mne import Epochs
from mne.time_frequency import AverageTFR
from typing import List, Union, Optional, Tuple
from loguru import logger
import pywt

# Optional: clouddrift for morse wavelets
try:
    import clouddrift as cd
    CLOUDDRIFT_AVAILABLE = True
except ImportError:
    CLOUDDRIFT_AVAILABLE = False
    cd = None


def clouddrift_morse_wavelet(
    data: np.ndarray,
    sfreq: float,
    freq_range: List[float],
    n_freqs: int,
    gamma: float = 3.0,
    beta: float = 3.0,
    pad_len: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Morse wavelet transform using clouddrift.

    Based on existing implementation from preprocessing_tools.py.

    Args:
        data: Signal data (n_times,) or (n_channels, n_times)
        sfreq: Sampling frequency in Hz
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequencies (logarithmically spaced)
        gamma: Morse gamma parameter (time decay)
        beta: Morse beta parameter (frequency bandwidth)
        pad_len: Padding length in seconds

    Returns:
        Tuple of (coefficients, frequencies)
        - coefficients: Complex TFR (n_freqs, n_times) or (n_channels, n_freqs, n_times)
        - frequencies: Frequency vector in Hz

    Raises:
        ValueError: If clouddrift not available
    """
    if not CLOUDDRIFT_AVAILABLE:
        raise ValueError(
            "Morse wavelets require the 'clouddrift' package. "
            "Install with: pip install clouddrift"
        )

    # Handle multi-channel data
    if data.ndim == 1:
        # Single channel
        data = data[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False

    n_channels, n_times = data.shape
    pad_samples = int(sfreq * pad_len)

    # Initialize output
    all_coefficients = []

    # Process each channel
    for ch_idx in range(n_channels):
        # Pad signal
        padded_y = pywt.pad(data[ch_idx], pad_widths=pad_samples, mode='reflect')

        # Generate frequency scale using clouddrift
        N = len(padded_y)

        # Create logarithmically spaced radian frequencies
        # clouddrift uses radian frequencies: ω = 2πf/sfreq
        min_radian = 2 * np.pi * freq_range[0] / sfreq
        max_radian = 2 * np.pi * freq_range[1] / sfreq

        # Use clouddrift's frequency scaling
        radian_freq = cd.wavelet.morse_logspace_freq(
            gamma, beta, N,
            highset=(0.1, max_radian),
            lowset=(0, min_radian)
        )

        # Convert to Hz
        frequencies = (radian_freq / (2 * np.pi)) * sfreq

        # Generate Morse wavelets
        normtype = 'bandpass'
        order = 1
        wavelet, _ = cd.wavelet.morse_wavelet(N, gamma, beta, radian_freq, order, normtype)

        # Apply wavelet transform
        coefficients_cd = cd.wavelet.wavelet_transform(padded_y, wavelet, time_axis=0)

        # Transpose and trim padding
        tf = coefficients_cd.T  # (n_freqs, n_times_padded)
        tf = tf[:, pad_samples:-pad_samples]  # Trim padding

        all_coefficients.append(tf)

    # Stack channels
    coefficients = np.stack(all_coefficients, axis=0)  # (n_channels, n_freqs, n_times)

    if squeeze_output:
        coefficients = coefficients.squeeze(0)  # (n_freqs, n_times)

    return coefficients, frequencies


def compute_custom_cwt_tfr(
    epochs: Epochs,
    wavelet_type: str = "morse",
    freq_range: List[float] = [1, 50],
    n_freqs: int = 100,
    # Morse-specific parameters
    morse_gamma: float = 3.0,
    morse_beta: float = 3.0,
    # Processing options
    pad_len: int = 20,
    compute_itc: bool = True,
    baseline: Optional[Tuple[float, float]] = None,
    baseline_mode: str = "mean",
    **kwargs
) -> AverageTFR:
    """
    Compute time-frequency representation using custom wavelets.

    Experimental stage for advanced wavelet analyses not available in MNE.
    Uses clouddrift's native wavelet transform pipeline.

    Args:
        epochs: Input epochs
        wavelet_type: Type of custom wavelet ('morse' currently supported)
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequency points (logarithmically spaced)
        morse_gamma: Morse wavelet gamma parameter (default: 3.0)
            Controls temporal decay. Higher values = better temporal resolution.
            Recommended range: 1-10
        morse_beta: Morse wavelet beta parameter (default: 3.0)
            Controls frequency bandwidth. Higher values = better frequency resolution.
            Recommended range: 1-20
        pad_len: Padding length in seconds (default: 20)
        compute_itc: Whether to compute inter-trial coherence
        baseline: Baseline period (tmin, tmax) for correction
        baseline_mode: 'mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'
        **kwargs: Additional parameters

    Returns:
        AverageTFR object containing power and optionally ITC

    Raises:
        ValueError: If wavelet_type not supported or clouddrift unavailable

    Scientific Rationale:
        Morse wavelets (Lilly & Olhede 2009) provide flexible time-frequency
        resolution control through independent gamma and beta parameters,
        offering advantages over fixed Morlet wavelets for certain analyses.

    References:
        - Lilly & Olhede (2009). IEEE Trans. Signal Process., 57(1), 146-160.
        - Lilly (2017). Proc. R. Soc. A, 473(2200), 20160776.

    Example:
        >>> # High temporal resolution
        >>> power = compute_custom_cwt_tfr(
        ...     epochs, wavelet_type='morse',
        ...     morse_gamma=6.0, morse_beta=3.0,
        ...     freq_range=[1, 50], n_freqs=100
        ... )
    """
    # Validate wavelet type
    if wavelet_type != "morse":
        raise ValueError(f"Unsupported wavelet_type: '{wavelet_type}'. Currently only 'morse' is supported.")

    if not CLOUDDRIFT_AVAILABLE:
        raise ValueError(
            "Morse wavelets require the 'clouddrift' package. "
            "Install with: pip install clouddrift"
        )

    logger.info(f"Computing custom CWT TFR: {wavelet_type} wavelets")
    logger.info(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz, {n_freqs} frequencies")
    logger.info(f"Morse parameters: gamma={morse_gamma:.1f}, beta={morse_beta:.1f}")

    # Get epoch info
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    times = epochs.times
    sfreq = epochs.info['sfreq']

    logger.info(f"Processing {n_epochs} epochs, {n_channels} channels")

    # Initialize storage for complex TFR
    # We'll compute TFR for each epoch and store complex values
    complex_tfr_data = []

    # Process each epoch
    for epoch_idx in range(n_epochs):
        epoch_data = epochs[epoch_idx].get_data()[0]  # (n_channels, n_times)

        # Compute Morse wavelet transform
        coefficients, frequencies = clouddrift_morse_wavelet(
            data=epoch_data,
            sfreq=sfreq,
            freq_range=freq_range,
            n_freqs=n_freqs,
            gamma=morse_gamma,
            beta=morse_beta,
            pad_len=pad_len
        )

        # coefficients: (n_channels, n_freqs, n_times)
        complex_tfr_data.append(coefficients)

    # Stack all epochs: (n_epochs, n_channels, n_freqs, n_times)
    complex_tfr_data = np.stack(complex_tfr_data, axis=0)

    logger.debug(f"Complex TFR shape: {complex_tfr_data.shape}")

    # Compute power
    power_data = np.abs(complex_tfr_data) ** 2

    # Compute average power across epochs
    avg_power = np.mean(power_data, axis=0)  # (n_channels, n_freqs, n_times)

    # Compute ITC if requested
    itc_data = None
    if compute_itc:
        logger.info("Computing inter-trial coherence...")
        # ITC = |mean(exp(i*phase))| across epochs
        phase = np.angle(complex_tfr_data)
        complex_unit = np.exp(1j * phase)
        itc_data = np.abs(np.mean(complex_unit, axis=0))  # (n_channels, n_freqs, n_times)

    # Create AverageTFR object using dummy morlet computation, then replace data
    # This workaround is needed because AverageTFR doesn't support direct construction
    logger.debug("Creating AverageTFR object structure using dummy morlet computation...")

    # Use morlet with matching frequency points to create the object structure
    dummy_power_tfr = epochs.compute_tfr(
        method='morlet',
        freqs=frequencies,
        n_cycles=frequencies / 2,  # Standard ratio
        use_fft=True,
        return_itc=False,
        average=True,
        verbose=False
    )

    # Replace data with our Morse wavelet results
    logger.debug("Replacing dummy data with Morse wavelet results...")
    dummy_power_tfr.data = avg_power
    # Note: comment and method are read-only, set during construction

    power_tfr = dummy_power_tfr

    # Apply baseline correction if requested
    if baseline is not None:
        logger.info(f"Applying baseline correction: {baseline}, mode: {baseline_mode}")
        power_tfr.apply_baseline(baseline, mode=baseline_mode)

    # Add ITC as a separate attribute if computed
    if itc_data is not None:
        # Create ITC using same dummy method
        logger.debug("Creating ITC AverageTFR object...")
        dummy_itc_tfr = epochs.compute_tfr(
            method='morlet',
            freqs=frequencies,
            n_cycles=frequencies / 2,
            use_fft=True,
            return_itc=False,
            average=True,
            verbose=False
        )
        dummy_itc_tfr.data = itc_data
        # Note: comment and method are read-only, set during construction

        power_tfr._itc_data = dummy_itc_tfr  # Matches ResultSaver convention
        logger.info("ITC computed and stored in power_tfr._itc_data")

    logger.success(f"Custom CWT TFR complete: {len(frequencies)} frequencies, {len(times)} time points")

    return power_tfr
