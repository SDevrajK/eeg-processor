import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from mne.time_frequency import tfr_morlet, tfr_multitaper, AverageTFR, Spectrum
from typing import List, Union, Optional, Tuple
from loguru import logger


def compute_epochs_tfr_average(epochs: Epochs,
                        freq_range: List[float] = [1, 50],
                        n_freqs: int = 100,
                        method: str = "morlet",
                        n_cycles: Union[float, List[float]] = None,
                        compute_itc: bool = True,
                        baseline: Optional[Tuple[float, float]] = None,
                        baseline_mode: str = "percent",
                        **kwargs) -> AverageTFR:
    """
    Compute averaged time-frequency representation from epochs

    Args:
        epochs: Input epochs
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequency points
        method: 'morlet' or 'multitaper'
        n_cycles: Cycles per frequency (auto-computed if None)
        compute_itc: Whether to compute inter-trial coherence
        baseline: Baseline period (tmin, tmax) for correction
        baseline_mode: Baseline correction mode ('percent', 'ratio', 'logratio', 'mean', 'zscore')
        **kwargs: Additional parameters for tfr functions

    Returns:
        AverageTFR object containing power and optionally ITC
    """

    # Generate logarithmically spaced frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    logger.info(f"Computing TFR: {freq_range[0]}-{freq_range[1]} Hz, {n_freqs} frequencies")

    # Set default number of cycles
    if n_cycles is None:
        if method == "morlet":
            n_cycles = freqs / 2.0  # Standard: frequency/2 cycles
        else:
            n_cycles = 4.0  # Fixed cycles for multitaper

    # Compute time-frequency decomposition
    if method == "morlet":
        power, itc = epochs.compute_tfr(
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=compute_itc,
            average=True,  # Return AverageTFR, not EpochsTFR
            n_jobs=1,
            picks='eeg',
            verbose=False,
            **kwargs
        )
    elif method == "multitaper":
        power, itc = epochs.compute_tfr(
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=compute_itc,
            average=True,  # Return AverageTFR, not EpochsTFR
            n_jobs=1,
            verbose=False,
            picks='eeg',
            **kwargs
        )
    else:
        raise ValueError(f"Unknown TFR method: {method}")

    # Apply baseline correction if specified
    if baseline is not None:
        logger.info(f"Applying baseline correction: {baseline}, mode: {baseline_mode}")
        power.apply_baseline(baseline, mode=baseline_mode)


    logger.success(f"TFR analysis complete: {len(freqs)} frequencies, {power.data.shape[-1]} time points")

    # Store ITC in power object metadata if computed
    if compute_itc and itc is not None:
        power.comment = f"ITC computed: {method} method"
        power._itc_data = itc        # Store ITC as an attribute on the power object
        logger.info("ITC data stored with power object")

    return power


def get_frequency_bands() -> dict:
    """
    Standard frequency bands for EEG analysis

    Returns:
        Dictionary of frequency bands with [min, max] ranges
    """
    return {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 100]
    }


def extract_band_power(tfr: AverageTFR, band_name: str = None,
                       freq_range: List[float] = None) -> AverageTFR:
    """
    Extract power in a specific frequency band

    Args:
        tfr: Input AverageTFR object
        band_name: Name of standard band ('theta', 'alpha', etc.)
        freq_range: Custom [min, max] frequency range

    Returns:
        AverageTFR object cropped to frequency band
    """
    if band_name is not None:
        bands = get_frequency_bands()
        if band_name not in bands:
            raise ValueError(f"Unknown band: {band_name}. Available: {list(bands.keys())}")
        freq_range = bands[band_name]

    if freq_range is None:
        raise ValueError("Must specify either band_name or freq_range")

    return tfr.copy().crop(fmin=freq_range[0], fmax=freq_range[1])


def compute_baseline_spectrum(raw: BaseRaw,
                              freq_range: List[float] = [1, 50],
                              n_freqs: int = 20,
                              method: str = "welch",
                              **kwargs) -> Spectrum:
    """
    Compute baseline power spectrum from continuous raw data

    Args:
        raw: Baseline raw data (already segmented)
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequency points
        method: 'welch' or 'multitaper'
        **kwargs: Additional parameters for PSD computation

    Returns:
        Spectrum object (time-averaged power)
    """
    logger.info(f"Computing baseline spectrum: {freq_range[0]}-{freq_range[1]} Hz, method={method}")

    if method == "welch":
        spectrum = raw.compute_psd(
            method='welch',
            fmin=freq_range[0],
            fmax=freq_range[1],
            n_fft=kwargs.get('n_fft', 2048),
            n_overlap=kwargs.get('n_overlap', 1024),
            verbose=False
        )
    elif method == "multitaper":
        spectrum = raw.compute_psd(
            method='multitaper',
            fmin=freq_range[0],
            fmax=freq_range[1],
            bandwidth=kwargs.get('bandwidth', 2.0),
            verbose=False
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'welch' or 'multitaper'")

    logger.success(f"Baseline spectrum computed: {len(spectrum.freqs)} frequencies")
    return spectrum


def compute_raw_tfr(raw: BaseRaw,
                    freq_range: List[float] = [1, 50],
                    n_freqs: int = 20,
                    method: str = "morlet",
                    n_cycles: Union[float, List[float]] = None,
                    decim: int = 8,
                    n_jobs: int = 1,
                    **kwargs):
    """
    Compute continuous time-frequency representation from raw data

    Args:
        raw: Raw EEG data
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequency points
        method: 'morlet' or 'stockwell'
        n_cycles: Cycles per frequency (auto-computed if None)
        decim: Decimation factor to reduce temporal resolution and memory usage
        n_jobs: Number of parallel jobs
        **kwargs: Additional parameters

    Returns:
        Continuous time-frequency power object
    """
    from mne.time_frequency import tfr_morlet, tfr_stockwell

    # Generate logarithmically spaced frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    logger.info(f"Computing continuous TFR: {freq_range[0]}-{freq_range[1]} Hz, {n_freqs} frequencies")
    logger.info(f"Data duration: {raw.times[-1]:.1f}s, method: {method}")

    # Set default number of cycles
    if n_cycles is None:
        if method == "morlet":
            n_cycles = freqs / 2.0  # Standard: frequency/2 cycles
        else:
            n_cycles = 4.0  # Default for other methods

    # Memory usage warning for long recordings
    n_samples = len(raw.times)
    estimated_size_gb = (len(raw.ch_names) * len(freqs) * n_samples * 8) / (1024 ** 3) / decim
    if estimated_size_gb > 2.0:
        logger.warning(f"Large TFR computation: ~{estimated_size_gb:.1f}GB. Consider increasing decim parameter.")

    # Compute continuous time-frequency decomposition directly on raw data
    if method == "morlet":
        raw_tfr = raw.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            decim=decim,
            n_jobs=n_jobs,
            verbose=False,
        )

    elif method == "stockwell":
        raw_tfr = raw.compute_tfr(
            method="stockwell",
            fmin=freq_range[0],
            fmax=freq_range[1],
            n_fft=kwargs.get('n_fft', None),
            width=kwargs.get('width', 1.0),
            decim=decim,
            n_jobs=n_jobs,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'morlet' or 'stockwell'")

    # Convert to power (if complex values returned)
    if np.iscomplexobj(raw_tfr.data):
        raw_tfr.data = np.abs(raw_tfr.data) ** 2

    logger.success(f"Continuous TFR computed: {raw_tfr.data.shape} (channels × freqs × times)")
    logger.info(f"Time resolution: {1 / (raw.info['sfreq'] / decim):.3f}s, Frequency resolution: {len(freqs)} points")

    return raw_tfr


def compute_raw_tfr_average(raw_tfr,
                            method: str = "mean",
                            **kwargs) -> AverageTFR:
    """
    Convert RawTFR to AverageTFR by averaging across time dimension

    Args:
        raw_tfr: RawTFR object from compute_raw_tfr()
        method: Averaging method ('mean', 'median')
        **kwargs: Additional parameters (reserved for future use)

    Returns:
        AverageTFR object with time dimension collapsed
    """
    logger.info(f"Converting RawTFR to AverageTFR: averaging across {raw_tfr.data.shape[-1]} time points")

    # Average across time dimension (last axis)
    if method == "mean":
        averaged_data = np.mean(raw_tfr.data, axis=-1, keepdims=True)
    elif method == "median":
        averaged_data = np.median(raw_tfr.data, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown averaging method: '{method}'. Use 'mean' or 'median'")

    # Create AverageTFR using the proper class method
    averaged_tfr = AverageTFR(
        info=raw_tfr.info,
        data=averaged_data,
        times=np.array([0.0]),
        freqs=raw_tfr.freqs,
        nave=1,
        comment=f"Time-averaged from RawTFR ({method})"
    )

    logger.success(f"RawTFR averaged: {raw_tfr.data.shape} → {averaged_tfr.data.shape}")
    return averaged_tfr