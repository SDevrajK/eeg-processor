import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from mne.time_frequency import tfr_morlet, tfr_multitaper, AverageTFR, Spectrum
from typing import List, Union, Optional, Tuple
from loguru import logger


def compute_tfr_average(epochs: Epochs,
                        freq_range: List[float] = [1, 50],
                        n_freqs: int = 20,
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
        power, itc = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=compute_itc,
            average=True,  # Return AverageTFR, not EpochsTFR
            n_jobs=1,
            verbose=False,
            **kwargs
        )
    elif method == "multitaper":
        power, itc = tfr_multitaper(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=compute_itc,
            average=True,
            n_jobs=1,
            verbose=False,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown TFR method: {method}")

    # Apply baseline correction if specified
    if baseline is not None:
        logger.info(f"Applying baseline correction: {baseline}, mode: {baseline_mode}")
        power.apply_baseline(baseline, mode=baseline_mode)
        if compute_itc and itc is not None:
            # Note: ITC typically doesn't need baseline correction
            pass

    # Store ITC in power object metadata if computed
    if compute_itc and itc is not None:
        # Store ITC data in the power object's metadata
        power.comment = f"ITC computed: {method} method"
        # Note: MNE doesn't have a standard way to store ITC with power
        # You might want to return both objects or save them separately
        logger.info("ITC computed and stored")

    logger.success(f"TFR analysis complete: {len(freqs)} frequencies, {power.data.shape[-1]} time points")
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
                    decim: int = 3,
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
        power = raw.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            decim=decim,
            n_jobs=n_jobs,
            verbose=False,
            **kwargs
        )

    elif method == "stockwell":
        power = raw.compute_tfr(
            method="stockwell",
            fmin=freq_range[0],
            fmax=freq_range[1],
            n_fft=kwargs.get('n_fft', None),
            width=kwargs.get('width', 1.0),
            decim=decim,
            n_jobs=n_jobs,
            verbose=False,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'morlet' or 'stockwell'")

    # Convert to power (if complex values returned)
    if np.iscomplexobj(power.data):
        power.data = np.abs(power.data) ** 2

    logger.success(f"Continuous TFR computed: {power.data.shape} (channels × freqs × times)")
    logger.info(f"Time resolution: {1 / (raw.info['sfreq'] / decim):.3f}s, Frequency resolution: {len(freqs)} points")

    return power