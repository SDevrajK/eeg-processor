"""
Time-frequency analysis module implementing Grandchamp & Delorme (2011)
single-trial baseline normalization.

Reference:
    Grandchamp R, Delorme A (2011) Single-trial normalization for event-related
    spectral decomposition reduces sensitivity to noisy trials.
    Front Psychol 2:236. doi: 10.3389/fpsyg.2011.00236
"""

import gc
import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from mne.time_frequency import AverageTFR, Spectrum
from typing import List, Union, Optional, Tuple
from loguru import logger
from ..utils.memory_tools import memory_profile, get_mne_object_memory


def apply_single_trial_baseline(power_data: np.ndarray,
                               times: np.ndarray,
                               baseline: Tuple[float, float]) -> np.ndarray:
    """
    Apply single-trial baseline correction using z-score normalization.

    Implements Grandchamp & Delorme (2011) Eq. 7-8:
    For each trial k: P_z(f,t) = (P(f,t) - μ_B(f)) / σ_B(f)

    Where μ_B and σ_B are computed from the baseline period for that trial.

    Args:
        power_data: Power data (n_epochs, n_channels, n_freqs, n_times)
        times: Time vector in seconds
        baseline: Baseline period (tmin, tmax) in seconds

    Returns:
        Z-score normalized power data (same shape as input)

    Reference:
        Grandchamp & Delorme (2011) Front. Psychol. 2:236
    """
    # Find baseline time indices
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    baseline_indices = np.where(baseline_mask)[0]

    if len(baseline_indices) == 0:
        raise ValueError(
            f"No baseline samples found in interval {baseline}. "
            f"Time range: [{times[0]:.3f}, {times[-1]:.3f}]"
        )

    logger.debug(f"Baseline: {len(baseline_indices)} time points "
                f"from {times[baseline_indices[0]]:.3f}s to {times[baseline_indices[-1]]:.3f}s")

    # Extract baseline period: (n_epochs, n_channels, n_freqs, n_baseline_times)
    baseline_power = power_data[..., baseline_indices]

    # Compute baseline statistics per trial (Grandchamp & Delorme Eq. 7)
    # Shape: (n_epochs, n_channels, n_freqs, 1)
    baseline_mean = np.mean(baseline_power, axis=-1, keepdims=True)
    baseline_std = np.std(baseline_power, axis=-1, keepdims=True)

    # Prevent division by zero - use 1.0 if std is zero (no variance in baseline)
    baseline_std = np.where(baseline_std == 0, 1.0, baseline_std)

    # Apply z-score normalization (Grandchamp & Delorme Eq. 8)
    corrected_data = (power_data - baseline_mean) / baseline_std

    return corrected_data


@memory_profile
def compute_epochs_tfr_average(epochs: Epochs,
                        freq_range: List[float] = [1, 50],
                        n_freqs: int = 100,
                        method: str = "morlet",
                        n_cycles: Union[float, np.ndarray, None] = None,
                        compute_itc: bool = True,
                        baseline: Optional[Tuple[float, float]] = None,
                        **kwargs) -> AverageTFR:
    """
    Compute averaged time-frequency representation with single-trial baseline correction.

    Implements Grandchamp & Delorme (2011) procedure:
    1. Compute TFR for each trial (complex representation)
    2. Convert to power: P[k,f,t] = |TFR[k,f,t]|²
    3. Apply z-score baseline to each trial: P_z[k,f,t] = (P[k,f,t] - μ_B[k,f]) / σ_B[k,f]
    4. Average across trials: ERSP[f,t] = mean(P_z[k,f,t])

    Args:
        epochs: Input epochs
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of logarithmically-spaced frequencies
        method: 'morlet' or 'multitaper'
        n_cycles: Cycles per frequency (auto: freq/2 for morlet, 4.0 for multitaper)
        compute_itc: Whether to compute inter-trial coherence
        baseline: Baseline period (tmin, tmax) for z-score correction (required)
        **kwargs: Additional parameters for epochs.compute_tfr()

    Returns:
        AverageTFR object with baseline-corrected power

    Reference:
        Grandchamp & Delorme (2011) Front. Psychol. 2:236
        doi: 10.3389/fpsyg.2011.00236
    """
    if baseline is None:
        raise ValueError(
            "Baseline period is required for single-trial normalization. "
            "Provide baseline=(tmin, tmax) in seconds."
        )

    # Generate logarithmically spaced frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    logger.info(f"Computing TFR: {freq_range[0]}-{freq_range[1]} Hz, {n_freqs} frequencies")

    # Set default number of cycles
    if n_cycles is None:
        if method == "morlet":
            n_cycles = freqs / 2.0  # Standard: frequency/2 cycles
        else:
            n_cycles = 4.0  # Fixed cycles for multitaper

    logger.info(f"Single-trial baseline correction (z-score): {baseline}")

    # Step 1: Compute TFR with complex output
    logger.info(f"Computing {method} TFR (complex)...")
    epochs_tfr = epochs.compute_tfr(
        method=method,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,  # Get EpochsTFR
        output='complex',
        n_jobs=4,
        picks='eeg',
        verbose='INFO',
        **kwargs
    )

    # Step 2: Convert to power
    logger.info("Converting to power...")
    complex_data = epochs_tfr.data  # (n_epochs, n_channels, n_freqs, n_times)
    power_data = np.abs(complex_data) ** 2
    logger.debug(f"Power shape: {power_data.shape}")

    # Step 3: Apply single-trial baseline correction (z-score)
    logger.info("Applying single-trial z-score baseline correction...")
    corrected_power = apply_single_trial_baseline(
        power_data,
        epochs_tfr.times,
        baseline
    )

    # Step 4: Average across trials
    logger.info("Averaging across trials...")
    averaged_power = np.mean(corrected_power, axis=0)  # (n_channels, n_freqs, n_times)

    # Compute ITC if requested
    itc = None
    if compute_itc:
        logger.info("Computing inter-trial coherence...")
        epsilon = np.finfo(float).eps
        # Normalize complex values
        complex_abs = np.abs(complex_data)
        normalized_complex = complex_data / (complex_abs + epsilon)
        # ITC is magnitude of mean normalized complex
        itc_data = np.abs(np.mean(normalized_complex, axis=0))

        # Create ITC object
        itc = epochs_tfr.copy().average()
        itc.data = itc_data
        itc.comment = "Inter-trial coherence"

    # Create AverageTFR object
    power = epochs_tfr.copy().average()
    power.data = averaged_power
    power.comment = f"Single-trial baseline: z-score {baseline}"

    # Store ITC if computed
    if compute_itc and itc is not None:
        power._itc_data = itc
        logger.info("ITC stored with power object")

    # Clean up
    del complex_data, power_data, corrected_power
    gc.collect()

    logger.success(f"TFR complete: {len(freqs)} frequencies, {power.data.shape[-1]} time points")

    return power


# Keep other functions unchanged
def get_frequency_bands() -> dict:
    """Standard frequency bands for EEG analysis"""
    return {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 100]
    }


def extract_band_power(tfr: AverageTFR, band_name: Optional[str] = None,
                       freq_range: Optional[List[float]] = None) -> AverageTFR:
    """Extract power in a specific frequency band"""
    if band_name is not None:
        bands = get_frequency_bands()
        if band_name not in bands:
            raise ValueError(f"Unknown band: {band_name}. Available: {list(bands.keys())}")
        freq_range = bands[band_name]

    if freq_range is None:
        raise ValueError("Must specify either band_name or freq_range")

    return tfr.copy().crop(fmin=freq_range[0], fmax=freq_range[1])


@memory_profile
def compute_baseline_spectrum(raw: BaseRaw,
                              freq_range: List[float] = [1, 50],
                              method: str = "welch",
                              **kwargs) -> Spectrum:
    """Compute baseline power spectrum from continuous raw data"""
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


@memory_profile
def compute_raw_tfr(raw: BaseRaw,
                    freq_range: List[float] = [1, 50],
                    n_freqs: int = 20,
                    method: str = "morlet",
                    n_cycles: Union[float, np.ndarray, None] = None,
                    decim: int = 8,
                    n_jobs: int = 1,
                    **kwargs):
    """
    Compute continuous time-frequency representation from raw data.

    Note: This function is for continuous (non-epoched) data and does not
    use single-trial baseline correction.
    """
    # Generate logarithmically spaced frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    logger.info(f"Computing continuous TFR: {freq_range[0]}-{freq_range[1]} Hz")

    # Set default number of cycles
    if n_cycles is None:
        n_cycles = freqs / 2.0 if method == "morlet" else 4.0

    # Memory analysis
    n_samples = len(raw.times)
    n_channels = len(raw.ch_names)
    input_memory = get_mne_object_memory(raw)

    output_samples = n_samples // decim
    estimated_output_gb = (n_channels * len(freqs) * output_samples * 8) / (1024 ** 3)

    logger.info(f"TFR Memory: Input {input_memory['total_mb']:.1f} MB → Output ~{estimated_output_gb:.2f} GB")

    if estimated_output_gb > 4.0:
        logger.error(f"Very large TFR: ~{estimated_output_gb:.1f}GB. Increase decim parameter!")
    elif estimated_output_gb > 2.0:
        logger.warning(f"Large TFR: ~{estimated_output_gb:.1f}GB. Consider increasing decim.")

    # Compute TFR
    raw_tfr = raw.compute_tfr(
        method=method,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        decim=decim,
        n_jobs=n_jobs,
        verbose=False,
    )

    # Convert to power if complex
    if np.iscomplexobj(raw_tfr.data):
        raw_tfr.data = np.abs(raw_tfr.data) ** 2

    logger.success(f"Continuous TFR computed: {raw_tfr.data.shape}")
    return raw_tfr


def compute_raw_tfr_average(raw_tfr,
                            method: str = "mean",
                            **kwargs) -> AverageTFR:
    """Convert RawTFR to AverageTFR by averaging across time"""
    logger.info(f"Averaging RawTFR across {raw_tfr.data.shape[-1]} time points")

    # Average across time
    if method == "mean":
        averaged_data = np.mean(raw_tfr.data, axis=-1, keepdims=True)
    elif method == "median":
        averaged_data = np.median(raw_tfr.data, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'mean' or 'median'")

    # Create AverageTFR
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
