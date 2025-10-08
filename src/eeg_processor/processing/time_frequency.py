import gc
import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from mne.time_frequency import tfr_morlet, tfr_multitaper, AverageTFR, Spectrum
from typing import List, Union, Optional, Tuple
from loguru import logger
from ..utils.memory_tools import memory_profile, get_mne_object_memory
from ..utils.performance import with_heartbeat


def apply_single_trial_baseline(power_data: np.ndarray,
                               times: np.ndarray,
                               baseline: Tuple[float, float],
                               mode: str = 'logratio') -> np.ndarray:
    """
    Apply baseline correction to each trial individually using optimized operations.

    Args:
        power_data: Power data (n_epochs, n_channels, n_freqs, n_times)
        times: Time vector
        baseline: Baseline period (tmin, tmax)
        mode: Baseline correction mode

    Returns:
        Baseline-corrected power data
    """
    # Find baseline time indices
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    baseline_indices = np.where(baseline_mask)[0].astype(int)

    if len(baseline_indices) == 0:
        raise ValueError(f"No baseline samples found in interval {baseline}")

    # Memory-efficient approach: avoid creating large intermediate arrays
    # Estimate memory usage
    n_epochs, n_channels, n_freqs, n_times = power_data.shape
    data_size_gb = power_data.nbytes / (1024**3)

    # Use vectorized approach for small datasets, chunked approach for large ones
    if data_size_gb < 0.5:  # Less than 0.5GB - use full vectorization (reduced from 1.0GB)
        # Extract baseline data: (n_epochs, n_channels, n_freqs, n_baseline_times)
        baseline_power = power_data[..., baseline_indices]

        # Calculate baseline statistics across all trials simultaneously
        baseline_mean = np.mean(baseline_power, axis=-1, keepdims=True)
        epsilon = np.finfo(float).eps

        if mode == 'logratio':
            ratio = power_data / (baseline_mean + epsilon)
            corrected_data = 10 * np.log10(ratio + epsilon)
        elif mode == 'percent':
            corrected_data = ((power_data - baseline_mean) / (baseline_mean + epsilon)) * 100
        elif mode == 'zscore':
            baseline_std = np.std(baseline_power, axis=-1, keepdims=True)
            baseline_std = np.where(baseline_std == 0, 1, baseline_std)
            corrected_data = (power_data - baseline_mean) / baseline_std
        elif mode == 'mean':
            corrected_data = power_data - baseline_mean
        elif mode == 'ratio':
            corrected_data = power_data / (baseline_mean + epsilon)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

    else:
        # Large dataset - use memory-efficient trial-wise processing
        corrected_data = np.empty_like(power_data)
        epsilon = np.finfo(float).eps

        # Process trials in chunks to balance memory and speed
        # More conservative chunk sizing for better memory management
        chunk_size = max(1, min(20, int(0.5 / (data_size_gb / n_epochs))))  # Adaptive chunk size based on memory per epoch

        for start_idx in range(0, n_epochs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_epochs)
            chunk = power_data[start_idx:end_idx]

            # Extract baseline for this chunk
            baseline_power = chunk[..., baseline_indices]
            baseline_mean = np.mean(baseline_power, axis=-1, keepdims=True)

            if mode == 'logratio':
                ratio = chunk / (baseline_mean + epsilon)
                corrected_data[start_idx:end_idx] = 10 * np.log10(ratio + epsilon)
            elif mode == 'percent':
                corrected_data[start_idx:end_idx] = ((chunk - baseline_mean) /
                                                   (baseline_mean + epsilon)) * 100
            elif mode == 'zscore':
                baseline_std = np.std(baseline_power, axis=-1, keepdims=True)
                baseline_std = np.where(baseline_std == 0, 1, baseline_std)
                corrected_data[start_idx:end_idx] = (chunk - baseline_mean) / baseline_std
            elif mode == 'mean':
                corrected_data[start_idx:end_idx] = chunk - baseline_mean
            elif mode == 'ratio':
                corrected_data[start_idx:end_idx] = chunk / (baseline_mean + epsilon)
            else:
                raise ValueError(f"Unknown baseline mode: {mode}")

    return corrected_data


@memory_profile
def compute_epochs_tfr_average(epochs: Epochs,
                        freq_range: List[float] = [1, 50],
                        n_freqs: int = 100,
                        method: str = "morlet",
                        n_cycles: Union[float, List[float]] = None,
                        compute_itc: bool = True,
                        compute_complex_average: bool = False,
                        baseline: Optional[Tuple[float, float]] = None,
                        baseline_mode: str = "logratio",
                        single_trial_baseline: bool = True,
                        memory_limit_gb: float = 2.0,
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
        compute_complex_average: Whether to compute and store complex average (needed for evoked/induced analysis)
        baseline: Baseline period (tmin, tmax) for correction
        baseline_mode: Baseline correction mode ('logratio', 'percent', 'zscore', 'mean', 'ratio')
        single_trial_baseline: If True, apply baseline correction per trial before averaging (recommended)
        memory_limit_gb: Memory limit in GB for automatic chunking (default: 2.0)
        **kwargs: Additional parameters for tfr functions

    Returns:
        AverageTFR object containing power and optionally ITC and complex average
    """

    # Generate logarithmically spaced frequencies
    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freqs)
    logger.info(f"Computing TFR: {freq_range[0]}-{freq_range[1]} Hz, {n_freqs} frequencies")

    # Memory estimation for single-trial baseline correction
    use_single_trial = single_trial_baseline and baseline is not None
    if use_single_trial:
        n_epochs, n_channels, n_times = len(epochs), len(epochs.ch_names), len(epochs.times)
    
        # Estimate memory usage for complex data (16 bytes per complex128 value)
        complex_data_gb = (n_epochs * n_channels * n_freqs * n_times * 16) / (1024**3)
        power_data_gb = (n_epochs * n_channels * n_freqs * n_times * 8) / (1024**3)  # 8 bytes for float64
        total_estimated_gb = complex_data_gb + power_data_gb
    
        logger.info(f"Memory Analysis for Single-Trial Baseline:")
        logger.info(f"  Epochs shape: ({n_epochs}, {n_channels}, {n_times})")
        logger.info(f"  TFR shape will be: ({n_epochs}, {n_channels}, {n_freqs}, {n_times})")
        logger.info(f"  Estimated complex data: {complex_data_gb:.2f} GB")
        logger.info(f"  Estimated power data: {power_data_gb:.2f} GB")
        logger.info(f"  Total estimated memory: {total_estimated_gb:.2f} GB")
    
        # Determine optimal chunk size based on memory limit
        if total_estimated_gb > memory_limit_gb:
            # Calculate chunk size to stay within memory limit
            memory_per_epoch = total_estimated_gb / n_epochs
            optimal_chunk_size = max(1, int(memory_limit_gb / memory_per_epoch))
            logger.warning(f"Large dataset detected! Using chunked processing:")
            logger.warning(f"  Memory limit: {memory_limit_gb:.1f} GB")
            logger.warning(f"  Chunk size: {optimal_chunk_size} epochs")
            logger.warning(f"  Number of chunks: {int(np.ceil(n_epochs / optimal_chunk_size))}")
        else:
            optimal_chunk_size = n_epochs  # Process all at once
            logger.info(f"Memory usage within limits - processing all {n_epochs} epochs at once")
    else:
        optimal_chunk_size = None  # Not used for traditional pipeline
    
    # Set default number of cycles
    if n_cycles is None:
        if method == "morlet":
            n_cycles = freqs / 2.0  # Standard: frequency/2 cycles
        else:
            n_cycles = 4.0  # Fixed cycles for multitaper
    
    # Determine if we should use single-trial baseline correction
    use_single_trial = single_trial_baseline and baseline is not None
    
    # Compute time-frequency decomposition
    if method == "morlet":
        # Wrap compute_tfr with heartbeat decorator for progress monitoring
        wrapped_compute_tfr = with_heartbeat(
            interval=5,
            message=f"Computing Morlet TFR ({freq_range[0]}-{freq_range[1]} Hz)"
        )(epochs.compute_tfr)
    
        if use_single_trial:
            # Get EpochsTFR with complex output for single-trial correction
            epochs_tfr = wrapped_compute_tfr(
                method=method,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,  # We'll compute ITC manually
                average=False,  # Get EpochsTFR, not AverageTFR
                output='complex',  # Get complex values
                n_jobs=4,
                picks='eeg',
                verbose='INFO',
                **kwargs
            )
        else:
            # Original implementation for backward compatibility
            power, itc = wrapped_compute_tfr(
                method=method,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=compute_itc,
                average=True,  # Return AverageTFR, not EpochsTFR
                n_jobs=4,
                picks='eeg',
                verbose='INFO',
                **kwargs
            )
    
    elif method == "multitaper":
        # Wrap compute_tfr with heartbeat decorator for progress monitoring
        wrapped_compute_tfr = with_heartbeat(
            interval=5,
            message=f"Computing Multitaper TFR ({freq_range[0]}-{freq_range[1]} Hz)"
        )(epochs.compute_tfr)
    
        if use_single_trial:
            # Get EpochsTFR with complex output for single-trial correction
            epochs_tfr = wrapped_compute_tfr(
                method=method,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,  # We'll compute ITC manually
                average=False,  # Get EpochsTFR, not EpochsTFR
                output='complex',  # Get complex values
                n_jobs=4,
                verbose='INFO',
                picks='eeg',
                **kwargs
            )
        else:
            # Original implementation for backward compatibility
            power, itc = wrapped_compute_tfr(
                method=method,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=compute_itc,
                average=True,  # Return AverageTFR, not EpochsTFR
                n_jobs=4,
                verbose='INFO',
                picks='eeg',
                **kwargs
            )
    else:
        raise ValueError(f"Unknown TFR method: {method}")
    
    # Handle single-trial baseline correction with memory-efficient chunked processing
    if use_single_trial:
        logger.info(f"Applying single-trial baseline correction: {baseline}, mode: {baseline_mode}")
    
        # Get full complex data shape for initialization
        complex_data = epochs_tfr.data  # (n_epochs, n_channels, n_freqs, n_times)
        n_epochs, n_channels, n_freqs, n_times = complex_data.shape
    
        # Initialize accumulators for chunked processing
        averaged_power = np.zeros((n_channels, n_freqs, n_times), dtype=np.float64)
        complex_average = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128) if compute_complex_average else None
        itc_accumulator = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128) if compute_itc else None
    
        # Process in chunks to manage memory usage
        n_processed = 0
        chunk_size_int = optimal_chunk_size if optimal_chunk_size is not None else n_epochs
        for start_idx in range(0, n_epochs, chunk_size_int):
            end_idx = min(start_idx + chunk_size_int, n_epochs)
            chunk_size = end_idx - start_idx
    
            logger.info(f"Processing chunk {start_idx//chunk_size_int + 1}/{int(np.ceil(n_epochs / chunk_size_int))}: epochs {start_idx}-{end_idx-1}")
    
            # Extract chunk of complex data
            chunk_complex = complex_data[start_idx:end_idx]  # (chunk_size, n_channels, n_freqs, n_times)
    
            # Accumulate complex average if requested
            if compute_complex_average:
                complex_average += np.sum(chunk_complex, axis=0)  # Sum across trials in chunk
    
            # Accumulate ITC if requested
            if compute_itc:
                epsilon = np.finfo(float).eps
                # Normalize complex values and accumulate
                chunk_abs = np.abs(chunk_complex)
                normalized_chunk = chunk_complex / (chunk_abs + epsilon)
                itc_accumulator += np.sum(normalized_chunk, axis=0)  # Sum normalized complex values
    
            # Convert to power for baseline correction
            chunk_power = np.abs(chunk_complex) ** 2
    
            # Apply baseline correction to chunk
            corrected_chunk_power = apply_single_trial_baseline(
                chunk_power, epochs_tfr.times, baseline, baseline_mode  # type: ignore
            )
    
            # Accumulate corrected power (sum across trials in chunk)
            averaged_power += np.sum(corrected_chunk_power, axis=0)
            n_processed += chunk_size
    
            # Free memory immediately and force garbage collection
            del chunk_complex, chunk_power, corrected_chunk_power
            if compute_itc and 'chunk_abs' in locals():
                del chunk_abs, normalized_chunk
            gc.collect()  # Force garbage collection to free memory immediately
    
        # Finalize averages
        averaged_power /= n_epochs  # Convert sum to average
    
        if compute_complex_average:
            complex_average /= n_epochs  # Convert sum to average
            logger.info("Complex average computed using chunked processing")
    
        if compute_itc:
            # Finalize ITC computation
            itc_data = np.abs(itc_accumulator / n_epochs)  # Convert sum to average, then magnitude
    
            # Create ITC AverageTFR object
            itc = epochs_tfr.copy()
            itc = itc.average()  # Convert EpochsTFR to AverageTFR
            itc.data = itc_data
            itc.comment = "Inter-trial coherence (chunked)"
            logger.info("ITC computed using chunked processing")
            del itc_accumulator
    
        # Create AverageTFR object from epochs_tfr
        power = epochs_tfr.copy()
        power = power.average()  # Convert EpochsTFR to AverageTFR
        power.data = averaged_power  # Replace with baseline-corrected averaged data
        power.comment = f"Single-trial baseline: {baseline_mode} (chunked: {chunk_size_int} epochs)"
    
        # Free the large complex_data array
        del complex_data
        logger.success(f"Chunked single-trial baseline correction complete: {n_processed} epochs processed")
    
    else:
        # Traditional post-averaging baseline correction
        # But we may still need to compute complex average if requested
        complex_average = None
        if compute_complex_average:
            # We need to get complex data even for traditional pipeline
            # This requires a separate computation without single-trial processing
            if method == "morlet":
                epochs_tfr_for_complex = with_heartbeat(
                    interval=5,
                    message=f"Computing complex data for complex average"
                )(epochs.compute_tfr)(
                    method=method,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=False,
                    output='complex',
                    n_jobs=4,
                    picks='eeg',
                    verbose='INFO',
                    **kwargs
                )
            elif method == "multitaper":
                epochs_tfr_for_complex = with_heartbeat(
                    interval=5,
                    message=f"Computing complex data for complex average"
                )(epochs.compute_tfr)(
                    method=method,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=False,
                    output='complex',
                    n_jobs=4,
                    picks='eeg',
                    verbose='INFO',
                    **kwargs
                )
    
            complex_average = np.mean(epochs_tfr_for_complex.data, axis=0)
            logger.info("Complex average computed for traditional pipeline")
    
        if baseline is not None:
            logger.info(f"Applying baseline correction: {baseline}, mode: {baseline_mode}")
            power.apply_baseline(baseline, mode=baseline_mode)
    
    logger.success(f"TFR analysis complete: {len(freqs)} frequencies, {power.data.shape[-1]} time points")
    
    # Store ITC in power object metadata if computed
    if compute_itc and itc is not None:
        power.comment = f"ITC computed: {method} method"
        power._itc_data = itc        # Store ITC as an attribute on the power object
        logger.info("ITC data stored with power object")
    
        # Store complex average in power object if computed
        if compute_complex_average and 'complex_average' in locals() and complex_average is not None:
            power._complex_average = complex_average  # Store complex average as an attribute
            logger.info("Complex average stored with power object")
    
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


@memory_profile
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


@memory_profile
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

    # Enhanced memory analysis for TFR computation
    n_samples = len(raw.times)
    n_channels = len(raw.ch_names)
    input_memory = get_mne_object_memory(raw)
    
    # Detailed memory estimation
    bytes_per_sample = 8  # float64
    output_samples = n_samples // decim
    estimated_output_bytes = n_channels * len(freqs) * output_samples * bytes_per_sample
    estimated_size_gb = estimated_output_bytes / (1024 ** 3)
    
    # Memory efficiency analysis
    memory_multiplier = estimated_output_bytes / (input_memory['data_mb'] * 1024 * 1024) if input_memory['data_mb'] > 0 else 0
    
    logger.info(f"TFR Memory Analysis:")
    logger.info(f"  Input data: {input_memory['total_mb']:.1f} MB ({input_memory['data_shape']})")
    logger.info(f"  Estimated output: {estimated_size_gb:.2f} GB ({n_channels}×{len(freqs)}×{output_samples})")
    logger.info(f"  Memory multiplier: {memory_multiplier:.1f}x")
    logger.info(f"  Decimation factor: {decim} (reduces size by {decim}x)")
    
    # Enhanced warnings
    if estimated_size_gb > 4.0:
        logger.error(f"Very large TFR computation: ~{estimated_size_gb:.1f}GB. Strongly recommend increasing decim parameter.")
        logger.error(f"  Suggestions: decim={decim*2} → {estimated_size_gb/2:.1f}GB, decim={decim*4} → {estimated_size_gb/4:.1f}GB")
    elif estimated_size_gb > 2.0:
        logger.warning(f"Large TFR computation: ~{estimated_size_gb:.1f}GB. Consider increasing decim parameter.")
        logger.warning(f"  Suggestion: decim={decim*2} → {estimated_size_gb/2:.1f}GB")
    
    if memory_multiplier > 20:
        logger.warning(f"High memory multiplier ({memory_multiplier:.1f}x) - consider reducing n_freqs or increasing decim")

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

    # Analyze actual memory usage vs estimation
    output_memory = get_mne_object_memory(raw_tfr)
    actual_size_gb = output_memory['total_mb'] / 1024
    estimation_accuracy = actual_size_gb / estimated_size_gb if estimated_size_gb > 0 else 1.0
    actual_memory_multiplier = output_memory['total_mb'] / input_memory['total_mb'] if input_memory['total_mb'] > 0 else 1.0

    logger.success(f"Continuous TFR computed: {raw_tfr.data.shape} (channels × freqs × times)")
    logger.info(f"Time resolution: {1 / (raw.info['sfreq'] / decim):.3f}s, Frequency resolution: {len(freqs)} points")
    logger.info(f"Memory Analysis Results:")
    logger.info(f"  Actual output size: {actual_size_gb:.2f} GB")
    logger.info(f"  Estimation accuracy: {estimation_accuracy:.2f}x {'(overestimated)' if estimation_accuracy < 0.8 else '(underestimated)' if estimation_accuracy > 1.2 else '(accurate)'}")
    logger.info(f"  Actual memory multiplier: {actual_memory_multiplier:.1f}x")
    
    # Store memory analysis on TFR object
    raw_tfr._memory_analysis = {
        'input_memory': input_memory,
        'output_memory': output_memory,
        'estimated_size_gb': estimated_size_gb,
        'actual_size_gb': actual_size_gb,
        'estimation_accuracy': estimation_accuracy,
        'memory_multiplier': actual_memory_multiplier,
        'decimation_factor': decim,
        'frequency_count': len(freqs)
    }

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