import gc
import numpy as np
from mne import Epochs
from mne.io import BaseRaw
from mne.time_frequency import tfr_morlet, tfr_multitaper, AverageTFR, Spectrum
from typing import List, Union, Optional, Tuple
from loguru import logger
from ..utils.memory_tools import memory_profile, get_mne_object_memory


def apply_single_trial_baseline(power_data: np.ndarray,
                               times: np.ndarray,
                               baseline: Tuple[float, float]) -> np.ndarray:
    """
    Apply single-trial baseline correction using z-score normalization.

    Implements the method from Grandchamp & Delorme (2011):
    "Single-trial normalization for event-related spectral decomposition
    reduces sensitivity to noisy trials"

    For each trial independently:
    1. Compute baseline mean and std from baseline time window
    2. Z-score normalize: (power - baseline_mean) / baseline_std

    Args:
        power_data: Power data (n_epochs, n_channels, n_freqs, n_times)
        times: Time vector
        baseline: Baseline period (tmin, tmax) in seconds

    Returns:
        Z-score normalized power data (same shape as input)

    Reference:
        Grandchamp & Delorme (2011) Front. Psychol. 2:236
        doi: 10.3389/fpsyg.2011.00236
    """
    # Find baseline time indices
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    baseline_indices = np.where(baseline_mask)[0]

    if len(baseline_indices) == 0:
        raise ValueError(
            f"No baseline samples found in interval {baseline}. "
            f"Time range: [{times[0]:.3f}, {times[-1]:.3f}]"
        )

    logger.debug(f"Baseline correction: {len(baseline_indices)} time points "
                f"from {times[baseline_indices[0]]:.3f}s to {times[baseline_indices[-1]]:.3f}s")

    # Extract baseline period: (n_epochs, n_channels, n_freqs, n_baseline_times)
    baseline_power = power_data[..., baseline_indices]

    # Compute baseline statistics per trial
    # Shape: (n_epochs, n_channels, n_freqs, 1)
    baseline_mean = np.mean(baseline_power, axis=-1, keepdims=True)
    baseline_std = np.std(baseline_power, axis=-1, keepdims=True)

    # Robust protection against division issues:
    # 1. Replace NaN/Inf in mean with 0
    # 2. Replace std < epsilon or NaN/Inf with 1.0 (no normalization)
    epsilon = 1e-10
    baseline_mean = np.nan_to_num(baseline_mean, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_std = np.where((baseline_std < epsilon) | ~np.isfinite(baseline_std), 1.0, baseline_std)

    # Apply z-score normalization to each trial
    # Shape: (n_epochs, n_channels, n_freqs, n_times)
    corrected_data = (power_data - baseline_mean) / baseline_std

    # Final safety check: replace any remaining NaN/Inf values
    # This should not happen with the protections above, but ensures robustness
    n_invalid = np.sum(~np.isfinite(corrected_data))
    if n_invalid > 0:
        logger.warning(f"Found {n_invalid} NaN/Inf values in baseline-corrected data - replacing with 0")
        corrected_data = np.nan_to_num(corrected_data, nan=0.0, posinf=0.0, neginf=0.0)

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
                        single_trial_baseline: bool = True,
                        memory_limit_gb: Optional[float] = None,
                        **kwargs) -> AverageTFR:
    """
    Compute averaged time-frequency representation from epochs.

    Implements single-trial baseline correction using z-score normalization
    as described in Grandchamp & Delorme (2011).

    Args:
        epochs: Input epochs
        freq_range: [min_freq, max_freq] in Hz
        n_freqs: Number of frequency points
        method: 'morlet' or 'multitaper'
        n_cycles: Cycles per frequency (auto-computed if None)
        compute_itc: Whether to compute inter-trial coherence
        compute_complex_average: Whether to compute and store complex average
        baseline: Baseline period (tmin, tmax) for z-score correction
        single_trial_baseline: If True, apply z-score baseline per trial before averaging
                               If False, compute average first then apply MNE's baseline
        memory_limit_gb: Memory limit in GB for chunking (None = auto-detect)
        **kwargs: Additional parameters for tfr functions

    Returns:
        AverageTFR object containing power and optionally ITC

    Reference:
        Grandchamp & Delorme (2011) Front. Psychol. 2:236
        doi: 10.3389/fpsyg.2011.00236
    """

    # Auto-detect memory limit if not specified
    if memory_limit_gb is None:
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            # Use 25% of available memory as default limit (conservative)
            memory_limit_gb = max(2.0, available_gb * 0.25)
            logger.info(f"Auto-detected memory limit: {memory_limit_gb:.1f} GB (25% of {available_gb:.1f} GB available)")
        except ImportError:
            # Fallback to conservative 2GB if psutil not available
            memory_limit_gb = 2.0
            logger.warning("psutil not available for memory detection. Using default 2.0 GB limit.")
            logger.warning("Install psutil for automatic memory detection: pip install psutil")

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
            # We need to account for BOTH complex data AND power data being in memory simultaneously
            # During processing: complex_chunk + power_chunk + baseline_corrected = 3x memory per epoch
            # Use conservative calculation to avoid memory errors

            # Use 50% of memory limit for safety (Windows per-process limits, overhead, etc.)
            usable_memory_gb = memory_limit_gb * 0.5

            # Memory per epoch needs to account for temporary arrays during processing
            # Complex (16 bytes) + Power (8 bytes) + Corrected (8 bytes) = 32 bytes per element
            # But we use complex_data_gb which is already calculated correctly
            # Multiply by 2 to account for power array creation
            memory_per_epoch_gb = (complex_data_gb / n_epochs) * 2.0

            # Calculate chunk size
            optimal_chunk_size = max(1, int(usable_memory_gb / memory_per_epoch_gb))

            # Safety check: don't exceed total epochs
            optimal_chunk_size = min(optimal_chunk_size, n_epochs)

            logger.warning(f"Large dataset detected! Using chunked processing:")
            logger.warning(f"  Memory limit: {memory_limit_gb:.1f} GB (using {usable_memory_gb:.1f} GB for chunk data)")
            logger.warning(f"  Memory per epoch (complex + power): {memory_per_epoch_gb:.3f} GB")
            logger.warning(f"  Chunk size: {optimal_chunk_size} epochs")
            logger.warning(f"  Number of chunks: {int(np.ceil(n_epochs / optimal_chunk_size))}")
            logger.warning(f"  Estimated memory per chunk: {optimal_chunk_size * memory_per_epoch_gb:.1f} GB")
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

    # For large datasets with single-trial baseline, compute TFR in chunks to avoid memory errors
    if use_single_trial and optimal_chunk_size is not None and optimal_chunk_size < len(epochs):
        logger.warning(f"Large dataset requires chunked TFR computation")
        logger.warning(f"Will process {len(epochs)} epochs in chunks of {optimal_chunk_size}")

        # We'll compute TFR chunk-by-chunk and accumulate results
        # This avoids the memory error during TFR computation
        epochs_tfr = None  # Will build this incrementally

    else:
        # Standard path: compute TFR for all epochs at once
        # Compute time-frequency decomposition with detailed error handling
        try:
            if method == "morlet":
                logger.info(f"Computing Morlet TFR ({freq_range[0]}-{freq_range[1]} Hz)...")
                logger.debug(f"TFR parameters: n_cycles={n_cycles}, use_fft=True, n_jobs=4")

                if use_single_trial:
                    # Get EpochsTFR with complex output for single-trial correction
                    logger.debug(f"Computing complex TFR for single-trial baseline correction...")
                    epochs_tfr = epochs.compute_tfr(
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
                    logger.debug(f"TFR computation complete. Shape: {epochs_tfr.data.shape}")
                else:
                    # Original implementation for backward compatibility
                    logger.debug(f"Computing averaged TFR (traditional pipeline)...")
                    power, itc = epochs.compute_tfr(
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
                    logger.debug(f"TFR computation complete. Power shape: {power.data.shape}")

            elif method == "multitaper":
                logger.info(f"Computing Multitaper TFR ({freq_range[0]}-{freq_range[1]} Hz)...")
                logger.debug(f"TFR parameters: n_cycles={n_cycles}, use_fft=True, n_jobs=4")

                if use_single_trial:
                    # Get EpochsTFR with complex output for single-trial correction
                    logger.debug(f"Computing complex TFR for single-trial baseline correction...")
                    epochs_tfr = epochs.compute_tfr(
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
                    logger.debug(f"TFR computation complete. Shape: {epochs_tfr.data.shape}")
                else:
                    # Original implementation for backward compatibility
                    logger.debug(f"Computing averaged TFR (traditional pipeline)...")
                    power, itc = epochs.compute_tfr(
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
                    logger.debug(f"TFR computation complete. Power shape: {power.data.shape}")
            else:
                raise ValueError(f"Unknown TFR method: {method}")

        except Exception as e:
            logger.error(f"TFR computation failed: {type(e).__name__}: {e}")
            logger.error(f"Error occurred during {method} TFR computation")
            logger.error(f"Epochs info: {len(epochs)} epochs, {len(epochs.ch_names)} channels, {len(epochs.times)} time points")
            logger.error(f"Frequency info: {len(freqs)} frequencies from {freqs[0]:.2f} to {freqs[-1]:.2f} Hz")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    # Handle single-trial baseline correction with memory-efficient chunked processing
    if use_single_trial:
        try:
            logger.info(f"Applying single-trial baseline correction: {baseline}, mode: z-score")

            # Check if we need to compute TFR in chunks (for very large datasets)
            if epochs_tfr is None:
                # Chunked TFR computation path - compute and process chunk-by-chunk
                logger.info("Using fully chunked TFR + baseline pipeline to avoid memory errors")
                n_epochs_total = len(epochs)

                # We'll initialize accumulators after computing the first chunk
                # because TFR output shape depends on picks='eeg' and decimation
                averaged_power = None
                complex_average = None
                itc_accumulator = None
                itc = None  # Initialize ITC to None
                tfr_times = None
                tfr_info = None

                # Process in chunks
                n_processed = 0
                chunk_size_int = optimal_chunk_size
                n_chunks = int(np.ceil(n_epochs_total / chunk_size_int))
                logger.info(f"Processing {n_epochs_total} epochs in {n_chunks} chunks of {chunk_size_int}")

                for start_idx in range(0, n_epochs_total, chunk_size_int):
                    end_idx = min(start_idx + chunk_size_int, n_epochs_total)
                    chunk_size = end_idx - start_idx
                    chunk_num = start_idx // chunk_size_int + 1

                    logger.info(f"Chunk {chunk_num}/{n_chunks}: Computing TFR for epochs {start_idx}-{end_idx-1}")

                    try:
                        # Extract epoch chunk
                        chunk_epochs = epochs[start_idx:end_idx]

                        # Compute TFR for this chunk only
                        logger.debug(f"  Computing {method} TFR for {chunk_size} epochs...")
                        if method == "morlet":
                            chunk_tfr = chunk_epochs.compute_tfr(
                                method=method,
                                freqs=freqs,
                                n_cycles=n_cycles,
                                use_fft=True,
                                return_itc=False,
                                average=False,
                                output='complex',
                                n_jobs=4,
                                picks='eeg',
                                verbose='WARNING',  # Reduce verbosity in loop
                                **kwargs
                            )
                        elif method == "multitaper":
                            chunk_tfr = chunk_epochs.compute_tfr(
                                method=method,
                                freqs=freqs,
                                n_cycles=n_cycles,
                                use_fft=True,
                                return_itc=False,
                                average=False,
                                output='complex',
                                n_jobs=4,
                                picks='eeg',
                                verbose='WARNING',
                                **kwargs
                            )

                        logger.debug(f"  TFR computed: shape={chunk_tfr.data.shape}")

                        # Extract complex data for this chunk
                        chunk_complex = chunk_tfr.data  # (chunk_size, n_channels, n_freqs, n_times)

                        # Initialize accumulators on first chunk (now we know the actual TFR output shape)
                        if averaged_power is None:
                            _, n_channels_tfr, n_freqs_tfr, n_times_tfr = chunk_complex.shape
                            logger.debug(f"  Initializing accumulators with TFR output shape: ({n_channels_tfr}, {n_freqs_tfr}, {n_times_tfr})")
                            averaged_power = np.zeros((n_channels_tfr, n_freqs_tfr, n_times_tfr), dtype=np.float64)
                            complex_average = np.zeros((n_channels_tfr, n_freqs_tfr, n_times_tfr), dtype=np.complex128) if compute_complex_average else None
                            itc_accumulator = np.zeros((n_channels_tfr, n_freqs_tfr, n_times_tfr), dtype=np.complex128) if compute_itc else None
                            tfr_times = chunk_tfr.times
                            tfr_info = chunk_tfr.info
                            logger.info(f"  Accumulators initialized: {n_channels_tfr} channels, {n_freqs_tfr} freqs, {n_times_tfr} time points")

                        # Accumulate complex average if requested
                        if compute_complex_average:
                            logger.debug(f"  Accumulating complex average...")
                            complex_average += np.sum(chunk_complex, axis=0)

                        # Accumulate ITC if requested
                        if compute_itc:
                            logger.debug(f"  Computing ITC...")
                            epsilon = np.finfo(float).eps
                            chunk_abs = np.abs(chunk_complex)
                            normalized_chunk = chunk_complex / (chunk_abs + epsilon)
                            itc_accumulator += np.sum(normalized_chunk, axis=0)

                        # Convert to power and apply baseline correction
                        logger.debug(f"  Converting to power and applying baseline...")
                        chunk_power = np.abs(chunk_complex) ** 2
                        corrected_chunk_power = apply_single_trial_baseline(
                            chunk_power, chunk_tfr.times, baseline
                        )

                        # Accumulate corrected power
                        averaged_power += np.sum(corrected_chunk_power, axis=0)
                        n_processed += chunk_size

                        logger.info(f"  Chunk {chunk_num} complete ({n_processed}/{n_epochs_total} epochs)")

                        # Free memory
                        del chunk_epochs, chunk_tfr, chunk_complex, chunk_power, corrected_chunk_power
                        if compute_itc and 'chunk_abs' in locals():
                            del chunk_abs, normalized_chunk
                        gc.collect()

                    except Exception as chunk_error:
                        logger.error(f"Error in chunk {chunk_num}: {type(chunk_error).__name__}: {chunk_error}")
                        import traceback
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                        raise

                # Finalize averages
                logger.info("Finalizing averages...")
                if averaged_power is None or tfr_info is None or tfr_times is None:
                    raise RuntimeError("No chunks were successfully processed - cannot finalize TFR")

                averaged_power /= n_epochs_total

                if compute_complex_average and complex_average is not None:
                    complex_average /= n_epochs_total
                    logger.info("Complex average finalized")

                if compute_itc and itc_accumulator is not None:
                    itc_data = np.abs(itc_accumulator / n_epochs_total)
                    # Create ITC object by copying structure from a template epoch
                    logger.debug("Creating ITC AverageTFR object...")
                    # Create a template AverageTFR from first epoch to get correct structure
                    template_epochs = epochs[0:1]
                    template_tfr = template_epochs.compute_tfr(
                        method=method,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,
                        average=True,
                        n_jobs=1,
                        picks='eeg',
                        verbose='WARNING',
                        **kwargs
                    )
                    itc = template_tfr.copy()
                    itc.data = itc_data
                    itc.nave = n_epochs_total
                    itc.comment = "Inter-trial coherence (chunked)"
                    logger.info("ITC finalized")
                    del itc_accumulator, template_epochs, template_tfr

                # Create final AverageTFR object
                logger.info("Creating final AverageTFR object...")
                # Create a template AverageTFR from first epoch to get correct structure
                template_epochs = epochs[0:1]
                template_tfr = template_epochs.compute_tfr(
                    method=method,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    average=True,
                    n_jobs=1,
                    picks='eeg',
                    verbose='WARNING',
                    **kwargs
                )
                power = template_tfr.copy()
                power.data = averaged_power
                power.nave = n_epochs_total
                power.comment = f"Single-trial baseline: z-score (fully chunked: {chunk_size_int} epochs)"
                del template_epochs, template_tfr

                logger.success(f"Fully chunked TFR + baseline complete: {n_processed} epochs processed")

            else:
                # Standard path: TFR already computed, just do chunked baseline correction
                logger.debug("Extracting complex TFR data...")
                complex_data = epochs_tfr.data  # (n_epochs, n_channels, n_freqs, n_times)
                n_epochs, n_channels, n_freqs, n_times = complex_data.shape
                logger.debug(f"Complex data extracted. Shape: {complex_data.shape}, dtype: {complex_data.dtype}")

                # Initialize accumulators for chunked processing
                logger.debug("Initializing accumulators for chunked processing...")
                averaged_power = np.zeros((n_channels, n_freqs, n_times), dtype=np.float64)
                complex_average = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128) if compute_complex_average else None
                itc_accumulator = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128) if compute_itc else None
                logger.debug(f"Accumulators initialized (power: {averaged_power.shape}, complex_avg: {complex_average is not None}, itc: {itc_accumulator is not None})")

                # Process in chunks to manage memory usage
                n_processed = 0
                chunk_size_int = optimal_chunk_size if optimal_chunk_size is not None else n_epochs
                n_chunks = int(np.ceil(n_epochs / chunk_size_int))
                logger.info(f"Starting chunked processing: {n_chunks} chunks of size {chunk_size_int}")

                for start_idx in range(0, n_epochs, chunk_size_int):
                    end_idx = min(start_idx + chunk_size_int, n_epochs)
                    chunk_size = end_idx - start_idx
                    chunk_num = start_idx // chunk_size_int + 1

                    logger.info(f"Processing chunk {chunk_num}/{n_chunks}: epochs {start_idx}-{end_idx-1}")

                    try:
                        # Extract chunk of complex data
                        logger.debug(f"  Extracting chunk data from indices {start_idx}:{end_idx}")
                        chunk_complex = complex_data[start_idx:end_idx]  # (chunk_size, n_channels, n_freqs, n_times)
                        logger.debug(f"  Chunk extracted: shape={chunk_complex.shape}, dtype={chunk_complex.dtype}")

                        # Accumulate complex average if requested
                        if compute_complex_average:
                            logger.debug(f"  Accumulating complex average...")
                            complex_average += np.sum(chunk_complex, axis=0)  # Sum across trials in chunk

                        # Accumulate ITC if requested
                        if compute_itc:
                            logger.debug(f"  Computing ITC for chunk...")
                            epsilon = np.finfo(float).eps
                            # Normalize complex values and accumulate
                            chunk_abs = np.abs(chunk_complex)
                            normalized_chunk = chunk_complex / (chunk_abs + epsilon)
                            itc_accumulator += np.sum(normalized_chunk, axis=0)  # Sum normalized complex values

                        # Convert to power for baseline correction
                        logger.debug(f"  Converting to power...")
                        chunk_power = np.abs(chunk_complex) ** 2
                        logger.debug(f"  Power computed: shape={chunk_power.shape}, min={chunk_power.min():.2e}, max={chunk_power.max():.2e}")

                        # Apply baseline correction to chunk
                        logger.debug(f"  Applying baseline correction (mode=z-score, baseline={baseline})...")
                        corrected_chunk_power = apply_single_trial_baseline(
                            chunk_power, epochs_tfr.times, baseline  # type: ignore
                        )
                        logger.debug(f"  Baseline correction complete: min={corrected_chunk_power.min():.2e}, max={corrected_chunk_power.max():.2e}")

                        # Accumulate corrected power (sum across trials in chunk)
                        logger.debug(f"  Accumulating corrected power...")
                        averaged_power += np.sum(corrected_chunk_power, axis=0)
                        n_processed += chunk_size
                        logger.debug(f"  Chunk {chunk_num} complete. Total epochs processed: {n_processed}/{n_epochs}")

                        # Free memory immediately for this chunk
                        del chunk_complex, chunk_power, corrected_chunk_power
                        if compute_itc and 'chunk_abs' in locals():
                            del chunk_abs, normalized_chunk
                        gc.collect()  # Force garbage collection

                    except Exception as chunk_error:
                        logger.error(f"Error processing chunk {chunk_num}: {type(chunk_error).__name__}: {chunk_error}")
                        logger.error(f"Chunk indices: {start_idx}:{end_idx}, size: {chunk_size}")
                        import traceback
                        logger.error(f"Chunk error traceback:\n{traceback.format_exc()}")
                        raise

                # Finalize averages
                logger.debug(f"Finalizing averages from {n_processed} epochs...")
                averaged_power /= n_epochs  # Convert sum to average

                if compute_complex_average:
                    complex_average /= n_epochs  # Convert sum to average
                    logger.info("Complex average computed using chunked processing")

                if compute_itc:
                    # Finalize ITC computation
                    logger.debug("Finalizing ITC computation...")
                    itc_data = np.abs(itc_accumulator / n_epochs)  # Convert sum to average, then magnitude

                    # Create ITC AverageTFR object
                    itc = epochs_tfr.copy()
                    itc = itc.average()  # Convert EpochsTFR to AverageTFR
                    itc.data = itc_data
                    itc.comment = "Inter-trial coherence (chunked)"
                    logger.info("ITC computed using chunked processing")
                    del itc_accumulator

                # Create AverageTFR object from epochs_tfr
                logger.debug("Creating final AverageTFR object...")
                power = epochs_tfr.copy()
                power = power.average()  # Convert EpochsTFR to AverageTFR
                power.data = averaged_power  # Replace with baseline-corrected averaged data
                power.comment = f"Single-trial baseline: z-score (chunked: {chunk_size_int} epochs)"

                # Free the large complex_data array
                del complex_data
                logger.success(f"Chunked single-trial baseline correction complete: {n_processed} epochs processed")

        except Exception as baseline_error:
            logger.error(f"Single-trial baseline correction failed: {type(baseline_error).__name__}: {baseline_error}")
            logger.error(f"Failed during initialization or chunked processing")
            import traceback
            logger.error(f"Full baseline correction traceback:\n{traceback.format_exc()}")
            raise
    
    else:
        # Traditional post-averaging baseline correction
        # But we may still need to compute complex average if requested
        complex_average = None
        if compute_complex_average:
            # We need to get complex data even for traditional pipeline
            # This requires a separate computation without single-trial processing
            logger.info("Computing complex data for complex average...")
            if method == "morlet":
                epochs_tfr_for_complex = epochs.compute_tfr(
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
                epochs_tfr_for_complex = epochs.compute_tfr(
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
            logger.info(f"Applying baseline correction: {baseline}, mode: z-score")
            # Non-single-trial mode removed - only z-score supported
    
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