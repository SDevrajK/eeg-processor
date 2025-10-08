from typing import Union, Optional, List, Dict, Any, TypeVar
from loguru import logger
from mne import Epochs, Evoked
from mne.io import BaseRaw
from mne.time_frequency import AverageTFR
from dataclasses import dataclass
import numpy as np

T = TypeVar('T', BaseRaw, Epochs, Evoked)

class DataProcessor:
    def __init__(self):
        self.current_condition = None
        self._batch_mode = False 
        self.stage_registry = {
            # Data handling
            "crop": self._crop,
            "crop_participant_segment": self._crop_participant_segment,
            "adjust_events": self._adjust_event_times,
            "correct_triggers": self._correct_triggers,
            "segment_condition": self._segment_by_condition,

            # Pre-processing
            "filter": self._apply_filter,
            "compute_eog": self._compute_eog,
            "detect_bad_channels": self._detect_bad_channels,
            "rereference": self._rereference,
            "remove_artifacts": self._remove_artifacts,
            "remove_blinks_emcp": self._remove_blinks_emcp,
            "clean_rawdata_asr": self._clean_rawdata_asr,
            "blink_artifact": self._remove_artifacts,  # Legacy alias

            # Condition handling
            "epoch": self._epoch_data,

            # Post-epoching
            "time_frequency": self._time_frequency_analysis,
            "time_frequency_raw": self._time_frequency_raw_analysis,
            "time_frequency_average": self._time_frequency_average,

            # I/O operations
            "save_raw": self._save_raw,

            # Other
            "view": self._view_data
        }

    def set_batch_mode(self, batch_mode: bool):
        """Set processing mode - called by pipeline"""
        self._batch_mode = batch_mode

    def apply_processing_stage(self, data: T, stage_name: str, **params) -> Union[BaseRaw, Epochs, Evoked]:
        """Enhanced to support memory-efficient batch processing"""
        if stage_name not in self.stage_registry:
            raise ValueError(f"Unknown stage: {stage_name}")

        processor = self.stage_registry[stage_name]

        if self._batch_mode:
            # Batch mode: prefer in-place operations where safe
            return processor(data, inplace=True, **params)
        else:
            # Interactive mode: always create copies for safety
            return processor(data, inplace=False, **params)

    def set_condition(self, condition: dict):
        self.current_condition = condition

    ## --- Stage Implementations --- ##
    ## --- Data Handling --- ##

    def _crop(self, data: BaseRaw, inplace: bool = False, **kwargs) -> BaseRaw:
        """Cropping with inplace parameter passed to external function"""
        from ..utils.raw_data_tools import crop_data
        return crop_data(data, inplace=inplace, **kwargs)

    def _adjust_event_times(self, data: BaseRaw,
                            shift_ms: float,
                            target_events: Optional[List[str]] = None,
                            protect_events: Optional[List[str]] = None,
                            inplace: bool = False,
                            **kwargs) -> BaseRaw:
        """Event time adjustment with inplace parameter passed to external function"""
        from ..utils.raw_data_tools import adjust_event_times
        return adjust_event_times(raw=data, shift_ms=shift_ms, target_events=target_events, protect_events=protect_events, inplace=inplace, **kwargs)

    # Add this method to your DataProcessor class
    def _correct_triggers(self, data: BaseRaw,
                          method: str = "alternating",
                          inplace: bool = False,
                          **kwargs) -> BaseRaw:
        """Trigger correction with inplace parameter passed to external function"""
        if not self.current_condition:
            raise ValueError("Condition must be set for trigger correction")

        from ..utils.correct_triggers import correct_triggers
        return correct_triggers(
            raw=data,
            condition=self.current_condition,
            method=method,
            inplace=inplace,
            **kwargs
        )
    
    ## --- Preprocessing --- ##

    def _apply_filter(self, data: Union[BaseRaw, Epochs],
                      l_freq: Optional[float] = None,
                      h_freq: Optional[float] = None,
                      inplace: bool = False, **kwargs) -> Union[BaseRaw, Epochs]:
        """Filtering with inplace parameter passed to external function"""
        from ..processing.filtering import filter_data
        return filter_data(data, l_freq=l_freq, h_freq=h_freq, inplace=inplace, **kwargs)

    def _compute_eog(self, data: BaseRaw,
                     heog_pair: List[str],
                     veog_pair: List[str],
                     inplace: bool = False,
                     **kwargs) -> BaseRaw:
        """EOG computation with inplace parameter"""
        from ..processing.montages import compute_eog_channels
        return compute_eog_channels(raw=data, heog_pair=heog_pair, veog_pair=veog_pair, inplace=inplace, **kwargs)

    def _detect_bad_channels(self, data: BaseRaw,
                             inplace: bool = False,
                             **kwargs) -> BaseRaw:
        """Bad channel detection with inplace parameter"""
        from ..processing.bad_channels import detect_bad_channels
        return detect_bad_channels(data, inplace=inplace, **kwargs)

    def _rereference(self, data: BaseRaw,
                     method: str = "average",
                     exclude: Optional[List[str]] = None,
                     inplace: bool = False,
                     **kwargs) -> BaseRaw:
        """Rereferencing with inplace parameter"""
        from ..processing.rereferencing import set_reference
        return set_reference(data, method=method, exclude=exclude or [], inplace=inplace, **kwargs)

    def _remove_artifacts(self, data: BaseRaw,
                          method: str = "ica",
                          inplace: bool = False,
                          **kwargs) -> BaseRaw:
        """Artifact removal with inplace parameter passed to external function"""
        if method == "ica":
            from ..processing.ica import remove_artifacts_ica
            return remove_artifacts_ica(raw=data, inplace=inplace, **kwargs)
        elif method == "regression":
            # Keep for backward compatibility but recommend ICA
            logger.warning("Regression method is deprecated. Consider using ICA method instead.")
            raise NotImplementedError("Regression method currently unavailable")
        else:
            raise ValueError(f"Unknown artifact removal method: {method}. Available: 'ica', 'regression'")

    def _clean_rawdata_asr(self, data: BaseRaw,
                           cutoff: Union[int, float] = 20,
                           method: str = "euclid",
                           calibration_duration: Optional[float] = None,
                           inplace: bool = False,
                           **kwargs) -> BaseRaw:
        """ASR data cleaning with inplace parameter passed to external function
        
        ASR (Artifact Subspace Reconstruction) is designed as an intermediate cleaning step
        between bad channel detection and ICA. It corrects transient high-amplitude artifacts
        while preserving brain signals.
        
        Recommended pipeline order:
        1. detect_bad_channels (+ interpolate)
        2. clean_rawdata_asr  ← This step
        3. remove_artifacts (ICA)
        """
        from ..processing.artifact import clean_rawdata_asr
        return clean_rawdata_asr(
            raw=data, 
            cutoff=cutoff, 
            method=method, 
            calibration_duration=calibration_duration,
            inplace=inplace, 
            **kwargs
        )

    def _remove_blinks_emcp(self, data: BaseRaw,
                            method: str = "eog_regression",
                            eog_channels: List[str] = ['HEOG', 'VEOG'],
                            inplace: bool = False,
                            **kwargs) -> BaseRaw:
        """
        Remove blink artifacts using Eye Movement Correction Procedures (EMCP).
        
        Supports two methods:
        - "eog_regression": MNE's EOGRegression method (standard approach)
        - "gratton_coles": Reference-agnostic Gratton & Coles (1983) method
        
        Args:
            data: Raw EEG data with EOG channels
            method: EMCP method to use ("eog_regression" or "gratton_coles")
            eog_channels: List of EOG channel names for blink detection
            inplace: Ignored - EMCP always creates new object
            **kwargs: Additional parameters passed to selected method
            
        Returns:
            Raw object with blink artifacts removed
            
        Raises:
            ValueError: If method is unknown or EOG channels are missing
        """
        if method not in ["eog_regression", "gratton_coles"]:
            raise ValueError(f"Unknown EMCP method: {method}. "
                           f"Available methods: 'eog_regression', 'gratton_coles'")
        
        if method == "eog_regression":
            from ..processing.emcp import remove_blinks_eog_regression
            return remove_blinks_eog_regression(
                raw=data,
                eog_channels=eog_channels,
                inplace=inplace,
                **kwargs
            )
        elif method == "gratton_coles":
            from ..processing.emcp import remove_blinks_gratton_coles
            return remove_blinks_gratton_coles(
                raw=data,
                eog_channels=eog_channels,
                inplace=inplace,
                **kwargs
            )

    ## --- Condition Handling --- ##

    def _epoch_data(self,
                    data: BaseRaw,
                    tmin: float = -0.2,
                    tmax: float = 1.0,
                    baseline=(-0.1, 0),
                    inplace: bool = False,
                    **kwargs) -> Epochs:
        """Epoching with condition context"""
        if not self.current_condition:
            raise ValueError("Condition must be set before epoching")

        from ..processing.epoching import create_epochs
        return create_epochs(
            raw=data,
            condition=self.current_condition,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            inplace=inplace,
            **kwargs
        )

    ## --- More --- ##

    def _time_frequency_analysis(self, data: Epochs,
                                 freq_range: List[float] = [1, 50],
                                 n_freqs: int = 100,
                                 method: str = "morlet",
                                 n_cycles: Union[float, List[float]] = None,
                                 compute_itc: bool = True,
                                 inplace: bool = False,  # Ignored - always creates new object
                                 **kwargs) -> AverageTFR:
        """
        Convert epochs to averaged time-frequency representation

        Similar to epochs.average() creating Evoked, this creates AverageTFR

        Args:
            data: Input epochs
            freq_range: [min_freq, max_freq] in Hz
            n_freqs: Number of frequency points (logarithmically spaced)
            method: 'morlet' (default) or 'multitaper'
            n_cycles: Cycles per frequency (default: freq/2 for morlet)
            compute_itc: Whether to compute inter-trial coherence
            inplace: Ignored - TFR analysis always creates new object

        Returns:
            AverageTFR object with power (and optionally ITC)
        """
        if inplace:
            logger.info("inplace=True ignored for time-frequency analysis - always creates new AverageTFR object")

        from ..processing.time_frequency import compute_epochs_tfr_average
        return compute_epochs_tfr_average(
            epochs=data,
            freq_range=freq_range,
            n_freqs=n_freqs,
            method=method,
            n_cycles=n_cycles,
            compute_itc=compute_itc,
            **kwargs
        )

    def _time_frequency_raw_analysis(self, data: BaseRaw,
                                     freq_range: List[float] = [1, 50],
                                     n_freqs: int = 20,
                                     method: str = "welch",
                                     output_type: str = "spectrum",  # Key parameter
                                     inplace: bool = False,
                                     **kwargs):
        """
        Raw time-frequency analysis for continuous/baseline data

        Args:
            data: Raw EEG data (typically baseline or continuous data)
            freq_range: [min_freq, max_freq] in Hz
            n_freqs: Number of frequency points
            method: 'welch', 'multitaper' for spectrum; 'morlet' for raw_tfr
            output_type: 'spectrum' for time-averaged baseline, 'raw_tfr' for continuous TFR
            inplace: Ignored - always creates new object

        Returns:
            Spectrum object for baseline or RawTFR object for continuous analysis
        """
        if output_type == "spectrum":
            from ..processing.time_frequency import compute_baseline_spectrum
            return compute_baseline_spectrum(
                raw=data,
                freq_range=freq_range,
                n_freqs=n_freqs,
                method=method,
                **kwargs
            )

        elif output_type == "raw_tfr":
            # Future implementation for continuous time-frequency
            from ..processing.time_frequency import compute_raw_tfr
            return compute_raw_tfr(
                raw=data,
                freq_range=freq_range,
                n_freqs=n_freqs,
                method=method,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown output_type: '{output_type}'. Use 'spectrum' or 'raw_tfr'")

    def _time_frequency_average(self, data,
                                method: str = "mean",
                                inplace: bool = False,  # Ignored - always creates new object
                                **kwargs) -> AverageTFR:
        """
        Convert TFR objects to AverageTFR by averaging

        Handles both:
        - RawTFR → AverageTFR (average across time)
        - EpochsTFR → AverageTFR (average across trials)

        Args:
            data: RawTFR or EpochsTFR object
            method: Averaging method ('mean', 'median')
            inplace: Ignored - always creates new AverageTFR object

        Returns:
            AverageTFR object
        """
        from ..processing.time_frequency import compute_raw_tfr_average, compute_epochs_tfr_average

        if hasattr(data, 'data') and data.data.ndim == 3:  # RawTFR case
            return compute_raw_tfr_average(data, method=method, **kwargs)
        elif hasattr(data, 'data') and data.data.ndim == 4:  # EpochsTFR case
            return compute_epochs_tfr_average(data, method=method, **kwargs)
        else:
            raise ValueError(f"Unsupported data type for averaging: {type(data)}")

    def _view_data(self,
                   data: Union[BaseRaw, Epochs, Evoked],
                   **kwargs) -> Union[BaseRaw, Epochs, Evoked]:
        """Visualization that preserves data"""
        from ..processing.visualization import plot_stage
        plot_stage(data, **kwargs)
        return data  # Return unchanged for chaining

    def _save_raw(self, data: Union[BaseRaw, Epochs], inplace: bool = False, **kwargs) -> Union[BaseRaw, Epochs]:
        """Save raw or epochs data - inplace parameter ignored"""
        from ..processing.io_operations import save_raw
        
        # Set output directory from config if not specified
        if 'output_dir' not in kwargs and hasattr(self, '_results_dir'):
            kwargs['output_dir'] = self._results_dir
            
        return save_raw(data, **kwargs)

    def _crop_participant_segment(self, data: BaseRaw, inplace: bool = False, **kwargs) -> BaseRaw:
        """Crop using participant-specific metadata"""
        from ..processing.io_operations import crop_participant_segment
        
        # participant_info is already in kwargs from the pipeline
        # Just pass everything through
        return crop_participant_segment(data, **kwargs)

    def _segment_by_condition(self, data: BaseRaw, inplace: bool = False, **kwargs) -> BaseRaw:
        """Segment raw data based on condition markers"""
        from ..processing.segmentation import segment_by_condition_markers
        
        # Get current condition from processor state
        if not self.current_condition:
            raise ValueError("No condition set for segmentation. This stage requires condition information.")
        
        return segment_by_condition_markers(data, condition=self.current_condition, inplace=inplace, **kwargs)