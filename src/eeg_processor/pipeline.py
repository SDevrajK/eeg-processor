# Standard library imports
import gc
from typing import Dict, List, Optional, Union, Any

# Third-party imports
import numpy as np
import mne
from mne.io import BaseRaw
from mne import BaseEpochs, Evoked
from mne.time_frequency import AverageTFR, Spectrum, RawTFR
from loguru import logger

# Internal imports
from .utils.unicode_path_fix import apply_unicode_path_fix
from .utils.config_loader import load_config
from .utils.memory_tools import (get_process_memory_detailed, 
                                 MemoryTracker, cleanup_and_monitor_gc, log_memory_with_context)
from .quality_control.quality_tracker import QualityTracker
from .state_management.data_processor import DataProcessor
from .state_management.participant_handler import ParticipantHandler
from .state_management.result_saver import ResultSaver
from .file_io import load_raw
from .processing.segmentation import segment_raw_by_conditions

# Apply Unicode normalization for file paths
apply_unicode_path_fix()


class EEGPipeline:
    """
    Comprehensive EEG data processing pipeline for scientific research.
    
    Supports multiple EEG file formats and provides a complete processing workflow
    from raw data to analysis results with built-in quality control and reporting.
    """
    
    # Class constants
    LARGE_MEMORY_INCREASE_THRESHOLD = 2000  # MB
    HIGH_MEMORY_USAGE_THRESHOLD = 8000      # MB
    
    # ==========================================
    # INITIALIZATION AND SETUP
    # ==========================================
    
    def __init__(self, config_path: Optional[Union[str, Dict[str, Any]]] = None) -> None:
        """Initialize EEG processing pipeline."""
        self.config: Optional[Any] = None
        self.processor: DataProcessor = DataProcessor()
        self.participant_handler: Optional[ParticipantHandler] = None
        self.result_saver: Optional[ResultSaver] = None
        self._current_raw: Optional[BaseRaw] = None
        self._current_epochs: Optional[BaseEpochs] = None
        self._current_evoked: Optional[Evoked] = None
        self.quality_tracker: Optional[QualityTracker] = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: Union[str, Dict[str, Any]]) -> 'EEGPipeline':
        """Load configuration and setup pipeline components."""
        self.config = load_config(config_path)
        
        self.participant_handler = ParticipantHandler(self.config)
        self.result_saver = ResultSaver(self.config.results_dir)
        self.quality_tracker = QualityTracker(self.config.results_dir)
        
        # Set results directory in processor for save_raw stage
        self.processor._results_dir = self.config.results_dir

        self._log_configuration_details()
        return self

    def set_quality_tracker(self, quality_tracker: QualityTracker) -> None:
        """Allow external quality tracker injection for session-based processing."""
        self.quality_tracker = quality_tracker
        logger.debug("External quality tracker injected for session processing")

    # ==========================================
    # PUBLIC API - MAIN PROCESSING METHODS
    # ==========================================

    def run(self) -> None:
        """Process all participants through the pipeline in batch mode."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        logger.info(f"Starting batch processing for {len(self.participant_handler.participants)} participants")

        for participant in self.participant_handler.participants:
            try:
                logger.debug(f"Processing participant: {participant.id}")
                self.quality_tracker.track_participant_start(participant.id)

                self._process_participant(participant)

                self.quality_tracker.track_participant_completion(participant.id)
                logger.success(f"Completed processing: {participant.id}")

                # Force garbage collection between participants
                gc.collect()

            except Exception as e:
                logger.error(f"Failed processing {participant.id}: {str(e)}")
                continue
            finally:
                # Reset processor to interactive mode
                self.processor.set_batch_mode(False)

        # Save quality metrics and generate reports
        self.quality_tracker.save_metrics()
        logger.success("Quality metrics saved. Run generate_quality_reports() to create HTML reports.")

    def apply_stage(self, data: Union[BaseRaw, BaseEpochs, Evoked], stage_name: str,
                    condition: Optional[dict] = None, **params) -> Union[BaseRaw, BaseEpochs, Evoked]:
        """Apply a processing stage - for interactive use."""
        # Set interactive mode (no in-place operations)
        self.processor.set_batch_mode(False)

        # Set condition if provided
        if condition:
            self.processor.set_condition(condition)

        try:
            result = self.processor.apply_processing_stage(data, stage_name, **params)
            self._update_pipeline_state(result)
            return result

        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed: {str(e)}")
            raise

    def generate_quality_reports(self) -> Optional[tuple[str, List[str]]]:
        """Generate HTML quality reports after processing."""
        if not self.quality_tracker:
            logger.error("No quality tracker available. Run batch processing first.")
            return None

        try:
            from .quality_control.quality_reporter import generate_quality_reports

            # Generate reports from the saved metrics
            summary_path, participant_paths = generate_quality_reports(self.config.results_dir)

            logger.success(f"Quality reports generated:")
            logger.success(f"  Summary report: {summary_path}")
            logger.success(f"  Individual reports: {len(participant_paths)} files")

            return summary_path, participant_paths

        except Exception as e:
            logger.error(f"Failed to generate quality reports: {str(e)}")
            return None

    # ==========================================
    # CONVENIENCE METHODS
    # ==========================================

    def get_analysis_interface(self) -> Any:
        """Return minimal interface for loading processed data."""
        from .utils.analysis_interface import AnalysisInterface
        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return AnalysisInterface(self.config, self.config.results_dir)

    def load_raw(self, file_path: str) -> BaseRaw:
        """Load raw data using file_io module."""
        self._current_raw = load_raw(file_path)
        return self._current_raw

    def load_participant(self, participant_id: str) -> BaseRaw:
        """Load data for a specific participant."""
        if not self.participant_handler:
            raise ValueError("Participant handler not initialized - load config first")
        participant = next(p for p in self.participant_handler.participants
                           if p.id == participant_id)
        return self.load_raw(participant.file_path)

    # ==========================================
    # PRIVATE METHODS - PARTICIPANT PROCESSING
    # ==========================================

    def _process_participant(self, participant) -> None:
        """Process a single participant through all conditions."""
        with MemoryTracker(f"loading data for {participant.id}") as tracker:
            raw = self._load_and_preload_raw_data(participant)
            
        self._log_raw_data_info(raw, participant.id, tracker.memory_before, tracker.memory_after)
        self._log_event_information(raw)
        self._log_additional_metadata(raw)
        
        segmented_data = segment_raw_by_conditions(raw, self.config.conditions)
        self._cleanup_original_raw_if_segmented(raw, segmented_data)
        
        for condition in self.config.conditions:
            self._process_condition(condition, segmented_data, participant, tracker.memory_before['rss_mb'])

    def _process_condition(self, condition: dict, segmented_data: Dict[str, BaseRaw], 
                          participant, memory_baseline: float) -> None:
        """Process a single condition through all pipeline stages with cleanup."""
        condition_name = condition['name']
        condition_raw = segmented_data[condition_name]
        
        log_memory_with_context(f"Processing condition: {condition_name}\n  - Data duration: {condition_raw.times[-1]:.1f}s", 
                               baseline_memory=memory_baseline, participant_id=participant.id)
        
        # Setup condition processing
        self._setup_condition_processing(condition, participant.id)
        
        baseline_memory_mb = memory_baseline if memory_baseline is not None else get_process_memory_detailed()['rss_mb']

        current_data = condition_raw
        previous_type = type(current_data)
        failed_stage = None
        epochs_created = False
        
        try:
            for stage_config in self.config.stages:
                stage_name, stage_params = self._parse_stage_config(stage_config)
                failed_stage = stage_name
                
                if stage_name == "epoch":
                    epochs_created = True
                    
                with MemoryTracker(stage_name, baseline_memory=baseline_memory_mb, 
                                 warn_threshold_mb=self.LARGE_MEMORY_INCREASE_THRESHOLD,
                                 critical_threshold_mb=self.HIGH_MEMORY_USAGE_THRESHOLD) as stage_tracker:

                    new_data = self._process_stage(current_data, stage_name, stage_params, condition, 
                                                 participant, epochs_created, stage_tracker)

                # Handle type transitions and cleanup
                current_type = type(new_data)
                if current_type != previous_type:
                    self._handle_type_transition(current_data, new_data, previous_type, current_type,
                                               stage_params, participant.id, condition, clear_raw_after_epoching=True)
                    previous_type = current_type

                current_data = new_data
                self._update_pipeline_state(current_data)

            self._finalize_condition_processing(current_data, condition, participant.id, baseline_memory_mb)

        except Exception as e:
            self._handle_condition_failure(e, failed_stage, condition, participant.id)
            
        # Clean up segmented data after processing (if multiple conditions)
        if len(self.config.conditions) > 1 and condition_raw is not segmented_data.get('original'):
            logger.debug(f"Freeing segmented data for {condition_name}")
            cleanup_and_monitor_gc(segmented_data[condition_name], f"segmented data cleanup for {condition_name}")
            del segmented_data[condition_name]

    # ==========================================
    # PRIVATE METHODS - STAGE PROCESSING
    # ==========================================

    def _process_stage(self, current_data, stage_name: str, stage_params: dict, condition: dict,
                      participant, epochs_created: bool, stage_tracker) -> Union[BaseRaw, BaseEpochs, Evoked, dict]:
        """Process a single pipeline stage."""
        logger.debug(f"Applying stage: {stage_name}")
        
        # Prepare data items for processing
        data_items = self._prepare_stage_data(current_data, stage_name, condition, epochs_created)
        
        # Process all data items
        results = {}
        for trigger_name, data_to_process in data_items:
            try:
                # Add participant info to stage params for stages that need it
                stage_params_with_info = stage_params.copy()
                
                # Only add participant_info for stages that need it
                stages_needing_participant_info = ['crop_participant_segment']
                if stage_name in stages_needing_participant_info:
                    if hasattr(participant, 'metadata') and participant.metadata:
                        # Pass the metadata directly since that's what contains clean_segment info
                        stage_params_with_info['participant_info'] = participant.metadata
                    elif hasattr(participant, '__dict__'):
                        stage_params_with_info['participant_info'] = participant.__dict__
                    else:
                        stage_params_with_info['participant_info'] = {}
                    
                result = self.processor.apply_processing_stage(data_to_process, stage_name, **stage_params_with_info)

                if trigger_name is not None:
                    results[trigger_name] = result
                    self._track_stage_quality(data_to_process, result, stage_name, participant.id, condition, {
                        'memory_before_mb': stage_tracker.memory_before['rss_mb'],
                        'trigger_name': trigger_name
                    })
                    logger.debug(f"Successfully processed trigger '{trigger_name}' for stage '{stage_name}'")
                else:
                    return result
                    
            except Exception as e:
                if trigger_name is not None:
                    logger.error(f"Failed to process trigger '{trigger_name}' for stage '{stage_name}': {e}")
                    logger.info(f"Continuing with remaining triggers...")
                    continue
                else:
                    raise
        
        # Handle results
        if len(results) > 0:
            return results
        elif len(data_items) > 1:
            raise RuntimeError(f"No triggers were successfully processed for stage '{stage_name}'. "
                             f"Check trigger configuration and data compatibility.")
        return current_data

    def _prepare_stage_data(self, current_data, stage_name: str, condition: dict, epochs_created: bool) -> List[tuple]:
        """Prepare data items for stage processing."""
        # Early return for non-epoched data or epoch stage
        if not epochs_created or stage_name == "epoch":
            return [(None, current_data)]

        # Handle dict of trigger-specific epochs (from previous multi-trigger stage)
        if isinstance(current_data, dict):
            logger.info(f"Processing {stage_name} for {len(current_data)} pre-split triggers: {list(current_data.keys())}")
            return [(trigger_name, epochs_data) for trigger_name, epochs_data in current_data.items()]

        # Handle single Epochs object - split by triggers
        if isinstance(current_data, BaseEpochs):
            # Post-epoching: process each trigger separately
            triggers = condition['triggers']
            data_items = []

            logger.info(f"Processing {stage_name} for {len(triggers)} trigger types: {list(triggers.keys())}")

            for trigger_name, trigger_code in triggers.items():
                trigger_epochs = self._extract_trigger_epochs(current_data, trigger_name, trigger_code)
                if trigger_epochs is not None and len(trigger_epochs) > 0:
                    data_items.append((trigger_name, trigger_epochs))

            return data_items

        # Default: pass through as-is
        return [(None, current_data)]

    def _handle_type_transition(self, current_data, new_data, previous_type: type, current_type: type,
                               stage_params: dict, participant_id: str, condition: dict, clear_raw_after_epoching: bool):
        """Handle data type transitions and cleanup."""
        should_save = stage_params.get('save', True)

        # Helper to safely check if type is subclass (handles dict and other non-class types)
        def is_subclass_safe(type_obj, parent_class):
            try:
                return isinstance(type_obj, type) and issubclass(type_obj, parent_class)
            except TypeError:
                return False

        # Special case: Don't save epochs at type transition if followed by baseline/rejection
        # Epochs will be saved at finalization after all processing is complete
        is_raw_to_epochs = (is_subclass_safe(previous_type, BaseRaw) and
                           (current_type is dict or is_subclass_safe(current_type, BaseEpochs)))
        if is_raw_to_epochs:
            logger.debug("Skipping epoch save at type transition - will save after all post-epoch stages complete")
            should_save = False

        if should_save:
            self._save_interim_data(current_data, participant_id, condition, previous_type)

        # Early return if no cleanup needed
        if current_data is None:
            return

        # Special handling for Raw -> Epochs transition
        is_raw_to_epochs_cleanup = (clear_raw_after_epoching and
                                    is_subclass_safe(previous_type, BaseRaw) and
                                    is_subclass_safe(current_type, BaseEpochs))
        if is_raw_to_epochs_cleanup:
            self._cleanup_raw_after_epoching(current_data)
            return

        # Standard cleanup
        cleanup_and_monitor_gc(current_data, f"{type(current_data).__name__} cleanup")

    # ==========================================
    # PRIVATE METHODS - CONDITION PROCESSING SUPPORT
    # ==========================================

    def _setup_condition_processing(self, condition: dict, participant_id: str):
        """Setup condition processing state."""
        self.quality_tracker.track_condition_start(participant_id, condition['name'])
        self.processor.set_batch_mode(True)
        self.processor.set_condition(condition)
        logger.debug(f"Processing with raw data object (no copy)")

    def _finalize_condition_processing(self, current_data, condition: dict, participant_id: str, baseline_memory_mb: float):
        """Finalize condition processing with saving and tracking."""
        log_memory_with_context(f"Condition '{condition['name']}' completed", 
                               baseline_memory=baseline_memory_mb, participant_id=participant_id)
        
        self._save_interim_data(current_data, participant_id, condition, type(current_data))
        self.quality_tracker.track_completion(participant_id, condition['name'], success=True)

    def _handle_condition_failure(self, error: Exception, failed_stage: Optional[str], condition: dict, participant_id: str):
        """Handle condition processing failure."""
        if failed_stage:
            self.quality_tracker.track_stage(participant_id, condition['name'], failed_stage, {
                'stage_failed': True,
                'error': str(error),
            })

        self.quality_tracker.track_completion(participant_id, condition['name'],
                                              success=False,
                                              error=f"Stage '{failed_stage}' failed: {str(error)}")
        logger.error(f"Condition '{condition['name']}' failed at stage '{failed_stage}' for {participant_id}: {str(error)}")

    def _track_stage_quality(self, input_data: Union[BaseRaw, BaseEpochs, Evoked], output_data: Union[BaseRaw, BaseEpochs, Evoked], 
                           stage_name: str, participant_id: str, condition: dict, memory_metrics: dict) -> None:
        """Minimal quality tracking - delegates to QualityTracker."""
        if self.quality_tracker:
            self.quality_tracker.track_stage_data(
                input_data=input_data,
                output_data=output_data,
                stage_name=stage_name,
                participant_id=participant_id,
                condition_name=condition['name'],
                memory_metrics=memory_metrics
            )
        else:
            logger.warning("Quality tracker not initialized - skipping quality tracking")

    # ==========================================
    # HELPER METHODS - UTILITIES
    # ==========================================


    def _parse_stage_config(self, stage_config: Union[str, Dict]) -> tuple[str, dict]:
        """Extract stage name and parameters from config."""
        if isinstance(stage_config, dict):
            stage_name = list(stage_config.keys())[0]
            stage_params = stage_config[stage_name] or {}
        else:
            stage_name = stage_config
            stage_params = {}
        return stage_name, stage_params

    def _extract_trigger_epochs(self, epochs: BaseEpochs, trigger_name: str, trigger_code: Union[int, str]) -> Optional[BaseEpochs]:
        """Extract epochs for a specific trigger type using MNE's built-in selection."""
        try:
            # Use trigger name directly as selection key
            trigger_epochs = epochs[trigger_name]
                
            if len(trigger_epochs) == 0:
                logger.warning(f"No epochs found for trigger '{trigger_name}' with code '{trigger_code}'")
                return None
                
            return trigger_epochs
            
        except KeyError:
            # Try alternative selection methods
            try:
                available_events = list(epochs.event_id.keys())

                # Try to find a matching event
                for event_name in available_events:
                    if (str(trigger_code) == event_name or
                        trigger_name.lower() == event_name.lower() or
                        str(trigger_code) == str(epochs.event_id[event_name])):
                        return epochs[event_name]

                logger.warning(f"Could not find matching event for trigger '{trigger_name}' (code: {trigger_code})")
                return None
                
            except Exception as e:
                logger.error(f"Error extracting epochs for trigger '{trigger_name}': {e}")
                return None

    def _save_interim_data(self, data: Union[BaseRaw, BaseEpochs, Evoked, Dict], participant_id: str, condition: dict, data_type: type) -> None:
        """Save interim data based on type using the updated ResultSaver."""
        # Handle multi-trigger results (dictionary of trigger_name -> data)
        if isinstance(data, dict):
            logger.debug(f"Saving {len(data)} trigger-specific results")
            for trigger_name, trigger_data in data.items():
                # Use isinstance to handle subclasses (e.g., EpochsArray is a subclass of BaseEpochs)
                if isinstance(trigger_data, (BaseEpochs, Evoked, AverageTFR, Spectrum, RawTFR)):
                    logger.debug(f"Saving {type(trigger_data).__name__} for trigger '{trigger_name}'")
                    self.result_saver.save_data_object(
                        data_object=trigger_data,
                        participant_id=participant_id,
                        condition_name=condition['name'],
                        event_type=trigger_name
                    )
        elif isinstance(data, (BaseEpochs, Evoked, AverageTFR, Spectrum, RawTFR)):
            self.result_saver.save_data_object(
                data_object=data,
                participant_id=participant_id,
                condition_name=condition["name"]
            )

    def _update_pipeline_state(self, data: Union[BaseRaw, BaseEpochs, Evoked, Dict]) -> None:
        """Update internal state tracking."""
        data_type_mapping = {
            BaseRaw: '_current_raw',
            BaseEpochs: '_current_epochs',
            Evoked: '_current_evoked'
        }
        
        for data_type, attr_name in data_type_mapping.items():
            if isinstance(data, data_type):
                setattr(self, attr_name, data)
                break

    # ==========================================
    # HELPER METHODS - LOGGING AND CLEANUP
    # ==========================================

    def _log_configuration_details(self) -> None:
        """Log configuration loading details."""
        if self.config.dataset_name:
            logger.info(f"Configuration loaded: {len(self.config.participants)} participants for dataset '{self.config.dataset_name}'")
            logger.info(f"Results will be saved to: {self.config.results_dir}")
        else:
            logger.info(f"Configuration loaded: {len(self.config.participants)} participants")

    def _load_and_preload_raw_data(self, participant) -> BaseRaw:
        """Load and preload raw data for a participant."""
        raw = load_raw(participant.file_path)
        if not raw.preload:
            logger.debug(f"Preloading raw data for {participant.id} to avoid temp file issues")
            raw.load_data()
        
        # Add participant metadata to MNE subject_info using his_id field
        self._add_participant_metadata_to_raw(raw, participant)
        
        return raw

    def _add_participant_metadata_to_raw(self, raw: BaseRaw, participant) -> None:
        """Add participant metadata to MNE Raw object using subject_info."""
        if not participant.metadata:
            return
        
        try:
            from datetime import date
            
            # Create structured his_id with participant metadata
            metadata_parts = [participant.id]  # Start with participant ID
            for key, value in participant.metadata.items():
                # Convert values to strings and escape pipe characters
                value_str = str(value).replace('|', '_')
                metadata_parts.append(f"{key}={value_str}")
            
            structured_his_id = '|'.join(metadata_parts)
            
            # Create or update subject_info with proper MNE format
            subject_info = {
                'id': hash(participant.id) % 10000,  # Generate numeric ID from participant ID
                'his_id': structured_his_id,  # Pack metadata here
                'last_name': participant.id.split('_')[-1] if '_' in participant.id else 'Unknown',
                'first_name': participant.id.split('_')[0] if '_' in participant.id else participant.id,
                'birthday': date(1990, 1, 1),  # Default birthday
                'sex': 0,  # 0=unknown (can be overridden by metadata if needed)
            }
            
            # Override sex if available in metadata
            if 'sex' in participant.metadata:
                sex_mapping = {'male': 1, 'female': 2, 'm': 1, 'f': 2}
                sex_value = str(participant.metadata['sex']).lower()
                subject_info['sex'] = sex_mapping.get(sex_value, 0)
            elif '_F' in participant.id:
                subject_info['sex'] = 2  # Female
            elif '_M' in participant.id:
                subject_info['sex'] = 1  # Male
            
            raw.info['subject_info'] = subject_info
            
            logger.debug(f"Added participant metadata to raw data for {participant.id}")
            logger.debug(f"  Metadata keys: {list(participant.metadata.keys())}")
            logger.debug(f"  Structured his_id: {structured_his_id}")
            
        except Exception as e:
            logger.warning(f"Failed to add participant metadata to raw data: {e}")
            # Don't raise - this is non-critical functionality

    def _log_raw_data_info(self, raw: BaseRaw, participant_id: str, memory_before: dict, memory_after: dict) -> None:
        """Log raw data information and memory usage."""
        load_memory_cost = memory_after['rss_mb'] - memory_before['rss_mb']
        logger.info(f"Raw data loaded successfully for {participant_id} - Memory after load: {memory_after['rss_mb']:.0f} MB (+{load_memory_cost:.0f} MB)")
        logger.info(f"  - Sampling rate: {raw.info['sfreq']} Hz")
        logger.info(f"  - Recording duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
        logger.info(f"  - Number of channels: {len(raw.ch_names)} ({len([ch for ch in raw.info['chs'] if ch['kind'] == 2])} EEG)")
        logger.info(f"  - Channel names: {', '.join(raw.ch_names[:10])}{'...' if len(raw.ch_names) > 10 else ''}")

    def _log_event_information(self, raw: BaseRaw) -> None:
        """Log event information from raw data."""
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if events is not None and len(events) > 0:
                logger.info(f"  - Total events: {len(events)}")
                event_counts = []
                for name, code in sorted(event_id.items()):
                    count = np.sum(events[:, 2] == code)
                    if count > 0:
                        event_counts.append(f"{name}: {count}")
                if event_counts:
                    logger.info(f"  - Event counts: {', '.join(event_counts)}")
            else:
                logger.info("  - No events found in annotations")
        except Exception:
            logger.info("  - No events/annotations found in the data")

    def _log_additional_metadata(self, raw: BaseRaw) -> None:
        """Log additional metadata from raw data."""
        if raw.info.get('subject_info'):
            logger.info(f"  - Subject info: {raw.info['subject_info']}")
        if raw.info.get('experimenter'):
            logger.info(f"  - Experimenter: {raw.info['experimenter']}")
        if raw.info.get('meas_date'):
            logger.info(f"  - Measurement date: {raw.info['meas_date']}")

    def _cleanup_original_raw_if_segmented(self, raw: BaseRaw, segmented_data: Dict[str, BaseRaw]) -> None:
        """Free original raw data if all conditions were successfully segmented."""
        if len(self.config.conditions) > 1 and len(segmented_data) > 1:
            all_segmented = all(
                segmented_data[c['name']] is not raw 
                for c in self.config.conditions
            )
            if all_segmented:
                logger.info("All conditions segmented - freeing original raw data")
                cleanup_and_monitor_gc(raw, "original raw data cleanup")

    def _cleanup_raw_after_epoching(self, current_data):
        """Clean up raw data after epoching to free memory."""
        logger.info(f"Clearing raw data after epoching to free memory")
        
        mem_before = get_process_memory_detailed()
        logger.info(f"Memory before clearing raw: {mem_before['rss_mb']:.1f} MB")
        
        # Clear the current_data (which is the raw copy)
        if current_data is not None:
            del current_data
            current_data = None
        
        # Clear the pipeline's reference to raw data
        if hasattr(self, '_current_raw') and self._current_raw is not None:
            del self._current_raw
            self._current_raw = None
        
        # Force garbage collection for large raw data
        collected = gc.collect()
        
        # Get memory after cleanup
        mem_after = get_process_memory_detailed()
        freed_memory = mem_before['rss_mb'] - mem_after['rss_mb']
        
        logger.info(f"Freed raw data memory: {freed_memory:.1f} MB, GC collected {collected} objects")
        logger.info(f"Memory after clearing raw: {mem_after['rss_mb']:.1f} MB")

    # ==========================================
    # PROPERTIES
    # ==========================================

    @property
    def current_raw(self) -> Optional[BaseRaw]:
        """Get current raw data object."""
        return self._current_raw

    @current_raw.setter
    def current_raw(self, value: Optional[BaseRaw]) -> None:
        """Set current raw data object."""
        self._current_raw = value

    @property
    def current_epochs(self) -> Optional[BaseEpochs]:
        """Get current epochs object."""
        return self._current_epochs

    @current_epochs.setter
    def current_epochs(self, value: Optional[BaseEpochs]) -> None:
        """Set current epochs object."""
        self._current_epochs = value