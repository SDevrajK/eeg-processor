# At the very top of pipeline.py, before other imports
import pathlib
import unicodedata

# Monkey patch pathlib.Path to always normalize Unicode
_original_path_new = pathlib.Path.__new__

def _normalized_path_new(cls, *args, **kwargs):
    if args:
        normalized_arg = unicodedata.normalize('NFC', str(args[0]))
        args = (normalized_arg,) + args[1:]
    return _original_path_new(cls, *args, **kwargs)

pathlib.Path.__new__ = _normalized_path_new

from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import numpy as np
import gc
from mne.io import BaseRaw
from mne import Epochs, Evoked
from mne.time_frequency import AverageTFR, Spectrum, RawTFR

# helper imports
from .utils.config_loader import load_config, load_config_from_dict
from .quality_control.quality_tracker import QualityTracker
from .state_management.data_processor import DataProcessor
from .state_management.participant_handler import ParticipantHandler
from .state_management.result_saver import ResultSaver
from .file_io import load_raw
from .utils.memory_tools import get_memory_pressure, get_memory_metrics

class EEGPipeline:
    def __init__(self, config_path: str = None):
        self.config = None
        self.processor = DataProcessor()
        self.participant_handler = None
        self.result_saver = None
        self._current_raw = None
        self._current_epochs = None
        self._current_evoked = None
        self.quality_tracker = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str = None, config_data: Dict = None):
        """Load configuration and setup pipeline components"""
        if config_path and config_data:
            raise ValueError("Provide either config_path or config_data, not both")
        elif config_path:
            self.config = load_config(config_path)
        elif config_data:
            self.config = load_config_from_dict(config_data)
        else:
            raise ValueError("Must provide either config_path or config_data")
        self.participant_handler = ParticipantHandler(self.config)
        self.result_saver = ResultSaver(self.config.results_dir)

        # Initialize quality tracker
        self.quality_tracker = QualityTracker(self.config.results_dir)

        logger.info(f"Configuration loaded: {len(self.config.participants)} participants")
        return self

    def get_analysis_interface(self):
        from .utils.analysis_interface import AnalysisInterface
        """Return minimal interface for loading processed data"""
        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return AnalysisInterface(self.config, self.config.results_dir)

    def set_quality_tracker(self, quality_tracker):
        """Allow external quality tracker injection for session-based processing"""
        self.quality_tracker = quality_tracker
        logger.debug("External quality tracker injected for session processing")

    def apply_stage(self,
                    data: Union[BaseRaw, Epochs, Evoked],
                    stage_name: str,
                    condition: Optional[dict] = None,
                    **params) -> Union[BaseRaw, Epochs, Evoked]:
        """
        Apply a processing stage - for interactive use

        Args:
            data: Input MNE data object
            stage_name: Name of the processing stage
            condition: Condition dictionary for stages that need it
            **params: Stage-specific parameters

        Returns:
            Processed data object
        """
        # Set interactive mode (no in-place operations)
        self.processor.set_batch_mode(False)

        # Set condition if provided
        if condition:
            self.processor.set_condition(condition)

        try:
            result = self.processor.apply_processing_stage(data, stage_name, **params)

            # Update pipeline state for convenience
            self._update_pipeline_state(result)
            return result

        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed: {str(e)}")
            raise

    def run(self):
        """Process all participants through the pipeline in batch mode"""
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
                # Track any participants that failed completely
                continue
            finally:
                # Reset processor to interactive mode
                self.processor.set_batch_mode(False)

        # Save quality metrics and generate reports
        self.quality_tracker.save_metrics()
        logger.success("Quality metrics saved. Run generate_quality_reports() to create HTML reports.")

    def _process_participant(self, participant):
        """Process a single participant through all conditions"""
        raw = load_raw(participant.file_path)

        # Get conditions to process (if filtering is specified)
        conditions_to_process = self._get_target_conditions()

        for condition in self.config.conditions:
            if conditions_to_process and condition['name'] not in conditions_to_process:
                logger.debug(f"Skipping condition: {condition['name']}")
                continue

            logger.info(f"Processing condition: {condition['name']}")
            self._process_condition(raw, condition, participant.id)

    def _process_condition(self, raw: BaseRaw, condition: dict, participant_id: str):
        """Memory-efficient condition processing with quality tracking"""

        self.quality_tracker.track_condition_start(participant_id, condition['name'])

        # Set batch mode and condition
        self.processor.set_batch_mode(True)
        self.processor.set_condition(condition)

        current_data = raw.copy()
        previous_type = type(current_data)
        failed_stage = None

        try:
            for stage_config in self.config.stages:
                stage_name, stage_params = self._parse_stage_config(stage_config)
                failed_stage = stage_name  # Track which stage we're attempting

                # Track Memory Usage
                memory_before = get_memory_pressure()

                ### Apply the current stage ###
                logger.debug(f"Applying stage: {stage_name}")
                new_data = self.processor.apply_processing_stage(current_data, stage_name, **stage_params)

                # Track Memory Usage
                memory_after = get_memory_pressure()
                memory_metrics = get_memory_metrics(memory_before, memory_after)

                # Track successful stage
                self._track_stage_quality(current_data, new_data, stage_name, participant_id, condition, memory_metrics)

                # Handle type transitions - save interim results
                current_type = type(new_data)
                if current_type != previous_type:
                    # Check if saving is disabled for the stage that produced this data
                    should_save = stage_params.get('save', True)  # Default to True
                    if should_save:
                        self._save_interim_data(current_data, participant_id, condition, previous_type)

                    # Memory cleanup
                    if current_data is not raw and id(current_data) != id(raw):
                        del current_data

                    previous_type = current_type

                # Check if we're approaching limits
                if memory_after['pressure_level'] == 'critical':
                    logger.warning(f"High memory pressure after {stage_name}: {memory_after['used_percent']:.1f}%")
                elif memory_after['pressure_level'] == 'abort':
                    logger.error(f"Memory critical - aborting participant {participant_id}")
                    # Handle gracefully - mark participant as failed due to memory
                    self.quality_tracker.track_completion(participant_id, condition['name'],
                                                          success=False, error="Memory exceeded system limits")
                    break  # Exit stage loop for this condition

                current_data = new_data
                self._update_pipeline_state(current_data)

            # All stages completed successfully
            self._save_interim_data(current_data, participant_id, condition, type(current_data))
            self.quality_tracker.track_completion(participant_id, condition['name'], success=True)

        except Exception as e:
            # Record which stage failed
            if failed_stage:
                self.quality_tracker.track_stage(participant_id, condition['name'], failed_stage, {
                    'stage_failed': True,
                    'error': str(e),
                })

            # Mark condition as failed and continue to next condition (no raise)
            self.quality_tracker.track_completion(participant_id, condition['name'],
                                                  success=False,
                                                  error=f"Stage '{failed_stage}' failed: {str(e)}")
            logger.error(
                f"Condition '{condition['name']}' failed at stage '{failed_stage}' for {participant_id}: {str(e)}")


    def _track_stage_quality(self, input_data, output_data, stage_name: str,
                             participant_id: str, condition: dict, memory_metrics: dict) -> None:
        """Minimal quality tracking - delegates to QualityTracker"""

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

    def _parse_stage_config(self, stage_config: Union[str, Dict]) -> tuple:
        """Extract stage name and parameters from config"""
        if isinstance(stage_config, dict):
            stage_name = list(stage_config.keys())[0]
            stage_params = stage_config[stage_name] or {}
        else:
            stage_name = stage_config
            stage_params = {}
        return stage_name, stage_params

    def _save_interim_data(self, data, participant_id: str, condition: dict, data_type: type):
        """Save interim data based on type using the updated ResultSaver"""
        if data_type in [Epochs, Evoked, AverageTFR, Spectrum, RawTFR]:
            # Add metadata for epochs if needed
            # if isinstance(data, Epochs):
            #     from .processing.epoching import extract_event_metadata
            #     metadata_df = extract_event_metadata(data, condition)
            #     if metadata_df is not None:
            #         data.metadata = metadata_df

            # Use save_data_object for automatic type detection and proper method routing
            self.result_saver.save_data_object(
                data_object=data,
                participant_id=participant_id,
                condition_name=condition["name"]
            )

    def _update_pipeline_state(self, data):
        """Update internal state tracking"""
        if isinstance(data, BaseRaw):
            self._current_raw = data
        elif isinstance(data, Epochs):
            self._current_epochs = data
        elif isinstance(data, Evoked):
            self._current_evoked = data

    def _get_target_conditions(self) -> Optional[List[str]]:
        """Extract condition names from stage parameters if specified"""
        for stage in self.config.stages:
            if isinstance(stage, dict) and 'segment_condition' in stage:
                return stage['segment_condition']
        return None

    # Convenience methods
    def load_raw(self, file_path: str) -> BaseRaw:
        """Load raw data using file_io module"""
        self.current_raw = load_raw(file_path)
        return self.current_raw

    def load_participant(self, participant_id: str):
        """Load data for a specific participant"""
        if not self.participant_handler:
            raise ValueError("Participant handler not initialized - load config first")
        participant = next(p for p in self.participant_handler.participants
                           if p.id == participant_id)
        return self.load_raw(participant.file_path)

    def add_participant_metadata(self, csv_path: str, participant_id_column: str = "participant_id",
                                 data_types: Optional[Dict[str, str]] = None):
        """
        Add participant metadata from CSV file.

        Args:
            csv_path: Path to CSV file containing metadata
            participant_id_column: Column name containing participant IDs
            data_types: Optional dict for data type conversion (e.g., {'age': 'int'})

        Returns:
            Self for method chaining
        """
        if not self.participant_handler:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        self.participant_handler.add_metadata(csv_path, participant_id_column, data_types)
        return self

    def generate_quality_reports(self):
        """Generate HTML quality reports after processing"""
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

    @property
    def current_raw(self):
        return self._current_raw

    @current_raw.setter
    def current_raw(self, value):
        self._current_raw = value

    @property
    def current_epochs(self):
        return self._current_epochs

    @current_epochs.setter
    def current_epochs(self, value):
        self._current_epochs = value