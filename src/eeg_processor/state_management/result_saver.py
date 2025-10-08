from pathlib import Path
from mne import Epochs, Evoked
from mne.time_frequency import AverageTFR, EpochsTFR, Spectrum, RawTFR
import logging
from loguru import logger
import shutil

# Internal imports
from ..utils.metadata_utils import propagate_participant_metadata


class ResultSaver:
    def __init__(self, output_root: Path, naming_convention: str = "bids"):
        self.output_root = output_root
        self.naming_convention = naming_convention
        self.logger = logger
        self._create_dirs()

    def _create_dirs(self):
        """Ensure output directories exist"""
        (self.output_root / 'interim').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'figures').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'processed').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'quality').mkdir(parents=True, exist_ok=True)

    def _get_filename(self, participant_id: str, condition_name: str, data_type: str, file_ext: str, event_type: str = None) -> str:
        """Generate filename based on naming convention"""
        if self.naming_convention == "bids":
            # Clean participant ID: S_002_F -> S002F, GINMMN_6 -> GINMMN6
            clean_id = participant_id.replace("_", "").replace("-", "")
            # Clean condition name: convert to lowercase, replace spaces/special chars
            clean_condition = condition_name.lower().replace(" ", "").replace("_", "").replace("-", "")
            
            if event_type is not None:
                # Clean event type for BIDS compliance (lowercase, no spaces/special chars)
                clean_event = event_type.lower().replace(" ", "").replace("_", "").replace("-", "")
                return f"sub-{clean_id}_task-{clean_condition}_desc-{clean_event}_{data_type}.{file_ext}"
            else:
                return f"sub-{clean_id}_task-{clean_condition}_desc-{data_type}.{file_ext}"
        else:
            # Legacy naming
            if event_type is not None:
                return f"{participant_id}_{condition_name}_{event_type}-{data_type}.{file_ext}"
            else:
                return f"{participant_id}_{condition_name}-{data_type}.{file_ext}"

    def save_condition(self, epochs: Epochs = None, spectrum: Spectrum = None,
                       tfr: AverageTFR = None, epochs_tfr: EpochsTFR = None,
                       raw_tfr: RawTFR = None,
                       participant_id: str = None, condition_name: str = None,
                       event_type: str = None, overwrite: bool = True):
        """
        Save condition-specific results for any data type.

        Args:
            epochs: Epochs object to save (optional).
            spectrum: Spectrum object to save (optional).
            tfr: AverageTFR object to save (optional).
            epochs_tfr: EpochsTFR object to save (optional).
            raw_tfr: RawTFR object (continuous TFR) to save (optional).
            participant_id: ID string (e.g., "S_002_F").
            condition_name: Condition name from config.
            overwrite: Whether to replace existing files.
        """
        if not any([epochs, spectrum, tfr, epochs_tfr, raw_tfr]):
            raise ValueError("Must provide at least one data object to save")

        if not participant_id or not condition_name:
            raise ValueError("Must provide participant_id and condition_name")

        try:
            # Handle Epochs (existing logic)
            if epochs is not None:
                self._save_epochs(epochs, participant_id, condition_name, event_type, overwrite)

            # Handle Spectrum
            if spectrum is not None:
                self._save_spectrum(spectrum, participant_id, condition_name, event_type, overwrite)

            # Handle AverageTFR
            if tfr is not None:
                self._save_average_tfr(tfr, participant_id, condition_name, event_type, overwrite)

            # Handle EpochsTFR
            if epochs_tfr is not None:
                self._save_epochs_tfr(epochs_tfr, participant_id, condition_name, event_type, overwrite)

            # Handle RawTFR (continuous TFR)
            if raw_tfr is not None:
                self._save_raw_tfr(raw_tfr, participant_id, condition_name, event_type, overwrite)

        except Exception as e:
            self.logger.error(f"Failed saving condition {condition_name}: {str(e)}")
            raise

    def _save_epochs(self, epochs: Epochs, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """Save epochs with event grouping based on event_id"""
        # Check if epochs contains multiple event types
        if isinstance(epochs.event_id, dict) and len(epochs.event_id) > 1:
            # Multiple event types present - save each separately
            for event_name, event_code in epochs.event_id.items():
                try:
                    # Select epochs for this specific event type
                    subset_epochs = epochs[event_name]
                    
                    # Skip if no epochs for this event type
                    if len(subset_epochs) == 0:
                        self.logger.warning(f"No epochs found for event '{event_name}', skipping")
                        continue
                    
                    # Clean event name for filename (replace / with _)
                    clean_event_name = event_name.replace('/', '_')
                    
                    # Generate BIDS-compliant filenames
                    epochs_filename = self._get_filename(participant_id, condition_name, "epo", "fif", clean_event_name)
                    epochs_path = self.output_root / 'interim' / epochs_filename
                    subset_epochs.save(epochs_path, overwrite=overwrite)
                    self.logger.info(f"Saved {len(subset_epochs)} epochs for '{event_name}' to {epochs_path}")
                    
                    # Save evoked data for this event type
                    evoked = subset_epochs.average()
                    # Propagate participant metadata from epochs to evoked
                    propagate_participant_metadata(subset_epochs, evoked)
                    
                    evoked_filename = self._get_filename(participant_id, condition_name, "ave", "fif", clean_event_name)
                    evoked_path = self.output_root / 'processed' / evoked_filename
                    evoked.save(evoked_path, overwrite=overwrite)
                    self.logger.info(f"Saved evoked for '{event_name}' to {evoked_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to save epochs for event '{event_name}': {str(e)}")
                    continue
        else:
            # No event grouping - save normally (with optional event_type)
            epochs_filename = self._get_filename(participant_id, condition_name, "epo", "fif", event_type)
            epochs_path = self.output_root / 'interim' / epochs_filename
            epochs.save(epochs_path, overwrite=overwrite)
            self.logger.info(f"Saved epochs to {epochs_path}")

            # Save evoked
            evoked = epochs.average()
            # Propagate participant metadata from epochs to evoked
            propagate_participant_metadata(epochs, evoked)
            
            evoked_filename = self._get_filename(participant_id, condition_name, "ave", "fif", event_type)
            evoked_path = self.output_root / 'processed' / evoked_filename
            evoked.save(evoked_path, overwrite=overwrite)
            self.logger.info(f"Saved evoked to {evoked_path}")

    def _save_spectrum(self, spectrum: Spectrum, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """Save spectrum object"""
        spectrum_filename = self._get_filename(participant_id, condition_name, "spectrum", "h5", event_type)
        spectrum_path = self.output_root / 'processed' / spectrum_filename
        spectrum.save(spectrum_path, overwrite=overwrite)
        self.logger.info(f"Saved spectrum to {spectrum_path}")

    def _save_average_tfr(self, tfr: AverageTFR, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """Save AverageTFR object, extracting ITC and complex average if present"""
        import numpy as np

        # Check for ITC data and extract it
        itc_data = None
        if hasattr(tfr, '_itc_data'):
            itc_data = tfr._itc_data
            # Remove ITC from the power object before saving
            delattr(tfr, '_itc_data')

        # Check for complex average data and extract it
        complex_average = None
        if hasattr(tfr, '_complex_average'):
            complex_average = tfr._complex_average
            # Remove complex average from the power object before saving
            delattr(tfr, '_complex_average')

        # Save the power TFR
        tfr_filename = self._get_filename(participant_id, condition_name, "tfr", "h5", event_type)
        tfr_path = self.output_root / 'processed' / tfr_filename
        tfr.save(tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved AverageTFR to {tfr_path}")

        # Save ITC separately if it exists
        if itc_data is not None:
            itc_filename = self._get_filename(participant_id, condition_name, "itc", "h5", event_type)
            itc_path = self.output_root / 'processed' / itc_filename
            itc_data.save(itc_path, overwrite=overwrite)
            self.logger.info(f"Saved ITC to {itc_path}")

        # Save complex average separately if it exists
        if complex_average is not None:
            complex_avg_filename = self._get_filename(participant_id, condition_name, "complexavg", "npy", event_type)
            complex_avg_path = self.output_root / 'processed' / complex_avg_filename
            np.save(complex_avg_path, complex_average)
            self.logger.info(f"Saved complex average to {complex_avg_path}")

            # Also save metadata for loading
            metadata = {
                'shape': complex_average.shape,
                'times': tfr.times.copy(),
                'freqs': tfr.freqs.copy(),
                'ch_names': tfr.ch_names.copy(),
                'info_keys': ['sfreq', 'subject_info'] if hasattr(tfr.info, 'subject_info') else ['sfreq']
            }
            metadata_filename = self._get_filename(participant_id, condition_name, "complexavg-meta", "npy", event_type)
            metadata_path = self.output_root / 'processed' / metadata_filename
            np.save(metadata_path, metadata)
            self.logger.debug(f"Saved complex average metadata to {metadata_path}")

    def _save_epochs_tfr(self, epochs_tfr: EpochsTFR, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """Save EpochsTFR object"""
        epochs_tfr_filename = self._get_filename(participant_id, condition_name, "epochs_tfr", "h5", event_type)
        epochs_tfr_path = self.output_root / 'interim' / epochs_tfr_filename
        epochs_tfr.save(epochs_tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved EpochsTFR to {epochs_tfr_path}")

    def _save_raw_tfr(self, raw_tfr: RawTFR, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """Save RawTFR object (continuous time-frequency)"""
        raw_tfr_filename = self._get_filename(participant_id, condition_name, "continuous_tfr", "h5", event_type)
        raw_tfr_path = self.output_root / 'processed' / raw_tfr_filename
        raw_tfr.save(raw_tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved continuous TFR to {raw_tfr_path}")

    def save_data_object(self, data_object, participant_id: str, condition_name: str, event_type: str = None, overwrite: bool = True):
        """
        Convenience method that automatically detects data type and saves appropriately.
        
        Args:
            data_object: MNE data object to save
            participant_id: Participant identifier
            condition_name: Condition name (task name for BIDS)
            event_type: Optional event/trigger name for BIDS desc entity
            overwrite: Whether to overwrite existing files
        """
        if isinstance(data_object, Epochs):
            self.save_condition(epochs=data_object, participant_id=participant_id,
                                condition_name=condition_name, event_type=event_type, overwrite=overwrite)
        elif isinstance(data_object, Spectrum):
            self.save_condition(spectrum=data_object, participant_id=participant_id,
                                condition_name=condition_name, event_type=event_type, overwrite=overwrite)
        elif isinstance(data_object, AverageTFR):
            self.save_condition(tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, event_type=event_type, overwrite=overwrite)
        elif isinstance(data_object, EpochsTFR):
            self.save_condition(epochs_tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, event_type=event_type, overwrite=overwrite)
        elif isinstance(data_object, RawTFR):
            self.save_condition(raw_tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, event_type=event_type, overwrite=overwrite)
        else:
            raise ValueError(f"Unsupported data type: {type(data_object)}")

    def clear_output(self):
        """Clear all output directories (for testing)"""
        for dir_type in ['interim', 'figures', 'processed', 'quality']:
            dir_path = self.output_root / dir_type
            if dir_path.exists():
                shutil.rmtree(dir_path)
        self._create_dirs()