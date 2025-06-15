from pathlib import Path
from mne import Epochs, Evoked
from mne.time_frequency import AverageTFR, EpochsTFR, Spectrum, RawTFR
import logging
from loguru import logger
import shutil


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

    def _get_filename(self, participant_id: str, condition_name: str, data_type: str, file_ext: str) -> str:
        """Generate filename based on naming convention"""
        if self.naming_convention == "bids":
            # Clean participant ID: S_002_F -> S002F, GINMMN_6 -> GINMMN6
            clean_id = participant_id.replace("_", "").replace("-", "")
            # Clean condition name: convert to lowercase, replace spaces/special chars
            clean_condition = condition_name.lower().replace(" ", "").replace("_", "").replace("-", "")
            return f"sub-{clean_id}_task-{clean_condition}_desc-{data_type}.{file_ext}"
        else:
            # Legacy naming
            return f"{participant_id}_{condition_name}-{data_type}.{file_ext}"

    def save_condition(self, epochs: Epochs = None, spectrum: Spectrum = None,
                       tfr: AverageTFR = None, epochs_tfr: EpochsTFR = None,
                       raw_tfr: RawTFR = None,
                       participant_id: str = None, condition_name: str = None,
                       overwrite: bool = True):
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
                self._save_epochs(epochs, participant_id, condition_name, overwrite)

            # Handle Spectrum
            if spectrum is not None:
                self._save_spectrum(spectrum, participant_id, condition_name, overwrite)

            # Handle AverageTFR
            if tfr is not None:
                self._save_average_tfr(tfr, participant_id, condition_name, overwrite)

            # Handle EpochsTFR
            if epochs_tfr is not None:
                self._save_epochs_tfr(epochs_tfr, participant_id, condition_name, overwrite)

            # Handle RawTFR (continuous TFR)
            if raw_tfr is not None:
                self._save_raw_tfr(raw_tfr, participant_id, condition_name, overwrite)

        except Exception as e:
            self.logger.error(f"Failed saving condition {condition_name}: {str(e)}")
            raise

    def _save_epochs(self, epochs: Epochs, participant_id: str, condition_name: str, overwrite: bool):
        """Save epochs with event grouping if metadata exists"""
        if epochs.metadata is not None and 'event_type' in epochs.metadata.columns:
            unique_groups = epochs.metadata['event_type'].unique()

            for group in unique_groups:
                subset_epochs = epochs[epochs.metadata['event_type'] == group]

                # Generate BIDS-compliant filenames
                epochs_filename = self._get_filename(participant_id, f"{condition_name}{group}", "epo", "fif")
                epochs_path = self.output_root / 'interim' / epochs_filename
                subset_epochs.save(epochs_path, overwrite=overwrite)
                self.logger.info(f"Saved epochs for {group} to {epochs_path}")

                # Save evoked data for this group
                evoked = subset_epochs.average()
                evoked_filename = self._get_filename(participant_id, f"{condition_name}{group}", "ave", "fif")
                evoked_path = self.output_root / 'processed' / evoked_filename
                evoked.save(evoked_path, overwrite=overwrite)
                self.logger.info(f"Saved evoked for {group} to {evoked_path}")
        else:
            # No event grouping - save normally
            epochs_filename = self._get_filename(participant_id, condition_name, "epo", "fif")
            epochs_path = self.output_root / 'interim' / epochs_filename
            epochs.save(epochs_path, overwrite=overwrite)
            self.logger.info(f"Saved epochs to {epochs_path}")

            # Save evoked
            evoked = epochs.average()
            evoked_filename = self._get_filename(participant_id, condition_name, "ave", "fif")
            evoked_path = self.output_root / 'processed' / evoked_filename
            evoked.save(evoked_path, overwrite=overwrite)
            self.logger.info(f"Saved evoked to {evoked_path}")

    def _save_spectrum(self, spectrum: Spectrum, participant_id: str, condition_name: str, overwrite: bool):
        """Save spectrum object"""
        spectrum_filename = self._get_filename(participant_id, condition_name, "spectrum", "h5")
        spectrum_path = self.output_root / 'processed' / spectrum_filename
        spectrum.save(spectrum_path, overwrite=overwrite)
        self.logger.info(f"Saved spectrum to {spectrum_path}")

    def _save_average_tfr(self, tfr: AverageTFR, participant_id: str, condition_name: str, overwrite: bool):
        """Save AverageTFR object, extracting ITC if present"""

        # Check for ITC data and extract it
        itc_data = None
        if hasattr(tfr, '_itc_data'):
            itc_data = tfr._itc_data
            # Remove ITC from the power object before saving
            delattr(tfr, '_itc_data')

        # Save the power TFR
        tfr_filename = self._get_filename(participant_id, condition_name, "tfr", "h5")
        tfr_path = self.output_root / 'processed' / tfr_filename
        tfr.save(tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved AverageTFR to {tfr_path}")

        # Save ITC separately if it exists
        if itc_data is not None:
            itc_filename = self._get_filename(participant_id, condition_name, "itc", "h5")
            itc_path = self.output_root / 'processed' / itc_filename
            itc_data.save(itc_path, overwrite=overwrite)
            self.logger.info(f"Saved ITC to {itc_path}")

    def _save_epochs_tfr(self, epochs_tfr: EpochsTFR, participant_id: str, condition_name: str, overwrite: bool):
        """Save EpochsTFR object"""
        epochs_tfr_filename = self._get_filename(participant_id, condition_name, "epochs_tfr", "h5")
        epochs_tfr_path = self.output_root / 'interim' / epochs_tfr_filename
        epochs_tfr.save(epochs_tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved EpochsTFR to {epochs_tfr_path}")

    def _save_raw_tfr(self, raw_tfr: RawTFR, participant_id: str, condition_name: str, overwrite: bool):
        """Save RawTFR object (continuous time-frequency)"""
        raw_tfr_filename = self._get_filename(participant_id, condition_name, "continuous_tfr", "h5")
        raw_tfr_path = self.output_root / 'processed' / raw_tfr_filename
        raw_tfr.save(raw_tfr_path, overwrite=overwrite)
        self.logger.info(f"Saved continuous TFR to {raw_tfr_path}")

    def save_data_object(self, data_object, participant_id: str, condition_name: str, overwrite: bool = True):
        """
        Convenience method that automatically detects data type and saves appropriately
        """
        if isinstance(data_object, Epochs):
            self.save_condition(epochs=data_object, participant_id=participant_id,
                                condition_name=condition_name, overwrite=overwrite)
        elif isinstance(data_object, Spectrum):
            self.save_condition(spectrum=data_object, participant_id=participant_id,
                                condition_name=condition_name, overwrite=overwrite)
        elif isinstance(data_object, AverageTFR):
            self.save_condition(tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, overwrite=overwrite)
        elif isinstance(data_object, EpochsTFR):
            self.save_condition(epochs_tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, overwrite=overwrite)
        elif isinstance(data_object, RawTFR):
            self.save_condition(raw_tfr=data_object, participant_id=participant_id,
                                condition_name=condition_name, overwrite=overwrite)
        else:
            raise ValueError(f"Unsupported data type: {type(data_object)}")

    def clear_output(self):
        """Clear all output directories (for testing)"""
        for dir_type in ['interim', 'figures', 'processed', 'quality']:
            dir_path = self.output_root / dir_type
            if dir_path.exists():
                shutil.rmtree(dir_path)
        self._create_dirs()