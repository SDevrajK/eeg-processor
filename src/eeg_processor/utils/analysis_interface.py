from pathlib import Path
from typing import Dict, List, Optional, Union
import mne
from mne.time_frequency import Spectrum
from loguru import logger


class AnalysisInterface:
    """Minimal interface for loading processed EEG data with BIDS naming support"""

    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = Path(results_dir)

    def _get_bids_filename(self, participant_id: str, condition_name: str, data_type: str, file_ext: str) -> str:
        """Generate BIDS-compliant filename"""
        clean_id = participant_id.replace("_", "").replace("-", "")
        clean_condition = condition_name.lower().replace(" ", "").replace("_", "").replace("-", "")
        return f"sub-{clean_id}_task-{clean_condition}_desc-{data_type}.{file_ext}"

    def load_data(self, participant_id: str, condition_name: str,
                  data_type: str = "spectrum") -> Optional[Union[mne.Epochs, Spectrum]]:
        """Load processed data for a participant and condition"""

        # BIDS-compliant file patterns
        file_patterns = {
            "epochs": self._get_bids_filename(participant_id, condition_name, "epochs", "fif"),
            "spectrum": self._get_bids_filename(participant_id, condition_name, "spectrum", "h5"),
            "evoked": self._get_bids_filename(participant_id, condition_name, "evoked", "fif"),
            "average_tfr": self._get_bids_filename(participant_id, condition_name, "tfr", "h5"),
            "epochs_tfr": self._get_bids_filename(participant_id, condition_name, "epochs_tfr", "h5"),
            "continuous_tfr": self._get_bids_filename(participant_id, condition_name, "continuous_tfr", "h5")
        }

        filename = file_patterns.get(data_type)
        if not filename:
            raise ValueError(f"Unknown data_type: {data_type}. Use: {list(file_patterns.keys())}")

        # Search in appropriate directory
        if data_type in ["epochs", "epochs_tfr"]:
            search_dir = self.results_dir / "interim"
        else:
            search_dir = self.results_dir / "processed"

        # Look for exact BIDS filename
        file_path = search_dir / filename

        if not file_path.exists():
            logger.warning(f"No {data_type} file found: {file_path}")
            return None

        logger.debug(f"Loading {data_type} from: {file_path}")

        try:
            if data_type == "spectrum":
                return mne.time_frequency.read_spectrum(file_path)
            elif data_type == "epochs":
                return mne.read_epochs(file_path)
            elif data_type == "evoked":
                return mne.read_evokeds(file_path)[0]  # Return first evoked
            elif data_type == "average_tfr":
                return mne.time_frequency.read_tfrs(file_path)[0]
            elif data_type == "epochs_tfr":
                return mne.time_frequency.read_tfrs(file_path)[0]  # EpochsTFR
            elif data_type == "continuous_tfr":
                return mne.time_frequency.read_tfrs(file_path)  # RawTFR
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def get_participant_ids(self) -> List[str]:
        """Get list of participant IDs from config"""
        return [Path(p).stem for p in self.config.participants]

    def get_condition_names(self) -> List[str]:
        """Get list of condition names from config"""
        return [c['name'] for c in self.config.conditions]

    def list_available_data(self, data_type: str = "spectrum") -> Dict[str, List[str]]:
        """Return what data is actually available"""
        available = {}
        for participant_id in self.get_participant_ids():
            available[participant_id] = []
            for condition_name in self.get_condition_names():
                if self.load_data(participant_id, condition_name, data_type) is not None:
                    available[participant_id].append(condition_name)
        return available