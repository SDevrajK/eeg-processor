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
        self._current_participant: Optional[str] = None

    def set_participant(self, participant_id: str) -> None:
        """Set the current participant for subsequent operations
        
        Args:
            participant_id: The participant ID to select
            
        Raises:
            ValueError: If participant_id is not in the configuration
        """
        available_participants = self.get_participant_ids()
        if participant_id not in available_participants:
            raise ValueError(f"Participant '{participant_id}' not found. Available: {available_participants}")
        self._current_participant = participant_id
        logger.info(f"Current participant set to: {participant_id}")

    def get_participant(self) -> Optional[str]:
        """Get the currently selected participant
        
        Returns:
            The current participant ID or None if no participant is selected
        """
        return self._current_participant

    def clear_participant(self) -> None:
        """Clear the current participant selection (return to all-participant mode)"""
        self._current_participant = None
        logger.info("Participant selection cleared - operating on all participants")

    def _get_bids_filename(self, participant_id: str, condition_name: str, data_type: str, file_ext: str, event_type: Optional[str] = None) -> str:
        """Generate BIDS-compliant filename"""
        clean_id = participant_id.replace("_", "").replace("-", "")
        clean_condition = condition_name.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        if event_type is not None:
            # Clean event type for BIDS compliance (lowercase, no spaces/special chars)
            clean_event = event_type.lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "_")
            return f"sub-{clean_id}_task-{clean_condition}_desc-{clean_event}_{data_type}.{file_ext}"
        else:
            return f"sub-{clean_id}_task-{clean_condition}_{data_type}.{file_ext}"

    def load_data(self, participant_id: str, condition_name: str,
                  data_type: str = "spectrum", event_type: Optional[str] = None) -> Optional[Union[mne.Epochs, Spectrum]]:
        """Load processed data for a participant and condition
        
        Args:
            participant_id: The participant ID
            condition_name: The condition name
            data_type: Type of data to load ('epochs', 'spectrum', 'evoked', etc.)
            event_type: Optional event type name (e.g., 'standard', 'deviant') for files separated by event
            
        Returns:
            The loaded data object or None if not found
        """

        # BIDS-compliant file patterns matching ResultSaver naming
        file_patterns = {
            "epochs": self._get_bids_filename(participant_id, condition_name, "epo", "fif", event_type),
            "spectrum": self._get_bids_filename(participant_id, condition_name, "spectrum", "h5", event_type),
            "evoked": self._get_bids_filename(participant_id, condition_name, "ave", "fif", event_type),
            "average_tfr": self._get_bids_filename(participant_id, condition_name, "tfr", "h5", event_type),
            "epochs_tfr": self._get_bids_filename(participant_id, condition_name, "epochs_tfr", "h5", event_type),
            "continuous_tfr": self._get_bids_filename(participant_id, condition_name, "continuous_tfr", "h5", event_type),
            "itc": self._get_bids_filename(participant_id, condition_name, "itc", "h5", event_type)
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
            # If event_type was specified and file not found, log a more specific message
            if event_type:
                logger.debug(f"No {data_type} file found for participant {participant_id}, condition {condition_name}, event {event_type}")
            return None

        logger.debug(f"Loading {data_type} from: {file_path}")

        try:
            if data_type == "spectrum":
                return mne.time_frequency.read_spectrum(file_path)
            elif data_type == "epochs":
                return mne.read_epochs(file_path)
            elif data_type == "evoked":
                return mne.read_evokeds(file_path)  # Evoked
            elif data_type == "average_tfr":
                return mne.time_frequency.read_tfrs(file_path)
            elif data_type == "epochs_tfr":
                return mne.time_frequency.read_tfrs(file_path)  # EpochsTFR
            elif data_type == "continuous_tfr":
                return mne.time_frequency.read_tfrs(file_path)  # RawTFR
            elif data_type == "itc":
                return mne.time_frequency.read_tfrs(file_path)  # ITC
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def get_participant_ids(self) -> List[str]:
        """Get list of participant IDs from config"""
        return [Path(p).stem for p in self.config.participants]

    def get_condition_names(self) -> List[str]:
        """Get list of condition names from config"""
        return [c['name'] for c in self.config.conditions]

    def list_available_data(self, data_type: Optional[str] = None) -> Union[Dict[str, List[str]], List[str], Dict[str, Dict[str, List[str]]]]:
        """Return what data is actually available
        
        Behavior depends on both data_type parameter and current participant selection:
        - data_type=None, no participant: Dict[participant_id, Dict[data_type, List[conditions]]]
        - data_type=None, with participant: Dict[data_type, List[conditions]]
        - data_type specified, no participant: Dict[participant_id, List[conditions]]
        - data_type specified, with participant: List[conditions]
        
        Args:
            data_type: Specific data type ('epochs', 'spectrum', etc.) or None for all types
            
        Returns:
            Varies based on context (see behavior description above)
        """
        supported_types = ["epochs", "spectrum", "evoked", "average_tfr", "epochs_tfr", "continuous_tfr", "itc"]
        
        if data_type is None:
            # Check all data types
            if self._current_participant:
                # Single participant, all data types
                available = {}
                for dtype in supported_types:
                    conditions = []
                    for condition_name in self.get_condition_names():
                        if self.load_data(self._current_participant, condition_name, dtype) is not None:
                            conditions.append(condition_name)
                    if conditions:
                        available[dtype] = conditions
                return available
            else:
                # All participants, all data types
                all_available = {}
                for participant_id in self.get_participant_ids():
                    participant_data = {}
                    for dtype in supported_types:
                        conditions = []
                        for condition_name in self.get_condition_names():
                            if self.load_data(participant_id, condition_name, dtype) is not None:
                                conditions.append(condition_name)
                        if conditions:
                            participant_data[dtype] = conditions
                    if participant_data:
                        all_available[participant_id] = participant_data
                return all_available
        else:
            # Check specific data type
            if data_type not in supported_types:
                raise ValueError(f"Unknown data_type: {data_type}. Supported types: {supported_types}")
                
            if self._current_participant:
                # Single participant, specific data type
                conditions = []
                for condition_name in self.get_condition_names():
                    if self.load_data(self._current_participant, condition_name, data_type) is not None:
                        conditions.append(condition_name)
                return conditions
            else:
                # All participants, specific data type
                available = {}
                for participant_id in self.get_participant_ids():
                    available[participant_id] = []
                    for condition_name in self.get_condition_names():
                        if self.load_data(participant_id, condition_name, data_type) is not None:
                            available[participant_id].append(condition_name)
                return available