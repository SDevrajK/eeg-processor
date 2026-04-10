from abc import ABC, abstractmethod
from pathlib import Path
from mne.io import BaseRaw
from loguru import logger


class FileLoader(ABC):
    """Abstract base class for all EEG file loaders"""

    # Substring patterns (uppercased) mapped to MNE channel types.
    # Checked via `pattern in ch.upper()` so case variants and suffixes are handled.
    _CHANNEL_TYPE_PATTERNS: dict = {
        'eog': ['EOG', 'HEOG', 'VEOG', 'H_EOG', 'V_EOG', 'LEOG', 'REOG'],
        'ecg': ['ECG', 'EKG'],
        'emg': ['EMG'],
        'stim': ['STIM', 'TRIGGER', 'TRIG', 'STATUS'],
        'misc': ['ACC', 'GSR', 'RESP', 'TEMP'],
    }

    @classmethod
    def _retype_non_eeg_channels(cls, raw: BaseRaw) -> None:
        """Retype channels mislabeled as EEG based on name patterns."""
        mapping = {}
        for ch in raw.ch_names:
            ch_upper = ch.upper()
            for ch_type, patterns in cls._CHANNEL_TYPE_PATTERNS.items():
                if any(p in ch_upper for p in patterns):
                    mapping[ch] = ch_type
                    break
        if mapping:
            raw.set_channel_types(mapping)
            logger.info(f"Retyped channels: {mapping}")

    @classmethod
    @abstractmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        """Load a file and return Raw object"""
        pass

    @classmethod
    @abstractmethod
    def supports_format(cls, file_path: Path) -> bool:
        """Check if this loader supports the given file"""
        pass