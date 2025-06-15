from abc import ABC, abstractmethod
from pathlib import Path
from mne.io import BaseRaw


class FileLoader(ABC):
    """Abstract base class for all EEG file loaders"""

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