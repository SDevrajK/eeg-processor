from pathlib import Path
from mne.io import read_raw_cnt, BaseRaw
from mne.channels import make_standard_montage
from .base import FileLoader
import logging

logger = logging.getLogger(__name__)

class NeuroscanLoader(FileLoader):
    """Handler for Neuroscan (.cnt) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading Neuroscan file: {file_path.name}")
        raw = read_raw_cnt(file_path, **kwargs)
        # Always apply standard montage — embedded .cnt positions are often incorrectly scaled,
        # causing MNE to estimate an unrealistic head radius (100cm) during interpolation.
        raw.set_montage(make_standard_montage('standard_1005'), match_case=False, on_missing='ignore')
        logger.info("Applied standard 10-05 montage")
        cls._retype_non_eeg_channels(raw)
        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.cnt'

    @classmethod
    def _validate_file(cls, file_path: Path):
        """Basic validation for Neuroscan files"""
        if not file_path.exists():
            raise FileNotFoundError(f"Neuroscan file not found: {file_path}")
        if file_path.stat().st_size < 100:
            raise ValueError(f"File too small to be valid Neuroscan: {file_path}")