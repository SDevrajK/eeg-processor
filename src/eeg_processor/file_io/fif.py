from pathlib import Path
from mne.io import read_raw_fif, BaseRaw
from .base import FileLoader
from loguru import logger
import numpy as np

class FifLoader(FileLoader):
    """Handler for FIFF (.fif) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading FIF file: {file_path.name}")

        # Suppress common FIF warnings if needed
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="This filename.*does not conform to MNE naming conventions.*")

            raw = read_raw_fif(file_path, **kwargs, verbose=False)

        # Explicitly set channel types to exclude EOG from forward model
        eog_channels = ['HEOG', 'VEOG', 'EOG', 'heog', 'veog', 'eog']
        stim_channels = ['STIM', 'TRIGGER']

        # Set channel types
        raw.set_channel_types({ch: 'eog' for ch in eog_channels if ch in raw.ch_names})
        raw.set_channel_types({ch: 'stim' for ch in stim_channels if ch in raw.ch_names})

        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in ('.fif', '.fiff')

    @classmethod
    def _validate_file(cls, file_path: Path):
        """Basic validation for FIF files"""
        if not file_path.exists():
            raise FileNotFoundError(f"FIF file not found: {file_path}")
        if file_path.stat().st_size < 100:
            raise ValueError(f"File too small to be valid FIF: {file_path}")


# Convenience function for backward compatibility
def load_fif_file(file_path: Path, **kwargs) -> BaseRaw:
    return FifLoader.load(file_path, **kwargs)