from pathlib import Path
from mne.io import read_raw_eeglab, BaseRaw
from .base import FileLoader
from loguru import logger


class EEGLABLoader(FileLoader):
    """Handler for EEGLAB (.set) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading EEGLAB file: {file_path.name}")

        raw = read_raw_eeglab(file_path, **kwargs, verbose=False)
        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.set'

    @classmethod
    def _validate_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"EEGLAB file not found: {file_path}")

        # Check for .fdt companion file
        fdt_file = file_path.with_suffix('.fdt')
        if not fdt_file.exists():
            logger.warning(f"EEGLAB .fdt file not found: {fdt_file.name} (may be embedded)")
