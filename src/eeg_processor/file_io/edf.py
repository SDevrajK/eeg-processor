from pathlib import Path
from mne.io import read_raw_edf, BaseRaw
from .base import FileLoader
from loguru import logger


class EDFLoader(FileLoader):
    """Handler for EDF/EDF+ (.edf) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading EDF file: {file_path.name}")

        # EDF files often have encoding issues
        raw = read_raw_edf(file_path, encoding='latin1', **kwargs, verbose=False)
        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in ('.edf', '.edf+')

    @classmethod
    def _validate_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")
