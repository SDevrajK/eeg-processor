from pathlib import Path
from mne.io import read_raw_bdf, BaseRaw
from .base import FileLoader
from loguru import logger


class BDFLoader(FileLoader):
    """Handler for Biosemi BDF (.bdf) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading BDF file: {file_path.name}")

        # BDF files may need status channel specification
        raw = read_raw_bdf(file_path, **kwargs, verbose=False)

        # Common BDF post-processing
        if 'Status' in raw.ch_names:
            raw.set_channel_types({'Status': 'stim'})

        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.bdf'

    @classmethod
    def _validate_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"BDF file not found: {file_path}")

