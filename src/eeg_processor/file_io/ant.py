from pathlib import Path
from mne.io import read_raw_ant, BaseRaw
from .base import FileLoader
from loguru import logger


class ANTLoader(FileLoader):
    """Handler for ANT Neuro (.cnt) files"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading ANT Neuro file: {file_path.name}")

        raw = read_raw_ant(file_path, **kwargs, verbose=False)
        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        # Need to distinguish from Neuroscan .cnt files
        # ANT files typically have different internal structure
        if file_path.suffix.lower() != '.cnt':
            return False

        # Quick heuristic: try to detect ANT vs Neuroscan
        try:
            # ANT files typically have specific header patterns
            with open(file_path, 'rb') as f:
                header = f.read(100)
                # Look for ANT-specific markers (adjust as needed)
                return b'eego' in header.lower() or b'ant neuro' in header.lower()
        except:
            return False

    @classmethod
    def _validate_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"ANT Neuro file not found: {file_path}")
