from pathlib import Path
from mne.io import read_raw_brainvision, BaseRaw
from .base import FileLoader
from loguru import logger

class BrainVisionLoader(FileLoader):
    """Handler for BrainVision (.vhdr) files with companion file validation"""

    @classmethod
    def load(cls, file_path: Path, **kwargs) -> BaseRaw:
        cls._validate_file(file_path)
        logger.info(f"Loading BrainVision file: {file_path.name}")
        # Suppress only the specific, known warnings we want to ignore
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="No coordinate information found for channels.*")
            warnings.filterwarnings("ignore",
                                    message="Online software filter detected.*")
            warnings.filterwarnings("ignore",
                                    message="Not setting position.*misc channel.*")

            raw = read_raw_brainvision(file_path, **kwargs, verbose=False)

        # Set proper channel types after loading
        if 'EOG' in raw.ch_names:
            raw.set_channel_types({'EOG': 'eog'})

        return raw

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.vhdr'

    @classmethod
    def _validate_file(cls, file_path: Path):
        """Validate all required companion files exist"""
        if not file_path.exists():
            raise FileNotFoundError(f"Primary file not found: {file_path}")

        base_path = file_path.with_suffix('')
        required_suffixes = ['.vhdr', '.eeg', '.vmrk']
        missing = [s for s in required_suffixes
                  if not base_path.with_suffix(s).exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing BrainVision companion files: {missing}\n"
                f"Required for: {file_path.name}"
            )

# Convenience function remains for backward compatibility
def load_brainvision_file(file_path: Path, **kwargs) -> BaseRaw:
    return BrainVisionLoader.load(file_path, **kwargs)