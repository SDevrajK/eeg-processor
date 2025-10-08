from pathlib import Path
from mne.io import read_raw_fif, BaseRaw
from mne.channels import make_standard_montage, make_dig_montage
from .base import FileLoader
from loguru import logger
import numpy as np

class FIFLoader(FileLoader):
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

        # Detect and fix inverted electrode coordinates automatically
        cls._auto_correct_montage(raw)

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

    @classmethod
    def _auto_correct_montage(cls, raw: BaseRaw):
        """
        Automatically detect and correct inverted electrode coordinates.
        
        This method detects common coordinate system inversions and applies
        appropriate corrections to ensure proper electrode positioning.
        """
        montage = raw.get_montage()
        if montage is None:
            # If no montage exists, set standard montage
            montage = make_standard_montage('standard_1005')
            raw.set_montage(montage)
            logger.info("Applied standard 10-05 montage (no montage found)")
            return

        ch_pos = montage.get_positions()['ch_pos']
        if not ch_pos:
            logger.debug("No channel positions found in montage")
            return
            
        inversion_type = cls._detect_inversion(raw, ch_pos)

        if inversion_type == 'x':
            # Left-right flip (LAS → RAS)
            corrected_pos = {ch: np.array([-pos[0], pos[1], pos[2]])
                             for ch, pos in ch_pos.items()}
            logger.info("Corrected left-right electrode inversion (LAS → RAS)")
        elif inversion_type == 'xy':
            # Full 180° rotation
            corrected_pos = {ch: np.array([-pos[0], -pos[1], pos[2]])
                             for ch, pos in ch_pos.items()}
            logger.info("Corrected 180° electrode coordinate rotation")
        else:
            logger.debug("No electrode coordinate correction needed")
            return  # No correction needed

        # Apply corrected montage
        raw.set_montage(make_dig_montage(ch_pos=corrected_pos, coord_frame='head'))

    @classmethod
    def _detect_inversion(cls, raw: BaseRaw, ch_pos: dict) -> str:
        """
        Detect coordinate inversion type by comparing with standard electrode positions.
        
        Args:
            raw: Raw EEG data object
            ch_pos: Dictionary of channel positions {channel_name: [x, y, z]}
            
        Returns:
            'none' - No inversion detected
            'x'    - Left-right flip (LAS/RAS coordinate systems)
            'xy'   - Full 180° rotation in horizontal plane
        """
        # Get reference positions from standard montage
        std_montage = make_standard_montage('standard_1005')
        std_pos = std_montage.get_positions()['ch_pos']

        # Find common channels between data and standard
        common_chs = [ch for ch in raw.ch_names if ch in std_pos and ch in ch_pos]
        if len(common_chs) < 3:
            logger.debug(f"Insufficient common channels ({len(common_chs)}) for inversion detection")
            return 'none'  # Not enough channels to compare

        # Calculate coordinate correlations for each axis
        try:
            x_corr = np.corrcoef(
                [ch_pos[ch][0] for ch in common_chs],
                [std_pos[ch][0] for ch in common_chs]
            )[0, 1]

            y_corr = np.corrcoef(
                [ch_pos[ch][1] for ch in common_chs],
                [std_pos[ch][1] for ch in common_chs]
            )[0, 1]

            z_corr = np.corrcoef(
                [ch_pos[ch][2] for ch in common_chs],
                [std_pos[ch][2] for ch in common_chs]
            )[0, 1]

            logger.debug(f"Coordinate correlations - X: {x_corr:.3f}, Y: {y_corr:.3f}, Z: {z_corr:.3f}")

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to calculate coordinate correlations: {e}")
            return 'none'

        # Determine inversion type based on correlation patterns
        correlation_threshold = -0.8
        
        if x_corr < correlation_threshold and y_corr < correlation_threshold:
            return 'xy'  # Full 180° rotation in horizontal plane
        elif x_corr < correlation_threshold:
            return 'x'   # Left-right flip only
        else:
            return 'none'  # No significant inversion detected


# Convenience function for backward compatibility
def load_fif_file(file_path: Path, **kwargs) -> BaseRaw:
    return FIFLoader.load(file_path, **kwargs)