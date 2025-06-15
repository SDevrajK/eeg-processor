from pathlib import Path
from mne.io import read_raw_brainvision, BaseRaw
from mne.channels import make_standard_montage
from .base import FileLoader
from loguru import logger
import numpy as np


class BrainVisionLoader(FileLoader):
    """Handler for BrainVision (.vhdr) files with companion file validation and automatic montage detection"""

    @staticmethod
    def load(file_path: Path, **kwargs) -> BaseRaw:
        BrainVisionLoader._validate_file(file_path)
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
        BrainVisionLoader._set_channel_types(raw)

        # Check and fix missing channel locations
        BrainVisionLoader._fix_missing_montage(raw)

        return raw

    @staticmethod
    def _set_channel_types(raw: BaseRaw):
        """Set appropriate channel types for known channels"""
        type_mapping = {}

        for ch_name in raw.ch_names:
            if ch_name.upper() in ['EOG', 'LEOG', 'REOG', 'HEOG', 'VEOG']:
                type_mapping[ch_name] = 'eog'
            elif ch_name.upper() in ['ECG', 'EKG']:
                type_mapping[ch_name] = 'ecg'
            elif ch_name.upper() in ['EMG']:
                type_mapping[ch_name] = 'emg'

        if type_mapping:
            raw.set_channel_types(type_mapping)
            logger.info(f"Set channel types: {type_mapping}")

    @staticmethod
    def _fix_missing_montage(raw: BaseRaw):
        """Detect and fix missing channel locations using standard montages"""

        # Check if montage is missing or has empty positions
        if not BrainVisionLoader._has_valid_montage(raw):
            logger.warning("Missing or invalid channel locations detected - attempting to fix")

            # Try to find appropriate standard montage
            montage = BrainVisionLoader._find_best_montage(raw)

            if montage:
                # Apply montage but only for channels that exist in both
                BrainVisionLoader._apply_partial_montage(raw, montage)
            else:
                logger.warning("Could not find suitable standard montage for channel names")
        else:
            logger.debug("Valid channel locations already present")

    @staticmethod
    def _has_valid_montage(raw: BaseRaw) -> bool:
        """Check if the raw object has valid channel positions"""
        montage = raw.get_montage()

        if montage is None:
            return False

        # Check if any EEG channels have actual coordinate information
        eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg']

        if not eeg_channels:
            return True  # No EEG channels to check

        # Get positions for EEG channels
        ch_pos = montage.get_positions()['ch_pos']

        for ch_name in eeg_channels[:5]:  # Check first 5 EEG channels
            if ch_name in ch_pos:
                pos = ch_pos[ch_name]
                # Check if position is not just zeros or NaN
                if not (np.allclose(pos, 0) or np.any(np.isnan(pos))):
                    return True

        return False

    @staticmethod
    def _find_best_montage(raw: BaseRaw):
        """Find the best matching standard montage based on channel names"""
        eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg']

        if not eeg_channels:
            logger.debug("No EEG channels found - skipping montage detection")
            return None

        # Standard montages to try, in order of preference
        montage_names = [
            'standard_1020',  # Most common
            'standard_1005',  # Higher density
            'easycap-M1',  # EasyCap systems
            'biosemi64',  # BioSemi systems
            'biosemi128',  # Higher density BioSemi
            'GSN-HydroCel-128',  # EGI systems
            'GSN-HydroCel-256'
        ]

        best_montage = None
        best_match_count = 0

        for montage_name in montage_names:
            try:
                montage = make_standard_montage(montage_name)
                montage_channels = set(montage.ch_names)

                # Count matching channels (case-insensitive)
                matches = 0
                for ch in eeg_channels:
                    if any(ch.upper() == mont_ch.upper() for mont_ch in montage_channels):
                        matches += 1

                match_percentage = matches / len(eeg_channels) * 100
                logger.debug(
                    f"Montage {montage_name}: {matches}/{len(eeg_channels)} channels match ({match_percentage:.1f}%)")

                # Consider it a good match if >= 70% of channels match
                if matches > best_match_count and match_percentage >= 70:
                    best_match_count = matches
                    best_montage = montage

            except Exception as e:
                logger.debug(f"Could not load montage {montage_name}: {e}")
                continue

        if best_montage:
            match_percentage = best_match_count / len(eeg_channels) * 100
            logger.info(
                f"Selected montage with {best_match_count}/{len(eeg_channels)} matching channels ({match_percentage:.1f}%)")

        return best_montage

    @staticmethod
    def _apply_partial_montage(raw: BaseRaw, montage):
        """Apply montage but only for channels that exist in both raw and montage"""

        # Create a mapping of channel names (case-insensitive)
        montage_ch_map = {ch.upper(): ch for ch in montage.ch_names}
        raw_eeg_channels = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg']

        # Find channels that exist in both
        matched_channels = []
        rename_dict = {}

        for raw_ch in raw_eeg_channels:
            if raw_ch.upper() in montage_ch_map:
                montage_ch = montage_ch_map[raw_ch.upper()]
                if raw_ch != montage_ch:
                    rename_dict[raw_ch] = montage_ch
                matched_channels.append(montage_ch)

        if not matched_channels:
            logger.warning("No matching channels found between raw data and montage")
            return

        # Rename channels if necessary to match montage
        if rename_dict:
            logger.info(f"Renaming channels to match montage: {len(rename_dict)} channels")
            raw.rename_channels(rename_dict)

        try:
            # Apply the montage
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            logger.success(f"Applied montage: {len(matched_channels)} channels positioned")

        except Exception as e:
            logger.error(f"Failed to apply montage: {e}")
            # Revert channel names if montage application failed
            if rename_dict:
                reverse_dict = {v: k for k, v in rename_dict.items()}
                raw.rename_channels(reverse_dict)

    @staticmethod
    def supports_format(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.vhdr'

    @staticmethod
    def _validate_file(file_path: Path):
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