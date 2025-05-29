from pathlib import Path
from mne.io import read_raw_curry, BaseRaw
from mne.channels import make_standard_montage
from loguru import logger
from typing import Optional
from mne.channels import make_dig_montage
import numpy as np


class CurryLoader:
    @classmethod
    def load(cls, file_path: Path, **kwargs) -> "BaseRaw":
        raw = read_raw_curry(file_path, preload=True, **kwargs)

        # Detect and fix inverted channels automatically
        cls._auto_correct_montage(raw)

        # Explicitly set channel types to exclude EOG from forward model
        eog_channels = ['HEOG', 'VEOG']
        stim_channels = ['STIM', 'TRIGGER']

        # Set channel types
        raw.set_channel_types({ch: 'eog' for ch in eog_channels if ch in raw.ch_names})
        raw.set_channel_types({ch: 'stim' for ch in stim_channels if ch in raw.ch_names})

        return raw

    @classmethod
    def _auto_correct_montage(cls, raw: "BaseRaw"):
        montage = raw.get_montage()
        if montage is None:
            montage = make_standard_montage('standard_1005')
            raw.set_montage(montage)

        ch_pos = montage.get_positions()['ch_pos']
        inversion_type = cls._detect_inversion(raw, ch_pos)

        if inversion_type == 'x':
            # Left-right flip (LAS → RAS)
            corrected_pos = {ch: np.array([-pos[0], pos[1], pos[2]])
                             for ch, pos in ch_pos.items()}
            logger.info("Corrected left-right inversion (LAS → RAS)")
        elif inversion_type == 'xy':
            # Full 180° rotation
            corrected_pos = {ch: np.array([-pos[0], -pos[1], pos[2]])
                             for ch, pos in ch_pos.items()}
            logger.info("Corrected 180° coordinate rotation")
        else:
            return  # No correction needed

        raw.set_montage(make_dig_montage(ch_pos=corrected_pos, coord_frame='head'))

    @classmethod
    def _detect_inversion(cls, raw: "BaseRaw", ch_pos: dict) -> str:
        """
        Detect coordinate inversion type.
        Returns:
            'none'  - No inversion
            'x'     - Left-right flip (LAS/RAS)
            'xyz'   - Full 180° rotation (XYZ → -X-Y-Z)
        """
        # Get reference positions from standard montage
        std_montage = make_standard_montage('standard_1005')
        std_pos = std_montage.get_positions()['ch_pos']

        # Find common channels between data and standard
        common_chs = [ch for ch in raw.ch_names if ch in std_pos]
        if len(common_chs) < 3:
            return 'none'  # Not enough channels to compare

        # Calculate coordinate correlations
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

        # Determine inversion type
        if x_corr < -0.8 and y_corr < -0.8:
            return 'xy'  # Full 180° rotation
        elif x_corr < -0.8:
            return 'x'  # Left-right flip only
        else:
            return 'none'

    @classmethod
    def _find_montage_file(cls, file_path: Path) -> Optional[Path]:
        """Search for montage file in adjacent electrode_positions folder."""
        positions_dir = file_path.parent / "electrode_positions"
        if not positions_dir.exists():
            return None

        # Check for supported montage files (order determines priority)
        for ext in (".xml", ".sfp", ".elc", ".bvef"):
            for candidate in positions_dir.glob(f"*{ext}"):
                if candidate.is_file():
                    logger.debug(f"Found montage file: {candidate.name}")
                    return candidate
        return None

    @classmethod
    def _apply_montage(cls, raw: BaseRaw, montage_path: Path):
        """Apply montage to raw data."""
        try:
            if montage_path.suffix.lower() == ".xml":
                # Custom XML handling (e.g., ActiCap format)
                montage = cls._parse_xml_montage(montage_path)
            else:
                # Use MNE's built-in montage readers for other formats
                montage = make_standard_montage(montage_path.stem)

            raw.set_montage(montage)
            logger.info(f"Applied montage from {montage_path.name}")

        except Exception as e:
            logger.warning(f"Failed to apply montage: {str(e)}")

    @classmethod
    def _parse_xml_montage(cls, xml_path: Path) -> "DigMontage":
        """Parse ActiCap XML montage file and ensure proper RAS coordinates.

        Args:
            xml_path: Path to the ActiCap XML electrode positions file.

        Returns:
            DigMontage with corrected coordinates (RAS system).

        Raises:
            ValueError: If critical channels are missing or coordinates are invalid.
        """
        try:
            import xml.etree.ElementTree as ET
            from mne.channels import DigMontage

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Initialize coordinate storage
            ch_pos = {}
            fiducials = {}
            coord_frame = 'ras'  # Target coordinate frame

            # Names of fiducial points in ActiCap XML
            fiducial_tags = {
                'nasion': ['Nz', 'NASION'],
                'lpa': ['LPA', 'LEFT'],
                'rpa': ['RPA', 'RIGHT']
            }

            for channel in root.findall(".//CHANNEL"):
                label_elem = channel.find("LABEL")
                if label_elem is None:
                    continue

                label = label_elem.text.strip().upper()

                # Skip non-EEG channels
                if label in ('HEOG', 'VEOG', 'TRIGGER', 'ECG'):
                    continue

                # Get coordinates (adjust XML tags as needed)
                x = float(channel.find("X").text)  # Right-Left (positive = right)
                y = float(channel.find("Y").text)  # Anterior-Posterior (positive = front)
                z = float(channel.find("Z").text)  # Superior-Inferior (positive = up)

                # Fix common ActiCap coordinate issues:
                # 1. If Fp1 is at the back (Y < 0), flip Y-axis
                # 2. If left/right are reversed, flip X-axis
                if label == 'FP1' and y < 0:
                    y *= -1  # Flip front-back

                ch_pos[label] = [x, y, z]

                # Identify fiducials
                for fid_type, fid_names in fiducial_tags.items():
                    if label in fid_names and fid_type not in fiducials:
                        fiducials[fid_type] = [x, y, z]

            # Validate we have required channels
            required_channels = ['FZ', 'CZ', 'PZ', 'OZ']
            missing = [ch for ch in required_channels if ch not in ch_pos]
            if missing:
                raise ValueError(f"Missing required channels: {missing}")

            # Create montage
            montage = DigMontage(
                ch_pos=ch_pos,
                coord_frame=coord_frame,
                **fiducials
            )

            logger.info(f"Loaded montage from {xml_path.name} with {len(ch_pos)} channels")
            return montage

        except Exception as e:
            logger.error(f"Failed to parse XML montage: {str(e)}")
            raise RuntimeError(f"XML montage parsing failed: {str(e)}") from e

    @classmethod
    def supports_format(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in ('.dap', '.dat', '.ceo', '.cdt', '.rs3')

    @classmethod
    def _validate_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Curry file not found: {file_path}")