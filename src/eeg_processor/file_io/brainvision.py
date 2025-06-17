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
        
        # Handle .xhdr files (XML-based cropped BrainVision) with format conversion
        if file_path.suffix.lower() == '.xhdr':
            return BrainVisionLoader._load_xhdr_file(file_path, **kwargs)
        else:
            return BrainVisionLoader._load_vhdr_file(file_path, **kwargs)

    @staticmethod 
    def _load_vhdr_file(file_path: Path, **kwargs) -> BaseRaw:
        """Load standard .vhdr BrainVision file"""
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
    def _load_xhdr_file(file_path: Path, **kwargs) -> BaseRaw:
        """Load .xhdr (XML-based cropped BrainVision) file by converting to .vhdr format"""
        import tempfile
        import shutil
        import xml.etree.ElementTree as ET
        
        logger.info(f"Loading XML-based cropped BrainVision file (.xhdr): {file_path.name}")
        
        # Create temporary directory for converted files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            base_name = file_path.stem
            
            try:
                # Parse the XML header file
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Extract namespace if present
                namespace = ''
                if root.tag.startswith('{'):
                    namespace = root.tag.split('}')[0] + '}'
                
                # Convert XML to .vhdr format
                vhdr_content = BrainVisionLoader._convert_xml_to_vhdr(root, namespace, base_name)
                
                # Create temporary .vhdr file
                temp_vhdr = temp_dir_path / f"{base_name}.vhdr"
                temp_eeg = temp_dir_path / f"{base_name}.eeg"  
                temp_vmrk = temp_dir_path / f"{base_name}.vmrk"
                
                # Write converted .vhdr content
                with open(temp_vhdr, 'w', encoding='utf-8') as f:
                    f.write(vhdr_content)
                
                # Copy data file (.dat -> .eeg)
                orig_dat = file_path.parent / f"{base_name}.dat"
                if orig_dat.exists():
                    shutil.copy2(orig_dat, temp_eeg)
                    logger.debug(f"Copied {orig_dat.name} -> {temp_eeg.name}")
                else:
                    raise FileNotFoundError(f"Data file not found: {orig_dat}")
                
                # Convert marker file (.xmrk -> .vmrk) 
                orig_xmrk = file_path.parent / f"{base_name}.xmrk"
                if orig_xmrk.exists():
                    vmrk_content = BrainVisionLoader._convert_xmrk_to_vmrk(orig_xmrk, base_name)
                    with open(temp_vmrk, 'w', encoding='utf-8') as f:
                        f.write(vmrk_content)
                    logger.debug(f"Converted {orig_xmrk.name} -> {temp_vmrk.name}")
                else:
                    logger.warning(f"Marker file not found: {orig_xmrk}")
                    # Create empty marker file if missing
                    with open(temp_vmrk, 'w', encoding='utf-8') as f:
                        f.write("Brain Vision Data Exchange Marker File, Version 1.0\n")
                
                # Load using the temporary .vhdr file
                # For .xhdr files, we must preload data while temporary files exist
                # because MNE will try to access the files later after cleanup
                kwargs_copy = kwargs.copy()
                original_preload = kwargs_copy.get('preload', False)
                kwargs_copy['preload'] = True
                
                raw = BrainVisionLoader._load_vhdr_file(temp_vhdr, **kwargs_copy)
                
                # Log preload behavior
                if not original_preload:
                    logger.debug(f"Forced preload=True for .xhdr compatibility (user requested preload={original_preload})")
                
                # Now the data is loaded into memory, safe to clean up temp files
                logger.success(f"Successfully loaded cropped BrainVision file: {file_path.name}")
                
                return raw
                
            except ET.ParseError as e:
                raise ValueError(f"Invalid XML in .xhdr file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to convert .xhdr file {file_path}: {e}")
                raise

    @staticmethod
    def _convert_xml_to_vhdr(root, namespace: str, base_name: str) -> str:
        """Convert XML structure to .vhdr format string"""
        
        def find_elem(parent, tag):
            """Helper to find element with or without namespace"""
            elem = parent.find(f"{namespace}{tag}")
            return elem if elem is not None else parent.find(tag)
        
        # Extract basic information
        data_file_elem = find_elem(root, "DataFile")
        marker_file_elem = find_elem(root, "MarkerFile")
        sampling_interval_elem = find_elem(root, "SamplingInterval")
        data_orientation_elem = find_elem(root, "DataOrientation")
        
        # Extract binary format
        binary_format_elem = find_elem(root, "BinaryFormat")
        if binary_format_elem is not None:
            format_elem = find_elem(binary_format_elem, "Format")
            binary_format = format_elem.text if format_elem is not None else "IEEE_FLOAT_32"
        else:
            binary_format = "IEEE_FLOAT_32"
        
        # Map XML binary format to .vhdr format
        format_mapping = {
            "IEEE_FLOAT_32": "IEEE_FLOAT_32",
            "INT_16": "INT_16",
            "INT_32": "INT_32"
        }
        vhdr_binary_format = format_mapping.get(binary_format, "IEEE_FLOAT_32")
        
        # Extract channels
        channels_elem = find_elem(root, "Channels")
        channels = []
        if channels_elem is not None:
            for i, channel_elem in enumerate(channels_elem.findall(f"{namespace}Channel") or 
                                           channels_elem.findall("Channel"), 1):
                name_elem = find_elem(channel_elem, "Name")
                unit_elem = find_elem(channel_elem, "DataUnit")
                
                name = name_elem.text if name_elem is not None else f"Ch{i}"
                unit = unit_elem.text if unit_elem is not None else "µV"
                
                # Convert unit format
                if unit.lower() == "microvolt":
                    unit = "µV"
                
                # For XML format, we assume resolution of 1.0 since it's usually floating point
                resolution = "1.0"
                
                channels.append(f"Ch{i}={name},,{resolution},{unit}")
        
        # Build .vhdr content
        data_file = data_file_elem.text if data_file_elem is not None else f"{base_name}.eeg"
        marker_file = marker_file_elem.text if marker_file_elem is not None else f"{base_name}.vmrk"
        sampling_interval = sampling_interval_elem.text if sampling_interval_elem is not None else "2000"
        data_orientation = data_orientation_elem.text.upper() if data_orientation_elem is not None else "MULTIPLEXED"
        
        # Ensure data file has .eeg extension for MNE
        if not data_file.endswith('.eeg'):
            data_file = f"{base_name}.eeg"
        if not marker_file.endswith('.vmrk'):
            marker_file = f"{base_name}.vmrk"
        
        vhdr_content = f"""Brain Vision Data Exchange Header File Version 1.0
; Converted from XML format (.xhdr)

[Common Infos]
Codepage=UTF-8
DataFile={data_file}
MarkerFile={marker_file}
DataFormat=BINARY
DataOrientation={data_orientation}
NumberOfChannels={len(channels)}
SamplingInterval={sampling_interval}

[Binary Infos]
BinaryFormat={vhdr_binary_format}

[Channel Infos]
; Each entry: Ch<Channel number>=<Name>,<Reference channel name>,
; <Resolution in "Unit">,<Unit>, Future extensions..
; Fields are delimited by commas, some fields might be omitted (empty).
"""
        
        # Add channel information
        for channel_line in channels:
            vhdr_content += channel_line + "\n"
        
        vhdr_content += """
[Comment]
Converted from XML-based .xhdr format
"""
        
        return vhdr_content

    @staticmethod
    def _convert_xmrk_to_vmrk(xmrk_path: Path, base_name: str) -> str:
        """Convert XML marker file (.xmrk) to standard .vmrk format"""
        import xml.etree.ElementTree as ET
        
        try:
            # Parse the XML marker file
            tree = ET.parse(xmrk_path)
            root = tree.getroot()
            
            # Extract namespace if present
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'
            
            def find_elem(parent, tag):
                """Helper to find element with or without namespace"""
                elem = parent.find(f"{namespace}{tag}")
                return elem if elem is not None else parent.find(tag)
            
            # Build .vmrk content
            vmrk_content = f"""Brain Vision Data Exchange Marker File, Version 1.0
; Converted from XML format (.xmrk)

[Common Infos]
Codepage=UTF-8
DataFile={base_name}.eeg

[Marker Infos]
; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,
; <Size in data points>, <Channel number (0 = marker is related to all channels)>
; Fields are delimited by commas, some fields might be omitted (empty).
; Commas in type or description text are coded as "\\1".
"""
            
            # Extract markers
            markers_elem = find_elem(root, "Markers")
            marker_count = 0
            
            if markers_elem is not None:
                for marker_elem in markers_elem.findall(f"{namespace}Marker") or markers_elem.findall("Marker"):
                    marker_count += 1
                    
                    # Extract marker information
                    type_elem = find_elem(marker_elem, "Type")
                    desc_elem = find_elem(marker_elem, "Description")
                    pos_elem = find_elem(marker_elem, "Position")
                    points_elem = find_elem(marker_elem, "Points")
                    channel_elem = find_elem(marker_elem, "Channel")
                    date_elem = find_elem(marker_elem, "Date")
                    
                    # Extract values with defaults
                    marker_type = type_elem.text if type_elem is not None and type_elem.text else "Unknown"
                    description = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                    position = pos_elem.text if pos_elem is not None and pos_elem.text else "1"
                    points = points_elem.text if points_elem is not None and points_elem.text else "1"
                    
                    # Convert channel information
                    channel = "0"  # Default to all channels
                    if channel_elem is not None:
                        if channel_elem.text and channel_elem.text.lower() != "all":
                            try:
                                channel = str(int(channel_elem.text))
                            except ValueError:
                                channel = "0"
                    
                    # Handle date information (optional in .vmrk format)
                    date_info = ""
                    if date_elem is not None and date_elem.text:
                        date_str = date_elem.text
                        # Try to convert to timestamp format if it's a valid date
                        if date_str and date_str != "0001-01-01T00:00:00":
                            try:
                                from datetime import datetime
                                # Handle different datetime formats
                                if 'T' in date_str:
                                    # ISO format with T separator
                                    dt_str = date_str.replace('T', ' ').replace('Z', '')
                                    # Remove fractional seconds if present
                                    if '.' in dt_str:
                                        dt_str = dt_str.split('.')[0]
                                    dt = datetime.fromisoformat(dt_str)
                                else:
                                    dt = datetime.fromisoformat(date_str)
                                # Convert to BrainVision timestamp format (microseconds since epoch)
                                timestamp = int(dt.timestamp() * 1000000)
                                date_info = f",{timestamp}"
                            except Exception:
                                pass  # Skip date if conversion fails
                    
                    # Escape commas in type and description
                    marker_type = marker_type.replace(',', '\\1')
                    description = description.replace(',', '\\1')
                    
                    # Build marker line
                    marker_line = f"Mk{marker_count}={marker_type},{description},{position},{points},{channel}{date_info}"
                    vmrk_content += marker_line + "\n"
            
            if marker_count == 0:
                logger.warning(f"No markers found in {xmrk_path}")
            else:
                logger.debug(f"Converted {marker_count} markers from XML to .vmrk format")
            
            return vmrk_content
            
        except ET.ParseError as e:
            logger.error(f"Invalid XML in marker file {xmrk_path}: {e}")
            # Return minimal valid .vmrk file
            return f"""Brain Vision Data Exchange Marker File, Version 1.0
; Error parsing XML marker file

[Common Infos]
Codepage=UTF-8
DataFile={base_name}.eeg

[Marker Infos]
; No markers could be parsed from XML file
"""
        except Exception as e:
            logger.error(f"Failed to convert marker file {xmrk_path}: {e}")
            # Return minimal valid .vmrk file
            return f"""Brain Vision Data Exchange Marker File, Version 1.0
; Error converting marker file

[Common Infos]
Codepage=UTF-8
DataFile={base_name}.eeg

[Marker Infos]
; Marker conversion failed
"""

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
        return file_path.suffix.lower() in ['.vhdr', '.xhdr']

    @staticmethod
    def _validate_file(file_path: Path):
        """Validate all required companion files exist"""
        if not file_path.exists():
            raise FileNotFoundError(f"Primary file not found: {file_path}")

        base_path = file_path.with_suffix('')
        
        # Determine required companion files based on header file type
        if file_path.suffix.lower() == '.vhdr':
            # Standard BrainVision format
            required_suffixes = ['.vhdr', '.eeg', '.vmrk']
        elif file_path.suffix.lower() == '.xhdr':
            # Cropped BrainVision format
            required_suffixes = ['.xhdr', '.dat', '.xmrk']
        else:
            raise ValueError(f"Unsupported BrainVision header format: {file_path.suffix}")
        
        missing = [s for s in required_suffixes
                   if not base_path.with_suffix(s).exists()]

        if missing:
            file_type = "cropped BrainVision" if file_path.suffix.lower() == '.xhdr' else "BrainVision"
            raise FileNotFoundError(
                f"Missing {file_type} companion files: {missing}\n"
                f"Required for: {file_path.name}"
            )


# Convenience function remains for backward compatibility
def load_brainvision_file(file_path: Path, **kwargs) -> BaseRaw:
    return BrainVisionLoader.load(file_path, **kwargs)