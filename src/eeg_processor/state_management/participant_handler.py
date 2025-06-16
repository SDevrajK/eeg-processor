from pathlib import Path
from typing import List, Optional, Dict, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class Participant:
    id: str
    file_path: Path
    conditions: List[str] = None
    current_condition: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ParticipantHandler:
    def __init__(self, config):
        self.config = config
        self.participants = self._load_participants()
        self.current = None

        logger.info(f"Participant handler initialized with {len(self.participants)} participants")

        # Log participant info for debugging
        for participant in self.participants:
            logger.debug(f"Participant {participant.id}: {participant.file_path}")

    def _load_participants(self) -> List[Participant]:
        """Create participants with properly resolved paths - supports both dict and list formats"""
        participants = []
        raw_data_path = Path(self.config.raw_data_dir).resolve()

        # Handle the config.participants which could be:
        # 1. List[Path] (old format from config loader)
        # 2. Dict[str, str] (new format: {participant_id: filename})
        # 3. List[str] (legacy list format)

        participants_data = self.config.participants

        if isinstance(participants_data, dict):
            # New dictionary format: {participant_id: filename or {file: filename, ...}}
            for participant_id, participant_info in participants_data.items():
                # Handle both simple and detailed formats
                if isinstance(participant_info, str):
                    # Simple format: participant_id: "filename.ext"
                    filename = participant_info
                    metadata = {}
                elif isinstance(participant_info, dict):
                    # Detailed format: participant_id: {file: "filename.ext", age: 25, ...}
                    filename = participant_info.get('file')
                    if not filename:
                        raise ValueError(f"Participant '{participant_id}' missing required 'file' property")
                    # Extract metadata (everything except 'file')
                    metadata = {k: v for k, v in participant_info.items() if k != 'file'}
                else:
                    raise ValueError(f"Participant '{participant_id}' must be either a filename string or dictionary with metadata")
                
                file_path = self._resolve_participant_file(raw_data_path, filename)
                participants.append(
                    Participant(
                        id=participant_id,
                        file_path=file_path,
                        conditions=[cond['name'] for cond in self.config.conditions],
                        metadata=metadata
                    )
                )

        elif isinstance(participants_data, list):
            # List format - could be Path objects or strings
            for file_spec in participants_data:
                if hasattr(file_spec, 'id') and hasattr(file_spec, 'file_path'):
                    # Already Participant objects from updated config loader
                    participants.append(file_spec)
                else:
                    # Legacy format: strings or Path objects
                    file_path = self._resolve_participant_file(raw_data_path, file_spec)
                    participant_id = file_path.stem  # Generate ID from filename
                    participants.append(
                        Participant(
                            id=participant_id,
                            file_path=file_path,
                            conditions=[cond['name'] for cond in self.config.conditions]
                        )
                    )
        else:
            raise ValueError(f"Unsupported participants format: {type(participants_data)}")

        return participants

    def _resolve_participant_file(self, raw_data_path: Path, file_spec: Union[str, Path]) -> Path:
        """Resolve participant file path with case-insensitive search"""
        # Convert to Path object if needed
        file_path = Path(file_spec) if isinstance(file_spec, str) else file_spec

        # Resolve relative to raw_data directory
        if not file_path.is_absolute():
            file_path = (raw_data_path / file_path).resolve()

        # Case-insensitive file search
        if not file_path.exists():
            found_file = self._find_matching_file(raw_data_path, file_path.name)
            if not found_file:
                raise FileNotFoundError(
                    f"Participant file not found in {raw_data_path}. "
                    f"Tried: {file_path} (from spec: {file_spec})"
                )
            file_path = found_file

        return file_path

    def _find_matching_file(self, directory: Path, filename: str) -> Optional[Path]:
        """Case-insensitive file search in directory"""
        target = filename.lower()
        for f in directory.iterdir():
            if f.name.lower() == target:
                return f.resolve()
        return None

    def get_participant_by_id(self, participant_id: str) -> Optional[Participant]:
        """Get participant by ID"""
        for participant in self.participants:
            if participant.id == participant_id:
                return participant
        return None

    def get_participant_ids(self) -> List[str]:
        """Get list of all participant IDs"""
        return [p.id for p in self.participants]

    def get_participant_files(self) -> List[Path]:
        """Get list of all participant file paths (for backward compatibility)"""
        return [p.file_path for p in self.participants]

    def validate_participant_files(self) -> List[str]:
        """
        Validate that all participant files exist
        Returns list of missing files
        """
        missing_files = []
        for participant in self.participants:
            if not participant.file_path.exists():
                missing_files.append(f"{participant.id}: {participant.file_path}")

        if missing_files:
            logger.warning(f"Missing participant files: {missing_files}")

        return missing_files

    def filter_participants(self, participant_ids: List[str]) -> List[Participant]:
        """
        Filter participants by IDs
        Returns subset of participants that match the provided IDs
        """
        filtered = []
        for participant_id in participant_ids:
            participant = self.get_participant_by_id(participant_id)
            if participant:
                filtered.append(participant)
            else:
                logger.warning(f"Participant ID not found: {participant_id}")

        return filtered

    def get_participant_info(self) -> dict:
        """Get summary information about participants"""
        return {
            'total_participants': len(self.participants),
            'participant_ids': [p.id for p in self.participants],
            'file_extensions': list(set(p.file_path.suffix for p in self.participants)),
            'base_directory': str(self.config.raw_data_dir),
            'missing_files': self.validate_participant_files()
        }

    def add_metadata(self, csv_path: str, participant_id_column: str = "participant_id",
                     data_types: Optional[Dict[str, str]] = None):
        """
        Add metadata to participants from CSV file.

        Args:
            csv_path: Path to CSV file containing metadata
            participant_id_column: Column name containing participant IDs
            data_types: Optional dict mapping column names to data types
                       (e.g., {'age': 'int', 'score': 'float'})

        Raises:
            ValueError: If participant IDs don't match exactly
            FileNotFoundError: If CSV file doesn't exist
        """
        import pandas as pd

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        # Load CSV
        try:
            metadata_df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV {csv_path}: {e}")

        # Validate participant_id column exists
        if participant_id_column not in metadata_df.columns:
            raise ValueError(f"Column '{participant_id_column}' not found in {csv_path}. "
                             f"Available columns: {list(metadata_df.columns)}")

        # Get participant IDs from both sources
        config_ids = set(self.get_participant_ids())
        csv_ids = set(metadata_df[participant_id_column].astype(str))  # Ensure string comparison

        # Exact match validation - FATAL ERRORS
        missing_in_csv = config_ids - csv_ids
        missing_in_config = csv_ids - config_ids

        if missing_in_csv:
            raise ValueError(f"Participants in config but missing from CSV {csv_path}: {missing_in_csv}")
        if missing_in_config:
            raise ValueError(f"Participants in CSV {csv_path} but missing from config: {missing_in_config}")

        # Apply data type conversions if specified
        if data_types:
            for column, dtype in data_types.items():
                if column in metadata_df.columns and column != participant_id_column:
                    try:
                        if dtype == 'int':
                            metadata_df[column] = pd.to_numeric(metadata_df[column], errors='coerce').astype('Int64')
                        elif dtype == 'float':
                            metadata_df[column] = pd.to_numeric(metadata_df[column], errors='coerce')
                        elif dtype == 'bool':
                            metadata_df[column] = metadata_df[column].astype(bool)
                        # 'str' or any other type - leave as is
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{column}' to {dtype}: {e}")

        # Add metadata to participant objects
        added_columns = []
        for _, row in metadata_df.iterrows():
            participant_id = str(row[participant_id_column])
            participant = self.get_participant_by_id(participant_id)

            # Add all columns except the ID column as metadata
            for column, value in row.items():
                if column != participant_id_column:
                    # Handle NaN values from pandas
                    if pd.isna(value):
                        value = None
                    participant.metadata[column] = value
                    if column not in added_columns:
                        added_columns.append(column)

        logger.info(f"Added metadata from {csv_path}: {added_columns} for {len(config_ids)} participants")
        return self