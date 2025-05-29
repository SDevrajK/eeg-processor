from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Participant:
    id: str
    file_path: Path  # This will now store the FULL resolved path
    conditions: List[str] = None
    current_condition: Optional[str] = None


class ParticipantHandler:
    def __init__(self, config):
        self.config = config
        self.participants = self._load_participants()
        self.current = None

    def _load_participants(self) -> List[Participant]:
        """Create participants with properly resolved paths"""
        participants = []
        raw_data_path = Path(self.config.raw_data_path).resolve()

        for file_spec in self.config.participants:
            # Convert to Path object if needed
            file_path = Path(file_spec) if isinstance(file_spec, str) else file_spec

            # Resolve relative to raw_data directory
            if not file_path.is_absolute():
                file_path = (raw_data_path / file_path).resolve()

            # Case-insensitive file search
            if not file_path.exists():
                file_path = self._find_matching_file(raw_data_path, file_path.name)
                if not file_path:
                    raise FileNotFoundError(
                        f"Participant file not found in {raw_data_path}. "
                        f"Tried: {file_spec}"
                    )

            participants.append(
                Participant(
                    id=file_path.stem,
                    file_path=file_path,
                    conditions=[cond['name'] for cond in self.config.conditions]
                )
            )
        return participants

    def _find_matching_file(self, directory: Path, filename: str) -> Optional[Path]:
        """Case-insensitive file search in directory"""
        target = filename.lower()
        for f in directory.iterdir():
            if f.name.lower() == target:
                return f.resolve()
        return None