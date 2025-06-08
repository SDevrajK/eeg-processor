from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import shutil
from datetime import datetime
import yaml

from .pipeline import EEGPipeline
from .quality_control.session_quality_tracker import SessionQualityTracker


class EEGSession:
    """
    Manages multi-config EEG processing sessions with unified participant tracking

    Example:
        session = EEGSession(session_name="RIEEG_2025-06-04",
                           results_base_dir="/path/to/results/")
        session.add_config("baseline_params.yml")
        session.add_config("exp_params.yml")
        session.add_config("control_params.yml")
        session.run_all()
        session.generate_reports()
    """

    def __init__(self, session_name: str, results_base_dir: str):
        self.session_name = session_name
        self.session_dir = Path(results_base_dir) / f"session_{session_name}"
        self.configs = []
        self.config_names = []
        self.unified_participants = {}

        # Initialize session quality tracker
        self.session_quality_tracker = None

        self._setup_session_directories()
        logger.info(f"EEG Session initialized: {self.session_name}")

    def _setup_session_directories(self):
        """Create session directory structure"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "configs").mkdir(exist_ok=True)
        (self.session_dir / "interim").mkdir(exist_ok=True)
        (self.session_dir / "processed").mkdir(exist_ok=True)
        (self.session_dir / "figures").mkdir(exist_ok=True)
        (self.session_dir / "quality").mkdir(exist_ok=True)

    def add_config(self, config_path: str, config_name: Optional[str] = None) -> 'EEGSession':
        """
        Add a config to the session

        Args:
            config_path: Path to the YAML config file
            config_name: Optional friendly name (defaults to filename stem)
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Generate config name
        if config_name is None:
            config_name = config_path.stem.replace("_processing_params", "").replace("RIEEG_", "")

        self.configs.append(str(config_path))
        self.config_names.append(config_name)

        # Copy config to session directory for traceability
        session_config_path = self.session_dir / "configs" / f"{config_name}_processing_params.yml"
        shutil.copy2(config_path, session_config_path)

        logger.info(f"Added config '{config_name}': {config_path}")
        return self

    def run_all(self):
        """Process all configs in sequence with unified quality tracking"""
        if not self.configs:
            raise ValueError("No configs added to session. Use add_config() first.")

        logger.info(f"Starting session processing: {len(self.configs)} configs")

        # Initialize session-wide quality tracker
        self.session_quality_tracker = SessionQualityTracker(
            session_dir=self.session_dir,
            session_name=self.session_name
        )

        # First pass: build unified participant mapping
        self._build_unified_participant_mapping()

        # Track availability for all participants across all configs
        self._track_all_participant_availability()

        # Process each config
        for i, (config_path, config_name) in enumerate(zip(self.configs, self.config_names)):
            logger.info(f"Processing config {i + 1}/{len(self.configs)}: {config_name}")

            try:
                self._process_config(config_path, config_name)
                logger.success(f"Completed config: {config_name}")

            except Exception as e:
                logger.error(f"Failed processing config '{config_name}': {str(e)}")
                continue

        # Finalize session tracking
        self.session_quality_tracker.finalize_session()
        logger.success(f"Session processing complete: {self.session_name}")

    def _build_unified_participant_mapping(self):
        """Build unified participant mapping across all configs"""
        logger.info("Building unified participant mapping...")

        self.unified_participants = {}

        for config_path, config_name in zip(self.configs, self.config_names):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            participants = config_data.get('paths', {}).get('participants', {})

            if not isinstance(participants, dict):
                raise ValueError(f"Config '{config_name}' must use dictionary format for participants")

            for participant_id, filename in participants.items():
                if participant_id not in self.unified_participants:
                    self.unified_participants[participant_id] = {}
                self.unified_participants[participant_id][config_name] = filename

        logger.info(
            f"Unified participant mapping: {len(self.unified_participants)} participants across {len(self.config_names)} configs")

        # Log participant availability
        for participant_id, config_files in self.unified_participants.items():
            available_configs = list(config_files.keys())
            missing_configs = [name for name in self.config_names if name not in available_configs]
            if missing_configs:
                logger.debug(f"{participant_id}: missing from {missing_configs}")

    def _process_config(self, config_path: str, config_name: str):
        """Process a single config with session context"""

        # Create pipeline with session-specific results directory
        pipeline = EEGPipeline()

        # Modify config to use session directory and inject session quality tracker
        modified_config = self._prepare_session_config(config_path, config_name)
        pipeline.load_config(modified_config)

        # Replace pipeline's quality tracker with session tracker
        pipeline.set_quality_tracker(self.session_quality_tracker)
        self.session_quality_tracker.set_config_context(config_name)

        # Run pipeline (uses existing logic)
        pipeline.run()

    def _prepare_session_config(self, config_path: str, config_name: str) -> str:
        """Create a modified config that uses session directories and filters participants"""

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Update results directory to session subdirectory
        config_data['paths']['results_dir'] = str(self.session_dir)

        # Filter participants to only those available for this config
        original_participants = config_data['paths']['participants']
        available_participants = {}

        for participant_id, filename in original_participants.items():
            if participant_id in self.unified_participants and config_name in self.unified_participants[participant_id]:
                available_participants[participant_id] = filename
            else:
                logger.debug(f"Participant {participant_id} not available for config {config_name}")

        config_data['paths']['participants'] = available_participants

        if not available_participants:
            raise ValueError(f"No participants available for config {config_name}")

        logger.info(f"Config {config_name}: {len(available_participants)} participants available")

        # Save modified config
        session_config_path = self.session_dir / "configs" / f"modified_{config_name}_config.yml"
        with open(session_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        return str(session_config_path)

    def _track_all_participant_availability(self):
        """Track participant availability across all configs before processing"""
        logger.info("Tracking participant availability across configs...")

        for participant_id, config_files in self.unified_participants.items():
            for config_name in self.config_names:
                if config_name in config_files:
                    # Participant has data for this config
                    self.session_quality_tracker.track_participant_availability(
                        participant_id=participant_id,
                        config_name=config_name,
                        status="available",
                        filename=config_files[config_name]
                    )
                else:
                    # Participant missing for this config
                    self.session_quality_tracker.track_participant_availability(
                        participant_id=participant_id,
                        config_name=config_name,
                        status="missing",
                        filename=None
                    )

    def generate_reports(self):
        """Generate unified session reports"""
        if not self.session_quality_tracker:
            logger.error("No session quality tracker available. Run run_all() first.")
            return None

        try:
            # Generate session-wide quality reports using existing system
            summary_path, participant_paths = self.session_quality_tracker.generate_session_reports()

            logger.success(f"Session quality reports generated:")
            logger.success(f"  Session summary: {summary_path}")
            logger.success(f"  Individual reports: {len(participant_paths)} files")

            return summary_path, participant_paths

        except Exception as e:
            logger.error(f"Failed to generate session reports: {str(e)}")
            return None

    def get_participant_summary(self) -> Dict:
        """Get summary of participant availability across configs"""
        summary = {
            'total_participants': len(self.unified_participants),
            'configs': self.config_names,
            'availability_matrix': {}
        }

        for participant_id, config_files in self.unified_participants.items():
            summary['availability_matrix'][participant_id] = {
                config_name: (config_name in config_files)
                for config_name in self.config_names
            }

        return summary

    def list_session_files(self) -> Dict[str, List[Path]]:
        """List all files generated in the session"""
        file_types = {
            'configs': list((self.session_dir / "configs").glob("*.yml")),
            'interim': list((self.session_dir / "interim").glob("*")),
            'processed': list((self.session_dir / "processed").glob("*")),
            'quality': list((self.session_dir / "quality").glob("*")),
            'figures': list((self.session_dir / "figures").glob("*"))
        }

        return file_types

    def get_session_participant_summary(self) -> Dict:
        """Get summary of participant availability and processing across session"""
        if not self.session_quality_tracker:
            return {'error': 'Session not processed yet. Run run_all() first.'}

        summary = {
            'session_name': self.session_name,
            'total_participants': len(self.unified_participants),
            'total_configs': len(self.config_names),
            'config_names': self.config_names,
            'participants': {}
        }

        for participant_id, config_files in self.unified_participants.items():
            participant_summary = {
                'available_configs': list(config_files.keys()),
                'missing_configs': [name for name in self.config_names if name not in config_files],
                'coverage_rate': len(config_files) / len(self.config_names) * 100,
                'processing_status': {}
            }

            # Add processing status if available
            if hasattr(self.session_quality_tracker,
                       'data') and participant_id in self.session_quality_tracker.data.get('participants', {}):
                participant_data = self.session_quality_tracker.data['participants'][participant_id]
                for config_name in config_files.keys():
                    if config_name in participant_data:
                        config_data = participant_data[config_name]
                        participant_summary['processing_status'][config_name] = {
                            'completed': config_data.get('completed', False),
                            'conditions_processed': len(config_data.get('conditions', {})),
                            'successful_conditions': sum(
                                1 for cond in config_data.get('conditions', {}).values()
                                if cond.get('completion', {}).get('success', False)
                            )
                        }

            summary['participants'][participant_id] = participant_summary

        return summary