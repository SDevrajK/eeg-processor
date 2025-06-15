from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import shutil
from datetime import datetime
import yaml
import json

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

        # Create combined session config (replaces individual modified configs)
        combined_config_path = self._create_combined_session_config()
        logger.info(f"Created combined session config: {combined_config_path}")

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

            participants = config_data.get('paths', {}).get('participants', [])

            # Handle both list and dictionary formats
            if isinstance(participants, list):
                # Convert list format to dict format using filename as both key and value
                participants_dict = {}
                for filename in participants:
                    # Use filename without extension as participant ID
                    participant_id = Path(filename).stem
                    participants_dict[participant_id] = filename
                participants = participants_dict
                logger.debug(f"Config '{config_name}': converted list format to dict format ({len(participants)} participants)")
            
            elif isinstance(participants, dict):
                logger.debug(f"Config '{config_name}': using existing dict format ({len(participants)} participants)")
            
            else:
                raise ValueError(f"Config '{config_name}' participants must be list or dict format, got {type(participants)}")

            # Build unified mapping
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

        # Get config data and pass directly to pipeline (no temp files)
        config_data = self._prepare_session_config_data(config_path, config_name)
        pipeline.load_config(config_data=config_data)

        # Replace pipeline's quality tracker with session tracker
        pipeline.set_quality_tracker(self.session_quality_tracker)
        self.session_quality_tracker.set_config_context(config_name)

        # Run pipeline (uses existing logic)
        pipeline.run()

    def _create_combined_session_config(self) -> str:
        """Create a single combined config that merges all session configs"""
        
        combined_config = {
            'paths': {
                'results_dir': str(self.session_dir),
                'participants': self.unified_participants  # Unified participant mapping
            },
            'session_info': {
                'session_name': self.session_name,
                'source_configs': self.config_names,
                'created_at': datetime.now().isoformat()
            },
            'config_groups': {}  # Each config becomes a group
        }
        
        # Merge each config as a separate group
        for config_path, config_name in zip(self.configs, self.config_names):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Store original config (minus paths) as a group
            config_group = {
                'source_file': str(Path(config_path).name),
                'raw_data_path': config_data['paths'].get('raw_data', ''),
                'conditions': config_data.get('conditions', []),
                'stages': config_data.get('stages', [])
            }
            
            # Add any other top-level keys (excluding paths)
            for key, value in config_data.items():
                if key not in ['paths', 'conditions', 'stages']:
                    config_group[key] = value
                    
            combined_config['config_groups'][config_name] = config_group
        
        # Save combined config
        combined_config_path = self.session_dir / "configs" / "session_combined_config.yml"
        with open(combined_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(combined_config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Created combined session config: {combined_config_path}")
        return str(combined_config_path)

    def _prepare_session_config_data(self, config_path: str, config_name: str) -> Dict:
        """Create config data for a specific config group from the combined config (no temp files)"""
        
        # Use the combined config if it exists, otherwise fall back to old method
        combined_config_path = self.session_dir / "configs" / "session_combined_config.yml"
        
        if not combined_config_path.exists():
            # Fall back to creating config data from original config
            return self._prepare_session_config_data_legacy(config_path, config_name)
            
        # Load combined config and extract specific config group
        with open(combined_config_path, 'r', encoding='utf-8') as f:
            combined_config = yaml.safe_load(f)
            
        if config_name not in combined_config['config_groups']:
            raise ValueError(f"Config group '{config_name}' not found in combined config")
            
        config_group = combined_config['config_groups'][config_name]
        
        # Create individual config data from combined config (return dict, not file)
        individual_config = {
            'paths': {
                'raw_data': config_group['raw_data_path'],
                'results_dir': combined_config['paths']['results_dir'],
                'participants': {
                    pid: files[config_name] 
                    for pid, files in combined_config['paths']['participants'].items()
                    if config_name in files
                }
            },
            'conditions': config_group['conditions'],
            'stages': config_group['stages']
        }
        
        # Add any additional keys from the config group
        for key, value in config_group.items():
            if key not in ['source_file', 'raw_data_path', 'conditions', 'stages']:
                individual_config[key] = value
        
        if not individual_config['paths']['participants']:
            raise ValueError(f"No participants available for config {config_name}")
            
        logger.info(f"Config {config_name}: {len(individual_config['paths']['participants'])} participants from combined config")
        
        return individual_config

    def _prepare_session_config_data_legacy(self, config_path: str, config_name: str) -> Dict:
        """Legacy method - create config data from original config (no temp files)"""

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Update results directory to session subdirectory
        config_data['paths']['results_dir'] = str(self.session_dir)

        # Filter participants to only those available for this config
        original_participants = config_data['paths']['participants']
        
        # Handle both list and dict formats
        if isinstance(original_participants, list):
            participants_dict = {}
            for filename in original_participants:
                participant_id = Path(filename).stem
                participants_dict[participant_id] = filename
            original_participants = participants_dict
        
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

        return config_data

    def _prepare_session_config(self, config_path: str, config_name: str) -> str:
        """Create a config for a specific config group from the combined config"""
        
        # Use the combined config if it exists, otherwise fall back to old method
        combined_config_path = self.session_dir / "configs" / "session_combined_config.yml"
        
        if not combined_config_path.exists():
            # Fall back to old method for backward compatibility
            return self._prepare_session_config_legacy(config_path, config_name)
            
        # Load combined config and extract specific config group
        with open(combined_config_path, 'r', encoding='utf-8') as f:
            combined_config = yaml.safe_load(f)
            
        if config_name not in combined_config['config_groups']:
            raise ValueError(f"Config group '{config_name}' not found in combined config")
            
        config_group = combined_config['config_groups'][config_name]
        
        # Create individual config from combined config
        individual_config = {
            'paths': {
                'raw_data': config_group['raw_data_path'],
                'results_dir': combined_config['paths']['results_dir'],
                'participants': {
                    pid: files[config_name] 
                    for pid, files in combined_config['paths']['participants'].items()
                    if config_name in files
                }
            },
            'conditions': config_group['conditions'],
            'stages': config_group['stages']
        }
        
        # Add any additional keys from the config group
        for key, value in config_group.items():
            if key not in ['source_file', 'raw_data_path', 'conditions', 'stages']:
                individual_config[key] = value
        
        if not individual_config['paths']['participants']:
            raise ValueError(f"No participants available for config {config_name}")
            
        logger.info(f"Config {config_name}: {len(individual_config['paths']['participants'])} participants from combined config")
        
        # Create temporary config file only if needed by pipeline
        # For now, we need to save it since pipeline.load_config() expects a file path
        individual_config_path = self.session_dir / "configs" / f"active_{config_name}_config.yml"
        with open(individual_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(individual_config, f, default_flow_style=False)
            
        return str(individual_config_path)
        
    def _prepare_session_config_legacy(self, config_path: str, config_name: str) -> str:
        """Legacy method - create individual modified config (for backward compatibility)"""

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

    def _reconstruct_session_quality_tracker(self):
        """Reconstruct session quality tracker from saved quality metrics"""
        quality_metrics_path = self.session_dir / "quality" / "quality_metrics.json"
        
        if not quality_metrics_path.exists():
            logger.warning(f"No quality metrics found at: {quality_metrics_path}")
            return False
            
        logger.info(f"Reconstructing session quality tracker from: {quality_metrics_path}")
        
        try:
            # Load the saved quality metrics
            with open(quality_metrics_path, 'r', encoding='utf-8') as f:
                saved_metrics = json.load(f)
            
            # Reconstruct session quality tracker
            self.session_quality_tracker = SessionQualityTracker(
                session_dir=self.session_dir,
                session_name=self.session_name
            )
            
            # Store the original session_dir before overwriting metrics
            original_session_dir = self.session_quality_tracker.session_dir
            
            # Restore the saved metrics data
            self.session_quality_tracker.metrics = saved_metrics
            
            # Restore the session_dir (it got overwritten by metrics assignment)
            self.session_quality_tracker.session_dir = original_session_dir
            
            # Extract session info if available
            session_info = saved_metrics.get('participants', {}).get('session_info', {})
            if session_info:
                self.session_quality_tracker.metrics['session_info'] = session_info
                
                # Restore config names and unified participants if we can extract them
                configs_processed = session_info.get('configs_processed', [])
                if configs_processed and not self.config_names:
                    self.config_names = configs_processed
                    logger.info(f"Restored config names from saved data: {self.config_names}")
            
            # Rebuild unified participants mapping from saved data if needed
            if not self.unified_participants:
                self._rebuild_unified_participants_from_saved_data(saved_metrics)
            
            # Mark the tracker as reconstructed so it knows to use saved data
            self.session_quality_tracker._reconstructed_from_saved_data = True
            
            logger.success(f"Successfully reconstructed session quality tracker")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconstruct session quality tracker: {str(e)}")
            return False

    def _rebuild_unified_participants_from_saved_data(self, saved_metrics: Dict):
        """Rebuild unified participants mapping from saved quality metrics"""
        try:
            # Extract participant data from saved metrics
            participants_data = saved_metrics.get('participants', {})
            
            # Look for participant availability data if it exists
            if 'participant_availability' in saved_metrics:
                availability_data = saved_metrics['participant_availability']
                self.unified_participants = {}
                
                for participant_id, config_data in availability_data.items():
                    self.unified_participants[participant_id] = {}
                    for config_name, avail_info in config_data.items():
                        if avail_info.get('status') == 'available' and avail_info.get('filename'):
                            self.unified_participants[participant_id][config_name] = avail_info['filename']
                
                logger.info(f"Rebuilt unified participants mapping from availability data: {len(self.unified_participants)} participants")
                
            else:
                # Fallback: try to infer from participant conditions data
                for participant_id, participant_info in participants_data.items():
                    if participant_id == 'session_info':
                        continue
                        
                    if participant_id not in self.unified_participants:
                        self.unified_participants[participant_id] = {}
                    
                    conditions = participant_info.get('conditions', {})
                    for condition_name, condition_data in conditions.items():
                        config_name = condition_data.get('config_name')
                        if config_name and config_name not in self.unified_participants[participant_id]:
                            # We don't have the original filename, but we can use a placeholder
                            self.unified_participants[participant_id][config_name] = f"{participant_id}_data"
                
                logger.info(f"Rebuilt unified participants mapping from conditions data: {len(self.unified_participants)} participants")
                
        except Exception as e:
            logger.warning(f"Could not rebuild unified participants mapping: {str(e)}")
            self.unified_participants = {}

    def generate_reports(self):
        """Generate unified session reports"""
        # If no live session tracker, try to reconstruct from saved data
        if not self.session_quality_tracker:
            logger.info("No live session quality tracker found. Attempting to reconstruct from saved data...")
            self._reconstruct_session_quality_tracker()

        # If still no tracker after reconstruction attempt, raise error
        if not self.session_quality_tracker:
            error_msg = "No session quality data found. Either run run_all() first or ensure session has been processed previously."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            # Generate session-wide quality reports using existing system
            summary_path, participant_paths = self.session_quality_tracker.generate_session_reports()

            logger.success(f"Session quality reports generated:")
            logger.success(f"  Session summary: {summary_path}")
            logger.success(f"  Individual reports: {len(participant_paths)} files")

            return summary_path, participant_paths

        except Exception as e:
            logger.error(f"Failed to generate session reports: {str(e)}")
            raise

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

    def is_session_processed(self) -> bool:
        """Check if the session has been processed (quality metrics exist)"""
        quality_metrics_path = self.session_dir / "quality" / "quality_metrics.json"
        return quality_metrics_path.exists()

    def get_session_status(self) -> Dict:
        """Get comprehensive session status information"""
        status = {
            'session_name': self.session_name,
            'session_dir': str(self.session_dir),
            'configs_added': len(self.configs),
            'config_names': self.config_names,
            'has_live_tracker': bool(self.session_quality_tracker),
            'has_saved_metrics': self.is_session_processed(),
            'can_generate_reports': False,
            'processing_status': 'unknown'
        }
        
        # Determine processing status
        if status['has_live_tracker']:
            status['processing_status'] = 'live_session'
            status['can_generate_reports'] = True
        elif status['has_saved_metrics']:
            status['processing_status'] = 'previously_processed'
            status['can_generate_reports'] = True
        elif status['configs_added'] > 0:
            status['processing_status'] = 'ready_to_process'
        else:
            status['processing_status'] = 'not_configured'
        
        return status

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