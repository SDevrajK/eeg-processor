from .quality_tracker import QualityTracker
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from loguru import logger


class SessionQualityTracker(QualityTracker):
    """
    Extends QualityTracker to handle multi-config sessions
    Minimal changes - just adds session context and config tracking
    """

    def __init__(self, session_dir: Path, session_name: str):
        # Initialize with session quality directory
        session_quality_dir = session_dir / "quality"
        super().__init__(session_quality_dir)

        self.session_name = session_name
        self.session_dir = session_dir
        self.current_config = None

        # Add session info to metrics
        self.metrics['session_info'] = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'configs_processed': [],
            'end_time': None
        }

        # Track participant availability across configs
        self.availability = {}  # participant_id -> config_name -> availability_info

    def set_config_context(self, config_name: str):
        """Set current config being processed"""
        self.current_config = config_name

        if config_name not in self.metrics['session_info']['configs_processed']:
            self.metrics['session_info']['configs_processed'].append(config_name)

        logger.info(f"Session quality tracker: processing config '{config_name}'")

    def track_participant_availability(self, participant_id: str, config_name: str,
                                       status: str, filename: Optional[str] = None):
        """Track which participants are available vs missing for each config"""
        if participant_id not in self.availability:
            self.availability[participant_id] = {}

        self.availability[participant_id][config_name] = {
            'status': status,  # 'available' or 'missing'
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }

    def finalize_session(self):
        """Finalize session and prepare for reporting"""
        self.metrics['session_info']['end_time'] = datetime.now().isoformat()
        self.metrics['participant_availability'] = self.availability

        # Call parent's save_metrics
        self.save_metrics()

        logger.success(f"Session '{self.session_name}' finalized")

    def generate_session_reports(self):
        """Generate session-wide quality reports"""
        try:
            from .session_quality_reporter import generate_session_quality_reports
            return generate_session_quality_reports(self.session_dir)
        except Exception as e:
            logger.error(f"Failed to generate session reports: {str(e)}")
            return None, []