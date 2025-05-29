# utils/quality_tracker.py

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import numpy as np
from loguru import logger


class QualityTracker:
    """Tracks quality metrics during pipeline processing"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.metrics = {}
        self.start_time = datetime.now()

        # Ensure quality directory exists
        self.quality_dir = self.results_dir / "quality"
        self.quality_dir.mkdir(exist_ok=True)

    def track_participant_start(self, participant_id: str):
        """Initialize tracking for a participant"""
        if participant_id not in self.metrics:
            self.metrics[participant_id] = {
                'conditions': {},
                'start_time': datetime.now().isoformat(),
                'completed': False
            }

    def track_condition_start(self, participant_id: str, condition_name: str):
        """Initialize tracking for a condition"""
        self.track_participant_start(participant_id)

        if condition_name not in self.metrics[participant_id]['conditions']:
            self.metrics[participant_id]['conditions'][condition_name] = {
                'stages': {},
                'completion': {'success': False, 'error': None},
                'start_time': datetime.now().isoformat()
            }

    def track_stage(self, participant_id: str, condition_name: str,
                    stage_name: str, metrics: Dict[str, Any]):
        """Track quality metrics for a specific stage"""
        self.track_condition_start(participant_id, condition_name)

        # Store stage metrics
        condition_data = self.metrics[participant_id]['conditions'][condition_name]
        condition_data['stages'][stage_name] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        logger.debug(f"Tracked {stage_name} for {participant_id}/{condition_name}: {metrics}")

    def track_completion(self, participant_id: str, condition_name: str,
                         success: bool, error: Optional[str] = None):
        """Track whether condition completed successfully"""
        if participant_id in self.metrics:
            if condition_name in self.metrics[participant_id]['conditions']:
                completion_data = self.metrics[participant_id]['conditions'][condition_name]['completion']
                completion_data['success'] = success
                completion_data['error'] = error
                completion_data['end_time'] = datetime.now().isoformat()

                logger.info(f"Condition {condition_name} for {participant_id}: {'SUCCESS' if success else 'FAILED'}")

    def track_participant_completion(self, participant_id: str):
        """Mark participant as completed"""
        if participant_id in self.metrics:
            self.metrics[participant_id]['completed'] = True
            self.metrics[participant_id]['end_time'] = datetime.now().isoformat()

    def get_bad_channel_summary(self, participant_id: str, condition_name: str) -> Dict:
        """Extract bad channel information for a participant/condition"""
        try:
            stages = self.metrics[participant_id]['conditions'][condition_name]['stages']

            # Look for bad channel info in detect_bad_channels stage
            if 'detect_bad_channels' in stages:
                return stages['detect_bad_channels']['metrics']

            return {'bad_channels': [], 'interpolated': 0}
        except KeyError:
            return {'bad_channels': [], 'interpolated': 0}

    def get_epoch_rejection_summary(self, participant_id: str, condition_name: str) -> Dict:
        """Extract epoch rejection information"""
        try:
            stages = self.metrics[participant_id]['conditions'][condition_name]['stages']

            if 'epoch' in stages:
                return stages['epoch']['metrics']

            return {'total_epochs': 0, 'rejected_epochs': 0, 'rejection_rate': 0}
        except KeyError:
            return {'total_epochs': 0, 'rejected_epochs': 0, 'rejection_rate': 0}

    def get_ica_summary(self, participant_id: str, condition_name: str) -> Dict:
        """Extract ICA artifact removal information"""
        try:
            stages = self.metrics[participant_id]['conditions'][condition_name]['stages']

            if 'blink_artifact' in stages:
                return stages['blink_artifact']['metrics']

            return {'components_removed': 0, 'eog_components': [], 'ecg_components': []}
        except KeyError:
            return {'components_removed': 0, 'eog_components': [], 'ecg_components': []}

    def save_metrics(self):
        """Export metrics to JSON file"""
        metrics_file = self.quality_dir / "quality_metrics.json"

        # Add summary statistics
        summary_data = {
            'dataset_info': {
                'total_participants': len(self.metrics),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'completed_participants': sum(1 for p in self.metrics.values() if p.get('completed', False))
            },
            'participants': self.metrics
        }

        with open(metrics_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"Quality metrics saved to: {metrics_file}")
        return metrics_file


# Helper functions for extracting common metrics
def extract_bad_channel_metrics(raw, stage_result=None) -> Dict:
    """Extract bad channel metrics from raw data"""
    if hasattr(raw, '_bad_channel_metrics'):
        # Use detailed metrics if available
        return raw._bad_channel_metrics
    else:
        # Fallback to basic metrics
        bad_channels = raw.info.get('bads', [])
        return {
            'bad_channels': bad_channels,
            'n_bad_channels': len(bad_channels),
            'original_bads': [],
            'detected_bads': bad_channels,
            'interpolation_attempted': False
        }

def extract_epoch_metrics(epochs) -> Dict:
    """Extract epoch rejection metrics from epochs"""
    if hasattr(epochs, 'drop_log'):
        total_epochs = len(epochs.drop_log)
        rejected_epochs = sum(1 for log in epochs.drop_log if len(log) > 0)
        rejection_rate = (rejected_epochs / total_epochs) * 100 if total_epochs > 0 else 0

        # Count rejection reasons
        rejection_reasons = {}
        for log in epochs.drop_log:
            for reason in log:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        return {
            'total_epochs': total_epochs,
            'kept_epochs': len(epochs),
            'rejected_epochs': rejected_epochs,
            'rejection_rate': round(rejection_rate, 2),
            'rejection_reasons': rejection_reasons
        }

    return {
        'total_epochs': len(epochs),
        'kept_epochs': len(epochs),
        'rejected_epochs': 0,
        'rejection_rate': 0,
        'rejection_reasons': {}
    }


def extract_ica_metrics(excluded_components, eog_components=None, ecg_components=None) -> Dict:
    """Extract ICA artifact removal metrics"""
    return {
        'components_removed': len(excluded_components),
        'excluded_components': excluded_components,
        'eog_components': eog_components or [],
        'ecg_components': ecg_components or [],
        'n_eog_components': len(eog_components) if eog_components else 0,
        'n_ecg_components': len(ecg_components) if ecg_components else 0
    }