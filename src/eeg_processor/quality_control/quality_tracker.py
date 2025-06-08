from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne import Epochs


class QualityTracker:
    """Tracks quality metrics during pipeline processing with intelligent stage detection"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.metrics = {}
        self.start_time = datetime.now()

        self.memory_thresholds = {
            'warning': 70,  # 70% system memory used
            'critical': 85,  # 85% system memory used
            'abort': 95  # 95% system memory used (stop processing)
        }

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

    def track_stage_data(self, input_data, output_data, stage_name: str,
                         participant_id: str, condition_name: str, memory_metrics: dict = None):
        """
        Main entry point for stage quality tracking.

        Intelligently extracts metrics based on stage type and available data.
        """
        logger.info(f"Tracking quality for stage '{stage_name}' - {participant_id}/{condition_name}")

        try:
            # Extract metrics based on stage type
            metrics = self._extract_stage_metrics(input_data, output_data, stage_name)

            if memory_metrics:
                metrics['memory'] = memory_metrics

                # Log memory warnings
                pressure_level = memory_metrics.get('pressure_level', 'normal')
                if pressure_level in ['warning', 'critical']:
                    logger.warning(f"Memory pressure {pressure_level} after {stage_name}: "
                                   f"{memory_metrics['memory_after']['used_percent']:.1f}% system memory")

            if metrics:
                # Store the metrics
                self.track_stage(participant_id, condition_name, stage_name, metrics)
                logger.info(
                    f"Quality metrics captured for {stage_name}: {self._summarize_metrics(stage_name, metrics)}")
            else:
                logger.debug(f"No specific metrics available for stage {stage_name}")

        except Exception as e:
            logger.error(f"Failed to track quality for {stage_name}: {e}")
            # Store error information
            self.track_stage(participant_id, condition_name, stage_name, {
                'error': str(e),
                'stage_completed': False,
                'timestamp': datetime.now().isoformat()
            })

    def _extract_stage_metrics(self, input_data, output_data, stage_name: str) -> Dict[str, Any]:
        """Extract metrics based on stage type and available data"""

        if stage_name == "detect_bad_channels":
            return self._extract_bad_channel_metrics(input_data, output_data)

        elif stage_name == "epoch":
            return self._extract_epoch_metrics(output_data)

        elif stage_name == "blink_artifact":
            return self._extract_ica_metrics(input_data, output_data)

        elif stage_name == "filter":
            return self._extract_filter_metrics(output_data)

        elif stage_name == "rereference":
            return self._extract_reference_metrics(output_data)

        elif stage_name in ["crop", "adjust_events", "segment_condition"]:
            return self._extract_generic_metrics(input_data, output_data, stage_name)

        else:
            # Generic tracking for unknown stages
            return {
                'stage_completed': True,
                'data_type': type(output_data).__name__,
                'timestamp': datetime.now().isoformat()
            }

    def _extract_bad_channel_metrics(self, input_data, output_data) -> Dict[str, Any]:
        """Extract bad channel detection metrics"""

        # Check if processing function stored detailed metrics
        if hasattr(output_data, '_bad_channel_metrics'):
            metrics = output_data._bad_channel_metrics.copy()
            logger.debug(f"Found stored bad channel metrics: {metrics['n_detected']} detected")
            return metrics

        # Fallback: compute basic metrics from data
        logger.debug("Computing fallback bad channel metrics")

        input_bads = set(input_data.info.get('bads', []))
        output_bads = set(output_data.info.get('bads', []))
        detected_bads = output_bads - input_bads

        return {
            'original_bads': sorted(input_bads),
            'detected_bads': sorted(detected_bads),
            'final_bads': sorted(output_bads),
            'n_original': len(input_bads),
            'n_detected': len(detected_bads),
            'n_final': len(output_bads),
            'interpolation_attempted': len(output_bads) < len(input_bads) + len(detected_bads),
            'method': 'fallback_computation'
        }

    def _extract_epoch_metrics(self, epochs_data) -> Dict[str, Any]:
        """Extract epoch rejection metrics"""

        if not hasattr(epochs_data, 'drop_log'):
            logger.debug("Epochs object missing drop_log - using basic metrics")
            return {
                'total_epochs': len(epochs_data),
                'kept_epochs': len(epochs_data),
                'rejected_epochs': 0,
                'rejection_rate': 0,
                'method': 'no_drop_log'
            }

        # Extract detailed rejection information
        drop_log = epochs_data.drop_log
        total_epochs = len(drop_log)
        rejected_epochs = sum(1 for log in drop_log if len(log) > 0)
        rejection_rate = (rejected_epochs / total_epochs) * 100 if total_epochs > 0 else 0

        # Count rejection reasons
        rejection_reasons = {}
        for log in drop_log:
            for reason in log:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        return {
            'total_epochs': total_epochs,
            'kept_epochs': len(epochs_data),
            'rejected_epochs': rejected_epochs,
            'rejection_rate': round(rejection_rate, 2),
            'rejection_reasons': rejection_reasons,
            'method': 'drop_log_analysis'
        }

    def _extract_ica_metrics(self, input_data, output_data) -> Dict[str, Any]:
        """Extract ICA artifact removal metrics"""

        # Check if processing function stored detailed metrics
        if hasattr(output_data, '_ica_metrics'):
            metrics = output_data._ica_metrics.copy()
            logger.debug(f"Found stored ICA metrics: {metrics.get('n_components_excluded', 0)} components excluded")
            return metrics

        # Try to find ICA object stored on the data
        ica_obj = None
        for obj in [output_data, input_data]:
            if hasattr(obj, '_last_ica'):
                ica_obj = obj._last_ica
                break

        if ica_obj and hasattr(ica_obj, 'exclude'):
            logger.debug("Found ICA object - extracting metrics")
            excluded = ica_obj.exclude
            return {
                'n_components_fitted': getattr(ica_obj, 'n_components_', 'unknown'),
                'n_components_excluded': len(excluded),
                'excluded_components': sorted(excluded),
                'method': 'ica_object_extraction'
            }

        # Minimal fallback
        logger.debug("Using minimal ICA fallback metrics")
        return {
            'ica_applied': True,
            'method': 'minimal_fallback',
            'note': 'Update ICA function to store detailed metrics'
        }

    def _extract_filter_metrics(self, output_data) -> Dict[str, Any]:
        """Extract filtering metrics"""
        return {
            'filter_applied': True,
            'highpass': getattr(output_data.info, 'highpass', None),
            'lowpass': getattr(output_data.info, 'lowpass', None),
            'sfreq': output_data.info['sfreq'],
            'method': 'filter_info_extraction'
        }

    def _extract_reference_metrics(self, output_data) -> Dict[str, Any]:
        """Extract rereferencing metrics"""
        ref_info = output_data.info.get('custom_ref_applied', None)
        return {
            'rereference_applied': True,
            'reference_type': str(ref_info) if ref_info else 'standard',
            'n_channels': len(output_data.ch_names),
            'method': 'reference_info_extraction'
        }

    def _extract_generic_metrics(self, input_data, output_data, stage_name: str) -> Dict[str, Any]:
        """Extract generic metrics for data transformation stages"""

        metrics = {
            'stage_completed': True,
            'input_type': type(input_data).__name__,
            'output_type': type(output_data).__name__,
            'method': 'generic_tracking'
        }

        # Add specific information based on data types
        if hasattr(input_data, 'times') and hasattr(output_data, 'times'):
            metrics.update({
                'input_duration': input_data.times[-1] - input_data.times[0],
                'output_duration': output_data.times[-1] - output_data.times[0],
                'duration_change': (output_data.times[-1] - output_data.times[0]) - (
                            input_data.times[-1] - input_data.times[0])
            })

        if hasattr(input_data, 'ch_names') and hasattr(output_data, 'ch_names'):
            metrics.update({
                'input_channels': len(input_data.ch_names),
                'output_channels': len(output_data.ch_names),
                'channels_changed': input_data.ch_names != output_data.ch_names
            })

        return metrics

    def _summarize_metrics(self, stage_name: str, metrics: Dict[str, Any]) -> str:
        """Create a brief summary of metrics for logging"""

        if stage_name == "detect_bad_channels":
            return f"{metrics.get('n_detected', 0)} detected, {metrics.get('n_final', 0)} final"

        elif stage_name == "epoch":
            return f"{metrics.get('rejection_rate', 0)}% rejection rate"

        elif stage_name == "blink_artifact":
            return f"{metrics.get('n_components_excluded', 'unknown')} components excluded"

        elif stage_name == "filter":
            hp = metrics.get('highpass', 'None')
            lp = metrics.get('lowpass', 'None')
            return f"highpass={hp}, lowpass={lp}"

        else:
            return f"completed ({metrics.get('method', 'unknown')})"

    def track_stage(self, participant_id: str, condition_name: str,
                    stage_name: str, metrics: Dict[str, Any]):
        """Store stage metrics (called by track_stage_data)"""
        self.track_condition_start(participant_id, condition_name)

        # Store stage metrics
        condition_data = self.metrics[participant_id]['conditions'][condition_name]
        condition_data['stages'][stage_name] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Stored metrics for {stage_name}")

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


# Keep the existing helper functions for backward compatibility
def extract_bad_channel_metrics(raw, stage_result=None) -> Dict:
    """Legacy function - use QualityTracker.track_stage_data instead"""
    if hasattr(raw, '_bad_channel_metrics'):
        return raw._bad_channel_metrics
    else:
        bad_channels = raw.info.get('bads', [])
        return {
            'bad_channels': bad_channels,
            'n_bad_channels': len(bad_channels),
            'original_bads': [],
            'detected_bads': bad_channels,
            'interpolation_attempted': False
        }


def extract_epoch_metrics(epochs) -> Dict:
    """Legacy function - use QualityTracker.track_stage_data instead"""
    if hasattr(epochs, 'drop_log'):
        total_epochs = len(epochs.drop_log)
        rejected_epochs = sum(1 for log in epochs.drop_log if len(log) > 0)
        rejection_rate = (rejected_epochs / total_epochs) * 100 if total_epochs > 0 else 0

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
    """Legacy function - use QualityTracker.track_stage_data instead"""
    return {
        'components_removed': len(excluded_components),
        'excluded_components': excluded_components,
        'eog_components': eog_components or [],
        'ecg_components': ecg_components or [],
        'n_eog_components': len(eog_components) if eog_components else 0,
        'n_ecg_components': len(ecg_components) if ecg_components else 0
    }