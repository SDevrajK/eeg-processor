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

    def __init__(self, results_dir: Union[str, Path]) -> None:
        self.results_dir: Path = Path(results_dir)
        self.metrics: Dict[str, Any] = {}
        self.start_time: datetime = datetime.now()

        self.memory_thresholds: Dict[str, int] = {
            'warning': 70,  # 70% system memory used
            'critical': 85,  # 85% system memory used
            'abort': 95  # 95% system memory used (stop processing)
        }

        # Ensure quality directory exists
        self.quality_dir: Path = self.results_dir / "quality"
        self.quality_dir.mkdir(exist_ok=True)

    def track_participant_start(self, participant_id: str) -> None:
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
        Main entry point for stage quality tracking with enhanced memory analysis.

        Intelligently extracts metrics based on stage type and available data.
        """
        logger.info(f"Tracking quality for stage '{stage_name}' - {participant_id}/{condition_name}")

        try:
            # Extract metrics based on stage type
            metrics = self._extract_stage_metrics(input_data, output_data, stage_name)

            if memory_metrics:
                # Simple memory tracking
                memory_data = self._extract_memory_metrics(memory_metrics)
                metrics['memory'] = memory_data
                
                # Check for memory issues with user-friendly messages
                memory_issues = self._detect_memory_issues(memory_data)
                if memory_issues:
                    for issue in memory_issues:
                        logger.warning(f"{stage_name}: {issue}")

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

        elif stage_name == "remove_blinks_emcp":
            return self._extract_emcp_metrics(output_data)

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
        raw_total_epochs = len(drop_log)
        
        # Count only actually rejected epochs, excluding 'IGNORED' entries
        rejected_epochs = 0
        rejection_reasons = {}
        
        for log in drop_log:
            if len(log) > 0:
                # Filter out 'IGNORED' entries - these are not actual rejections
                actual_reasons = [reason for reason in log if reason != 'IGNORED']
                if actual_reasons:
                    rejected_epochs += 1
                    for reason in actual_reasons:
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        kept_epochs = len(epochs_data)
        
        # Calculate actual total from kept + rejected to handle drop_log inconsistencies
        actual_total_epochs = kept_epochs + rejected_epochs
        rejection_rate = (rejected_epochs / actual_total_epochs) * 100 if actual_total_epochs > 0 else 0

        # Log warning if there's a mismatch between drop_log and actual epochs
        if raw_total_epochs != actual_total_epochs:
            logger.warning(f"Drop log length ({raw_total_epochs}) doesn't match actual epochs processed "
                         f"({actual_total_epochs}). Using actual count for accurate metrics.")

        return {
            'total_epochs': actual_total_epochs,
            'kept_epochs': kept_epochs,
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

    def _extract_emcp_metrics(self, output_data) -> Dict[str, Any]:
        """Extract essential EMCP blink correction metrics"""
        
        # Check if processing function stored detailed metrics
        if hasattr(output_data, '_emcp_metrics'):
            metrics = output_data._emcp_metrics.copy()
            logger.debug(f"Found stored EMCP metrics: {metrics['method']} method, "
                        f"{metrics['blink_events']} blinks, correlation: {metrics['mean_correlation']}")
            return metrics
        
        # Minimal fallback for when no detailed metrics are available
        logger.debug("Using minimal EMCP fallback metrics")
        return {
            'emcp_applied': True,
            'method': 'unknown',
            'correction_applied': True,
            'quality_flags': {'no_stored_metrics': True},
            'note': 'EMCP completed but no detailed metrics stored'
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

    def _extract_memory_metrics(self, memory_metrics: dict) -> Dict[str, Any]:
        """Extract simple memory metrics for user-friendly reporting"""
        return {
            'before_mb': memory_metrics.get('memory_before_mb', 0),
            'after_mb': memory_metrics.get('memory_after_mb', 0),
            'delta_mb': memory_metrics.get('memory_delta_mb', 0),
            'total_mb': memory_metrics.get('total_memory_mb', 0)
        }

    def _detect_memory_issues(self, memory_data: dict) -> List[str]:
        """Detect memory issues in user-friendly terms"""
        issues = []
        
        # Check for high memory usage
        if memory_data.get('total_mb', 0) > 8000:  # 8GB
            issues.append(f"High memory usage: {memory_data['total_mb']/1024:.1f} GB")
        
        # Check for large memory increases
        if memory_data.get('delta_mb', 0) > 2000:  # 2GB
            issues.append(f"Large memory increase: {memory_data['delta_mb']:.0f} MB")
        
        return issues

    def save_metrics(self):
        """Export metrics to JSON file with enhanced memory analysis summary"""
        metrics_file = self.quality_dir / "quality_metrics.json"

        # Generate memory analysis summary
        memory_summary = self._generate_memory_analysis_summary()

        # Add summary statistics
        summary_data = {
            'dataset_info': {
                'total_participants': len(self.metrics),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'completed_participants': sum(1 for p in self.metrics.values() if p.get('completed', False))
            },
            'memory_analysis_summary': memory_summary,
            'participants': self.metrics
        }

        with open(metrics_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"Quality metrics saved to: {metrics_file}")
        logger.info(f"Memory analysis summary: {memory_summary}")
        return metrics_file

    def _generate_memory_analysis_summary(self) -> Dict[str, Any]:
        """Generate comprehensive memory analysis summary across all participants and stages"""
        try:
            memory_summary = {
                'total_stages_analyzed': 0,
                'stages_with_high_memory': 0,
                'stages_with_leaks': 0,
                'stages_with_inefficient_use': 0,
                'average_memory_delta_mb': 0,
                'max_memory_delta_mb': 0,
                'memory_efficiency_scores': [],
                'problematic_stages': [],
                'memory_issues_by_stage': {}
            }
            
            total_memory_delta = 0
            max_delta = 0
            
            for participant_id, participant_data in self.metrics.items():
                for condition_name, condition_data in participant_data.get('conditions', {}).items():
                    for stage_name, stage_data in condition_data.get('stages', {}).items():
                        memory_data = stage_data.get('metrics', {}).get('memory', {})
                        
                        if memory_data:
                            memory_summary['total_stages_analyzed'] += 1
                            
                            # Extract memory metrics
                            if 'memory_analysis' in memory_data:
                                analysis = memory_data['memory_analysis']
                                delta_mb = analysis.get('rss_delta_mb', 0)
                                efficiency = analysis.get('efficiency_score', 0)
                                
                                total_memory_delta += delta_mb
                                max_delta = max(max_delta, delta_mb)
                                memory_summary['memory_efficiency_scores'].append(efficiency)
                                
                                # Check for issues
                                flags = analysis.get('analysis_flags', {})
                                if flags.get('high_memory_usage'):
                                    memory_summary['stages_with_high_memory'] += 1
                                if flags.get('potential_leak'):
                                    memory_summary['stages_with_leaks'] += 1
                                if flags.get('inefficient_memory_use'):
                                    memory_summary['stages_with_inefficient_use'] += 1
                                    
                            # Track memory issues by stage
                            memory_issues = stage_data.get('metrics', {}).get('memory_issues', [])
                            if memory_issues:
                                if stage_name not in memory_summary['memory_issues_by_stage']:
                                    memory_summary['memory_issues_by_stage'][stage_name] = []
                                memory_summary['memory_issues_by_stage'][stage_name].extend(memory_issues)
                                
                                # Add to problematic stages
                                stage_key = f"{participant_id}/{condition_name}/{stage_name}"
                                memory_summary['problematic_stages'].append({
                                    'stage_key': stage_key,
                                    'issues': memory_issues
                                })
            
            # Calculate averages
            if memory_summary['total_stages_analyzed'] > 0:
                memory_summary['average_memory_delta_mb'] = total_memory_delta / memory_summary['total_stages_analyzed']
                memory_summary['max_memory_delta_mb'] = max_delta
                
            if memory_summary['memory_efficiency_scores']:
                memory_summary['average_efficiency_score'] = np.mean(memory_summary['memory_efficiency_scores'])
                memory_summary['min_efficiency_score'] = np.min(memory_summary['memory_efficiency_scores'])
                
            # Remove raw efficiency scores from summary to save space
            memory_summary.pop('memory_efficiency_scores', None)
                
            return memory_summary
            
        except Exception as e:
            logger.error(f"Error generating memory analysis summary: {e}")
            return {'error': str(e)}


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
        raw_total_epochs = len(epochs.drop_log)
        
        # Count only actually rejected epochs, excluding 'IGNORED' entries
        rejected_epochs = 0
        rejection_reasons = {}
        
        for log in epochs.drop_log:
            if len(log) > 0:
                # Filter out 'IGNORED' entries - these are not actual rejections
                actual_reasons = [reason for reason in log if reason != 'IGNORED']
                if actual_reasons:
                    rejected_epochs += 1
                    for reason in actual_reasons:
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        kept_epochs = len(epochs)
        
        # Use actual total (kept + rejected) instead of drop_log length for accurate metrics
        actual_total_epochs = kept_epochs + rejected_epochs
        rejection_rate = (rejected_epochs / actual_total_epochs) * 100 if actual_total_epochs > 0 else 0

        return {
            'total_epochs': actual_total_epochs,
            'kept_epochs': kept_epochs,
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