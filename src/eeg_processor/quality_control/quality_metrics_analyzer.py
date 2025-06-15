# quality_control/quality_metrics_analyzer.py

from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
import numpy as np
from loguru import logger


class QualityMetricsAnalyzer:
    """
    Analyzes quality metrics data and computes comprehensive statistics.

    Handles all data processing and statistical computations for quality reporting.
    Focuses specifically on quality metrics (bad channels, rejection rates, ICA components)
    rather than raw EEG data processing.
    """

    def __init__(self, metrics_file: Path):
        self.metrics_file = Path(metrics_file)
        self.quality_dir = self.metrics_file.parent

        with open(self.metrics_file, 'r') as f:
            self.data = json.load(f)

        # Quality thresholds - could be configurable in future
        self.thresholds = {
            'bad_channels': {'warning': 4, 'critical': 8},
            'rejection_rate': {'warning': 15, 'critical': 30},
            'ica_components': {'warning': 16, 'critical': 24}
        }

        logger.info(f"QualityMetricsAnalyzer loaded metrics for {len(self.data['participants'])} participants")

    def compute_all_statistics(self) -> Dict:
        """
        Compute all statistics needed for quality reporting.

        Returns:
            Dictionary containing all computed statistics organized by category
        """
        participants = self.data['participants']
        dataset_info = self.data['dataset_info']

        return {
            'dataset_info': dataset_info,
            'participants': participants,
            'completion_stats': self.get_completion_statistics(participants),
            'quality_stats': self.get_quality_statistics(participants),
            'processing_stats': self.get_processing_statistics(participants)
        }

    def get_completion_statistics(self, participants: Dict) -> Dict:
        """Extract detailed completion statistics"""
        total_participants = len(participants)
        completed_participants = sum(1 for p in participants.values() if p['completed'])

        condition_success = {}
        condition_failures = {}
        stage_completion = {}

        for participant_data in participants.values():
            for condition_name, condition_data in participant_data['conditions'].items():
                success = condition_data['completion']['success']

                # Condition completion
                if condition_name not in condition_success:
                    condition_success[condition_name] = 0
                    condition_failures[condition_name] = 0

                if success:
                    condition_success[condition_name] += 1
                else:
                    condition_failures[condition_name] += 1

                # Stage completion tracking
                for stage_name in condition_data['stages'].keys():
                    if stage_name not in stage_completion:
                        stage_completion[stage_name] = 0
                    stage_completion[stage_name] += 1

        return {
            'total_participants': total_participants,
            'completed_participants': completed_participants,
            'completion_rate': (completed_participants / total_participants * 100),
            'condition_success': condition_success,
            'condition_failures': condition_failures,
            'stage_completion': stage_completion
        }

    def get_quality_statistics(self, participants: Dict) -> Dict:
        """Extract comprehensive quality statistics"""
        all_bad_channels = []
        all_rejection_rates = []
        all_ica_components = []
        bad_channel_details = []
        interpolation_success = []

        for participant_id, participant_data in participants.items():
            for condition_name, condition_data in participant_data['conditions'].items():
                stages = condition_data['stages']

                # Detailed bad channel analysis
                if 'detect_bad_channels' in stages:
                    bad_ch_metrics = stages['detect_bad_channels']['metrics']
                    n_detected = bad_ch_metrics['n_detected']
                    n_final = bad_ch_metrics['n_final']
                    detected_bads = bad_ch_metrics.get('detected_bads', [])
                    interpolation_successful = bad_ch_metrics.get('interpolation_successful', False)

                    all_bad_channels.append(n_detected)
                    bad_channel_details.extend(detected_bads)
                    interpolation_success.append(interpolation_successful)

                # Epoch rejection analysis
                if 'epoch' in stages:
                    epoch_metrics = stages['epoch']['metrics']
                    rejection_rate = epoch_metrics['rejection_rate']
                    all_rejection_rates.append(rejection_rate)

                # ICA component analysis
                if 'blink_artifact' in stages:
                    ica_metrics = stages['blink_artifact']['metrics']
                    if 'n_components_excluded' in ica_metrics:
                        n_components = ica_metrics['n_components_excluded']
                        all_ica_components.append(n_components)

        # Count bad channel frequencies
        bad_channel_frequency = {}
        for channel in bad_channel_details:
            bad_channel_frequency[channel] = bad_channel_frequency.get(channel, 0) + 1

        return {
            'bad_channels': self._compute_stats(all_bad_channels),
            'rejection_rates': self._compute_stats(all_rejection_rates),
            'ica_components': self._compute_stats(all_ica_components),
            'bad_channel_frequency': bad_channel_frequency,
            'interpolation_success_rate': (
                    sum(interpolation_success) / len(interpolation_success) * 100) if interpolation_success else 0,
            'quality_flags': self._compute_quality_flags(all_bad_channels, all_rejection_rates, all_ica_components)
        }

    def get_processing_statistics(self, participants: Dict) -> Dict:
        """Extract processing time and parameter statistics"""
        processing_times = []
        parameter_summary = {}

        for participant_data in participants.values():
            start_time = datetime.fromisoformat(participant_data['start_time'])
            end_time = datetime.fromisoformat(participant_data.get('end_time', participant_data['start_time']))
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            processing_times.append(duration)

            # Collect processing parameters
            for condition_data in participant_data['conditions'].values():
                for stage_name, stage_data in condition_data['stages'].items():
                    metrics = stage_data['metrics']
                    if 'parameters' in metrics:
                        if stage_name not in parameter_summary:
                            parameter_summary[stage_name] = {}
                        for param, value in metrics['parameters'].items():
                            if param not in parameter_summary[stage_name]:
                                parameter_summary[stage_name][param] = []
                            parameter_summary[stage_name][param].append(value)

        return {
            'processing_times': processing_times,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'parameter_summary': parameter_summary
        }

    def compute_participant_summary_stats(self, conditions: Dict) -> Dict:
        """Compute comprehensive summary statistics for a participant"""
        total_conditions = len(conditions)
        successful_conditions = sum(1 for c in conditions.values() if c['completion']['success'])
        success_rate = (successful_conditions / total_conditions * 100) if total_conditions > 0 else 0

        total_bad_channels = 0
        total_rejection_rate = 0
        total_ica_components = 0
        total_processing_time = 0
        total_stages = 0

        critical_flags = 0
        warning_flags = 0

        for condition_data in conditions.values():
            stages = condition_data['stages']

            # Bad channels
            if 'detect_bad_channels' in stages:
                n_detected = stages['detect_bad_channels']['metrics']['n_detected']
                total_bad_channels += n_detected
                if n_detected >= self.thresholds['bad_channels']['critical']:
                    critical_flags += 1
                elif n_detected >= self.thresholds['bad_channels']['warning']:
                    warning_flags += 1

            # Rejection rate
            if 'epoch' in stages:
                reject_rate = stages['epoch']['metrics']['rejection_rate']
                total_rejection_rate += reject_rate
                if reject_rate >= self.thresholds['rejection_rate']['critical']:
                    critical_flags += 1
                elif reject_rate >= self.thresholds['rejection_rate']['warning']:
                    warning_flags += 1

            # ICA components
            if 'blink_artifact' in stages:
                n_ica = stages['blink_artifact']['metrics'].get('n_components_excluded', 0)
                if isinstance(n_ica, int):
                    total_ica_components += n_ica
                    if n_ica >= self.thresholds['ica_components']['critical']:
                        critical_flags += 1
                    elif n_ica >= self.thresholds['ica_components']['warning']:
                        warning_flags += 1

            # Processing time
            start_time = datetime.fromisoformat(condition_data['start_time'])
            end_time = datetime.fromisoformat(
                condition_data['completion'].get('end_time', condition_data['start_time']))
            duration = (end_time - start_time).total_seconds() / 60
            total_processing_time += duration

            # Stages
            total_stages += len(stages)

        # Overall quality assessment
        if critical_flags > 0:
            overall_quality = "CRITICAL"
        elif warning_flags > 0:
            overall_quality = "WARNING"
        else:
            overall_quality = "GOOD"

        return {
            'total_conditions': total_conditions,
            'successful_conditions': successful_conditions,
            'success_rate': success_rate,
            'total_bad_channels': total_bad_channels,
            'avg_bad_channels': total_bad_channels / total_conditions if total_conditions > 0 else 0,
            'avg_rejection_rate': total_rejection_rate / total_conditions if total_conditions > 0 else 0,
            'total_ica_components': total_ica_components,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / total_conditions if total_conditions > 0 else 0,
            'total_stages': total_stages,
            'critical_flags': critical_flags,
            'warning_flags': warning_flags,
            'overall_quality': overall_quality
        }

    def _compute_quality_flags(self, bad_channels, rejection_rates, ica_components) -> Dict:
        """Compute quality flags based on thresholds"""
        flags = {
            'critical_participants': 0,
            'warning_participants': 0,
            'good_participants': 0
        }

        total_participants = len(bad_channels)

        for i in range(total_participants):
            bad_ch = bad_channels[i] if i < len(bad_channels) else 0
            reject_rate = rejection_rates[i] if i < len(rejection_rates) else 0
            ica_comp = ica_components[i] if i < len(ica_components) else 0

            is_critical = (bad_ch >= self.thresholds['bad_channels']['critical'] or
                           reject_rate >= self.thresholds['rejection_rate']['critical'] or
                           ica_comp >= self.thresholds['ica_components']['critical'])

            is_warning = (bad_ch >= self.thresholds['bad_channels']['warning'] or
                          reject_rate >= self.thresholds['rejection_rate']['warning'] or
                          ica_comp >= self.thresholds['ica_components']['warning'])

            if is_critical:
                flags['critical_participants'] += 1
            elif is_warning:
                flags['warning_participants'] += 1
            else:
                flags['good_participants'] += 1

        return flags

    def _compute_stats(self, values: List[float]) -> Dict:
        """Compute comprehensive distribution statistics"""
        if not values:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'median': 0, 'distribution': []}

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'max': max(values),
            'min': min(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'distribution': values
        }

    def get_participant_condition_data(self, participant_id: str) -> Dict:
        """Get processed data for a specific participant for individual reports"""
        if participant_id not in self.data['participants']:
            raise ValueError(f"Participant {participant_id} not found in data")

        participant_data = self.data['participants'][participant_id]
        conditions = participant_data['conditions']

        # Compute participant-specific statistics
        summary_stats = self.compute_participant_summary_stats(conditions)

        return {
            'participant_data': participant_data,
            'summary_stats': summary_stats,
            'thresholds': self.thresholds  # For plotting threshold lines
        }