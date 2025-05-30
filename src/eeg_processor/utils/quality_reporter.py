# utils/quality_reporter.py - ENHANCED VERSION

from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
from loguru import logger
import base64
from io import BytesIO


class QualityReporter:
    """Enhanced quality reporter with comprehensive visualizations"""

    def __init__(self, metrics_file: Path):
        self.metrics_file = Path(metrics_file)
        self.quality_dir = self.metrics_file.parent

        with open(self.metrics_file, 'r') as f:
            self.data = json.load(f)

        # Quality thresholds
        self.thresholds = {
            'bad_channels': {'warning': 4, 'critical': 8},
            'rejection_rate': {'warning': 15, 'critical': 30},
            'ica_components': {'warning': 16, 'critical': 24}
        }

    def generate_all_reports(self):
        """Generate both summary and individual participant reports"""
        logger.info("Generating enhanced quality reports...")

        summary_path = self._generate_summary_report()
        participant_paths = self._generate_participant_reports()

        logger.success(f"Enhanced reports generated: summary + {len(participant_paths)} individual reports")
        return summary_path, participant_paths

    def _generate_summary_report(self) -> Path:
        """Generate comprehensive dataset overview HTML report"""
        output_path = self.quality_dir / "quality_summary_report.html"

        participants = self.data['participants']
        dataset_info = self.data['dataset_info']

        completion_stats = self._get_completion_statistics(participants)
        quality_stats = self._get_quality_statistics(participants)
        processing_stats = self._get_processing_statistics(participants)
        plots = self._generate_comprehensive_summary_plots(participants)

        html_content = self._create_enhanced_summary_html(
            dataset_info, completion_stats, quality_stats, processing_stats, plots
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _get_completion_statistics(self, participants: Dict) -> Dict:
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

    def _get_quality_statistics(self, participants: Dict) -> Dict:
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

    def _get_processing_statistics(self, participants: Dict) -> Dict:
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

    def _generate_comprehensive_summary_plots(self, participants: Dict) -> Dict[str, str]:
        """Generate comprehensive summary plots"""
        return {
            'completion_matrix': self._plot_completion_matrix(participants),
            'quality_distributions': self._plot_enhanced_quality_distributions(participants),
            'bad_channels_analysis': self._plot_bad_channels_analysis(participants),
            'processing_overview': self._plot_processing_overview(participants),
            'quality_summary_dashboard': self._plot_quality_summary_dashboard(participants)
        }

    def _plot_enhanced_quality_distributions(self, participants: Dict) -> str:
        """Enhanced quality metrics distribution plots with better visualization"""
        quality_stats = self._get_quality_statistics(participants)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Quality Metrics Distributions', fontsize=16, fontweight='bold')

        # Bad channels histogram with statistics
        if quality_stats['bad_channels']['distribution']:
            axes[0, 0].hist(quality_stats['bad_channels']['distribution'], bins=10, alpha=0.7,
                            color='skyblue', edgecolor='black')
            axes[0, 0].axvline(self.thresholds['bad_channels']['warning'], color='orange',
                               linestyle='--', linewidth=2, label='Warning')
            axes[0, 0].axvline(self.thresholds['bad_channels']['critical'], color='red',
                               linestyle='--', linewidth=2, label='Critical')
            axes[0, 0].axvline(quality_stats['bad_channels']['mean'], color='green',
                               linestyle='-', linewidth=2, label='Mean')
            axes[0, 0].legend()
        axes[0, 0].set_title('Bad Channels Distribution')
        axes[0, 0].set_xlabel('Number of Bad Channels')
        axes[0, 0].set_ylabel('Frequency')

        # Bad channels box plot
        if quality_stats['bad_channels']['distribution']:
            axes[1, 0].boxplot(quality_stats['bad_channels']['distribution'])
            axes[1, 0].set_title('Bad Channels Box Plot')
            axes[1, 0].set_ylabel('Number of Bad Channels')

        # Rejection rates histogram
        if quality_stats['rejection_rates']['distribution']:
            axes[0, 1].hist(quality_stats['rejection_rates']['distribution'], bins=10, alpha=0.7,
                            color='lightgreen', edgecolor='black')
            axes[0, 1].axvline(self.thresholds['rejection_rate']['warning'], color='orange',
                               linestyle='--', linewidth=2, label='Warning')
            axes[0, 1].axvline(self.thresholds['rejection_rate']['critical'], color='red',
                               linestyle='--', linewidth=2, label='Critical')
            axes[0, 1].axvline(quality_stats['rejection_rates']['mean'], color='green',
                               linestyle='-', linewidth=2, label='Mean')
            axes[0, 1].legend()
        axes[0, 1].set_title('Epoch Rejection Rate Distribution')
        axes[0, 1].set_xlabel('Rejection Rate (%)')
        axes[0, 1].set_ylabel('Frequency')

        # Rejection rates box plot
        if quality_stats['rejection_rates']['distribution']:
            axes[1, 1].boxplot(quality_stats['rejection_rates']['distribution'])
            axes[1, 1].set_title('Rejection Rates Box Plot')
            axes[1, 1].set_ylabel('Rejection Rate (%)')

        # ICA components histogram
        if quality_stats['ica_components']['distribution']:
            axes[0, 2].hist(quality_stats['ica_components']['distribution'], bins=10, alpha=0.7,
                            color='salmon', edgecolor='black')
            axes[0, 2].axvline(quality_stats['ica_components']['mean'], color='green',
                               linestyle='-', linewidth=2, label='Mean')
            axes[0, 2].legend()
        axes[0, 2].set_title('ICA Components Removed')
        axes[0, 2].set_xlabel('Number of Components')
        axes[0, 2].set_ylabel('Frequency')

        # Quality flags pie chart
        quality_flags = quality_stats['quality_flags']
        labels = ['Good', 'Warning', 'Critical']
        sizes = [quality_flags['good_participants'],
                 quality_flags['warning_participants'],
                 quality_flags['critical_participants']]
        colors = ['lightgreen', 'orange', 'lightcoral']

        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Participant Quality Flags')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_bad_channels_analysis(self, participants: Dict) -> str:
        """Comprehensive bad channels analysis"""
        quality_stats = self._get_quality_statistics(participants)
        bad_channel_freq = quality_stats['bad_channel_frequency']

        if not bad_channel_freq:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No bad channels detected across dataset',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Bad Channels Analysis')
            return self._plot_to_base64(fig)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Bad Channels Analysis', fontsize=16, fontweight='bold')

        # Frequency bar chart
        channels = list(bad_channel_freq.keys())
        counts = [bad_channel_freq[ch] for ch in channels]

        bars = ax1.bar(channels, counts, alpha=0.7, color='coral')
        for bar, count in zip(bars, counts):
            if count >= 3:
                bar.set_color('red')
            elif count >= 2:
                bar.set_color('orange')
            # Add value labels on bars
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     str(count), ha='center', va='bottom')

        ax1.set_title('Bad Channel Frequency')
        ax1.set_xlabel('Channel Name')
        ax1.set_ylabel('Number of Occurrences')
        ax1.tick_params(axis='x', rotation=45)

        # Channel location heatmap (if we can infer locations)
        # For now, show frequency distribution
        freq_counts = {}
        for count in counts:
            freq_counts[count] = freq_counts.get(count, 0) + 1

        ax2.bar(freq_counts.keys(), freq_counts.values(), alpha=0.7, color='lightblue')
        ax2.set_title('Frequency Distribution')
        ax2.set_xlabel('Times Channel was Bad')
        ax2.set_ylabel('Number of Channels')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_processing_overview(self, participants: Dict) -> str:
        """Processing time and parameter overview"""
        processing_stats = self._get_processing_statistics(participants)
        completion_stats = self._get_completion_statistics(participants)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Processing Overview', fontsize=16, fontweight='bold')

        # Processing times
        if processing_stats['processing_times']:
            ax1.hist(processing_stats['processing_times'], bins=10, alpha=0.7,
                     color='lightblue', edgecolor='black')
            ax1.axvline(processing_stats['avg_processing_time'], color='red',
                        linestyle='--', linewidth=2, label='Average')
            ax1.legend()
        ax1.set_title('Processing Time Distribution')
        ax1.set_xlabel('Processing Time (minutes)')
        ax1.set_ylabel('Frequency')

        # Stage completion rates
        stage_names = list(completion_stats['stage_completion'].keys())
        stage_counts = list(completion_stats['stage_completion'].values())

        ax2.barh(stage_names, stage_counts, alpha=0.7, color='lightgreen')
        ax2.set_title('Stage Completion Counts')
        ax2.set_xlabel('Number of Completions')

        # Condition success rates
        condition_names = list(completion_stats['condition_success'].keys())
        success_rates = []
        for condition in condition_names:
            total = completion_stats['condition_success'][condition] + completion_stats['condition_failures'][condition]
            success_rate = (completion_stats['condition_success'][condition] / total * 100) if total > 0 else 0
            success_rates.append(success_rate)

        bars = ax3.bar(condition_names, success_rates, alpha=0.7, color='lightcoral')
        for bar, rate in zip(bars, success_rates):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom')
        ax3.set_title('Condition Success Rates')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)

        # Dataset overview stats
        total_participants = completion_stats['total_participants']
        completed = completion_stats['completed_participants']

        overview_data = [
            f"Total Participants: {total_participants}",
            f"Completed: {completed}",
            f"Success Rate: {(completed / total_participants * 100):.1f}%",
            f"Avg Processing Time: {processing_stats['avg_processing_time']:.1f} min",
            f"Total Conditions: {len(condition_names)}",
            f"Total Stages: {len(stage_names)}"
        ]

        ax4.text(0.1, 0.9, '\n'.join(overview_data), transform=ax4.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Dataset Overview')
        ax4.axis('off')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_quality_summary_dashboard(self, participants: Dict) -> str:
        """Quality summary dashboard with key metrics"""
        quality_stats = self._get_quality_statistics(participants)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Quality Summary Dashboard', fontsize=16, fontweight='bold')

        # Quality metrics summary
        metrics_summary = [
            f"Bad Channels (mean ± std): {quality_stats['bad_channels']['mean']:.1f} ± {quality_stats['bad_channels']['std']:.1f}",
            f"Bad Channels (median): {quality_stats['bad_channels']['median']:.1f}",
            f"Rejection Rate (mean ± std): {quality_stats['rejection_rates']['mean']:.1f}% ± {quality_stats['rejection_rates']['std']:.1f}%",
            f"Rejection Rate (median): {quality_stats['rejection_rates']['median']:.1f}%",
            f"ICA Components (mean ± std): {quality_stats['ica_components']['mean']:.1f} ± {quality_stats['ica_components']['std']:.1f}",
            f"Interpolation Success Rate: {quality_stats['interpolation_success_rate']:.1f}%"
        ]

        ax1.text(0.05, 0.95, '\n'.join(metrics_summary), transform=ax1.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.set_title('Quality Metrics Summary')
        ax1.axis('off')

        # Threshold violations
        bad_ch_violations = sum(1 for x in quality_stats['bad_channels']['distribution']
                                if x >= self.thresholds['bad_channels']['warning'])
        reject_violations = sum(1 for x in quality_stats['rejection_rates']['distribution']
                                if x >= self.thresholds['rejection_rate']['warning'])

        violations = ['Bad Channels ≥ Warning', 'Rejection Rate ≥ Warning']
        violation_counts = [bad_ch_violations, reject_violations]

        ax2.bar(violations, violation_counts, alpha=0.7, color=['orange', 'red'])
        ax2.set_title('Threshold Violations')
        ax2.set_ylabel('Number of Cases')
        ax2.tick_params(axis='x', rotation=45)

        # Quality distribution comparison
        if (quality_stats['bad_channels']['distribution'] and
                quality_stats['rejection_rates']['distribution']):
            ax3.scatter(quality_stats['bad_channels']['distribution'],
                        quality_stats['rejection_rates']['distribution'],
                        alpha=0.6, s=50)
            ax3.set_xlabel('Number of Bad Channels')
            ax3.set_ylabel('Rejection Rate (%)')
            ax3.set_title('Bad Channels vs Rejection Rate')
            ax3.grid(True, alpha=0.3)

        # Processing quality trends (if multiple participants)
        participant_ids = list(participants.keys())
        if len(participant_ids) > 1:
            bad_ch_by_participant = []
            for pid in participant_ids:
                participant_data = participants[pid]
                for condition_data in participant_data['conditions'].values():
                    if 'detect_bad_channels' in condition_data['stages']:
                        n_bad = condition_data['stages']['detect_bad_channels']['metrics']['n_detected']
                        bad_ch_by_participant.append(n_bad)
                        break

            ax4.plot(range(len(bad_ch_by_participant)), bad_ch_by_participant,
                     marker='o', alpha=0.7)
            ax4.set_xlabel('Participant Index')
            ax4.set_ylabel('Number of Bad Channels')
            ax4.set_title('Quality Trend Across Participants')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Single Participant\nNo Trend Analysis',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Quality Trends')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_completion_matrix(self, participants: Dict) -> str:
        """Enhanced completion matrix with better visualization"""
        participant_ids = list(participants.keys())
        conditions = set()
        for participant_data in participants.values():
            conditions.update(participant_data['conditions'].keys())
        conditions = sorted(list(conditions))

        completion_matrix = np.zeros((len(participant_ids), len(conditions)))

        for i, participant_id in enumerate(participant_ids):
            participant_conditions = participants[participant_id]['conditions']
            for j, condition_name in enumerate(conditions):
                if condition_name in participant_conditions:
                    success = participant_conditions[condition_name]['completion']['success']
                    completion_matrix[i, j] = 1 if success else -1
                else:
                    completion_matrix[i, j] = 0

        fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.2), max(6, len(participant_ids) * 0.4)))

        colors = ['red', 'lightgray', 'green']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(completion_matrix, cmap=cmap, norm=norm, aspect='auto')

        # Add text annotations
        for i in range(len(participant_ids)):
            for j in range(len(conditions)):
                value = completion_matrix[i, j]
                if value == 1:
                    text = '✓'
                elif value == -1:
                    text = '✗'
                else:
                    text = '○'
                ax.text(j, i, text, ha='center', va='center', fontsize=12, fontweight='bold')

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_yticks(range(len(participant_ids)))
        ax.set_yticklabels(participant_ids)
        ax.set_title('Processing Completion Matrix\n(✓=Success, ✗=Failed, ○=Not Processed)',
                     fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Failed', 'Not Processed', 'Success'])

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _generate_participant_reports(self) -> List[Path]:
        """Generate enhanced individual participant reports"""
        participants = self.data['participants']
        report_paths = []

        individual_dir = self.quality_dir / "individual_reports"
        individual_dir.mkdir(exist_ok=True)

        for participant_id, participant_data in participants.items():
            output_path = individual_dir / f"{participant_id}_quality_report.html"
            plots = self._generate_enhanced_participant_plots(participant_id, participant_data)
            html_content = self._create_enhanced_participant_html(participant_id, participant_data, plots)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            report_paths.append(output_path)

        return report_paths

    def _generate_enhanced_participant_plots(self, participant_id: str, participant_data: Dict) -> Dict[str, str]:
        """Generate comprehensive plots for individual participant"""
        return {
            'condition_overview': self._plot_participant_condition_overview(participant_data),
            'quality_details': self._plot_participant_quality_details(participant_data),
            'processing_timeline': self._plot_participant_processing_timeline(participant_data),
            'detailed_metrics': self._plot_participant_detailed_metrics(participant_data)
        }

    def _plot_participant_condition_overview(self, participant_data: Dict) -> str:
        """Enhanced condition overview with more details"""
        conditions = participant_data['conditions']

        condition_names = list(conditions.keys())
        bad_channels = []
        rejection_rates = []
        ica_components = []
        processing_times = []
        success_status = []

        for condition_name, condition_data in conditions.items():
            stages = condition_data['stages']

            # Extract metrics
            bad_channels.append(
                stages['detect_bad_channels']['metrics']['n_detected']
                if 'detect_bad_channels' in stages else 0
            )

            rejection_rates.append(
                stages['epoch']['metrics']['rejection_rate']
                if 'epoch' in stages else 0
            )

            ica_components.append(
                stages['blink_artifact']['metrics'].get('n_components_excluded', 0)
                if 'blink_artifact' in stages else 0
            )

            # Calculate processing time for this condition
            start_time = datetime.fromisoformat(condition_data['start_time'])
            end_time = datetime.fromisoformat(
                condition_data['completion'].get('end_time', condition_data['start_time']))
            duration = (end_time - start_time).total_seconds() / 60
            processing_times.append(duration)

            success_status.append(condition_data['completion']['success'])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        x = range(len(condition_names))

        # Bad channels with values on bars
        colors1 = ['red' if n >= self.thresholds['bad_channels']['critical']
                   else 'orange' if n >= self.thresholds['bad_channels']['warning']
                   else 'green' for n in bad_channels]
        bars1 = axes[0].bar(x, bad_channels, color=colors1, alpha=0.7)
        for i, v in enumerate(bad_channels):
            axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        axes[0].set_title('Detected Bad Channels per Condition')
        axes[0].set_ylabel('Number of Bad Channels')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(condition_names)

        # Rejection rates with values
        colors2 = ['red' if r >= self.thresholds['rejection_rate']['critical']
                   else 'orange' if r >= self.thresholds['rejection_rate']['warning']
        else 'green' for r in rejection_rates]
        if any(rejection_rates):
            bars2 = axes[1].bar(x, rejection_rates, color=colors2, alpha=0.7)
            for i, v in enumerate(rejection_rates):
                axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        axes[1].set_title('Epoch Rejection Rate per Condition')
        axes[1].set_ylabel('Rejection Rate (%)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(condition_names)

        # ICA components with values
        colors3 = ['red' if n >= self.thresholds['ica_components']['critical']
                   else 'orange' if n >= self.thresholds['ica_components']['warning']
        else 'green' for n in ica_components]
        bars3 = axes[2].bar(x, ica_components, color=colors3, alpha=0.7)
        for i, v in enumerate(ica_components):
            axes[2].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        axes[2].set_title('ICA Components Excluded per Condition')
        axes[2].set_ylabel('Number of Components')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(condition_names)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_participant_quality_details(self, participant_data: Dict) -> str:
        """Detailed quality analysis for participant"""
        conditions = participant_data['conditions']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Bad channels breakdown for each condition
        condition_names = []
        bad_channel_breakdown = {'original': [], 'detected': [], 'final': []}

        for condition_name, condition_data in conditions.items():
            condition_names.append(condition_name)
            stages = condition_data['stages']

            if 'detect_bad_channels' in stages:
                metrics = stages['detect_bad_channels']['metrics']
                bad_channel_breakdown['original'].append(metrics.get('n_original', 0))
                bad_channel_breakdown['detected'].append(metrics.get('n_detected', 0))
                bad_channel_breakdown['final'].append(metrics.get('n_final', 0))
            else:
                bad_channel_breakdown['original'].append(0)
                bad_channel_breakdown['detected'].append(0)
                bad_channel_breakdown['final'].append(0)

        x = np.arange(len(condition_names))
        width = 0.25

        ax1.bar(x - width, bad_channel_breakdown['original'], width, label='Original', alpha=0.7, color='lightgray')
        ax1.bar(x, bad_channel_breakdown['detected'], width, label='Detected', alpha=0.7, color='orange')
        ax1.bar(x + width, bad_channel_breakdown['final'], width, label='Final', alpha=0.7, color='red')

        ax1.set_title('Bad Channels Breakdown')
        ax1.set_ylabel('Number of Channels')
        ax1.set_xticks(x)
        ax1.set_xticklabels(condition_names, rotation=45)
        ax1.legend()

        # Stage completion status
        all_stages = set()
        for condition_data in conditions.values():
            all_stages.update(condition_data['stages'].keys())

        all_stages = sorted(list(all_stages))
        stage_matrix = np.zeros((len(condition_names), len(all_stages)))

        for i, condition_name in enumerate(condition_names):
            condition_stages = conditions[condition_name]['stages']
            for j, stage_name in enumerate(all_stages):
                stage_matrix[i, j] = 1 if stage_name in condition_stages else 0

        im = ax2.imshow(stage_matrix, cmap='RdYlGn', aspect='auto')
        ax2.set_title('Stage Completion Matrix')
        ax2.set_xticks(range(len(all_stages)))
        ax2.set_xticklabels(all_stages, rotation=45)
        ax2.set_yticks(range(len(condition_names)))
        ax2.set_yticklabels(condition_names)

        # Quality metrics comparison to thresholds
        metrics_names = ['Bad Channels', 'Rejection Rate', 'ICA Components']
        participant_values = []
        threshold_warnings = []
        threshold_criticals = []

        for condition_data in conditions.values():
            stages = condition_data['stages']

            # Get first condition's metrics (or average if multiple)
            bad_ch = stages['detect_bad_channels']['metrics']['n_detected'] if 'detect_bad_channels' in stages else 0
            reject_rate = stages['epoch']['metrics']['rejection_rate'] if 'epoch' in stages else 0
            ica_comp = stages['blink_artifact']['metrics'].get('n_components_excluded',
                                                               0) if 'blink_artifact' in stages else 0

            participant_values = [bad_ch, reject_rate, ica_comp]
            break

        threshold_warnings = [
            self.thresholds['bad_channels']['warning'],
            self.thresholds['rejection_rate']['warning'],
            self.thresholds['ica_components']['warning']
        ]
        threshold_criticals = [
            self.thresholds['bad_channels']['critical'],
            self.thresholds['rejection_rate']['critical'],
            self.thresholds['ica_components']['critical']
        ]

        x_pos = np.arange(len(metrics_names))
        ax3.bar(x_pos - 0.3, participant_values, 0.2, label='Participant', alpha=0.7, color='blue')
        ax3.bar(x_pos - 0.1, threshold_warnings, 0.2, label='Warning Threshold', alpha=0.7, color='orange')
        ax3.bar(x_pos + 0.1, threshold_criticals, 0.2, label='Critical Threshold', alpha=0.7, color='red')

        ax3.set_title('Participant vs Thresholds')
        ax3.set_ylabel('Values')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics_names, rotation=45)
        ax3.legend()

        # Processing efficiency metrics
        total_start = datetime.fromisoformat(participant_data['start_time'])
        total_end = datetime.fromisoformat(participant_data.get('end_time', participant_data['start_time']))
        total_duration = (total_end - total_start).total_seconds() / 60

        efficiency_metrics = [
            f"Total Processing Time: {total_duration:.1f} minutes",
            f"Conditions Processed: {len(conditions)}",
            f"Average Time per Condition: {total_duration / len(conditions):.1f} min",
            f"Success Rate: {sum(1 for c in conditions.values() if c['completion']['success']) / len(conditions) * 100:.1f}%",
            f"Total Bad Channels Detected: {sum(bad_channel_breakdown['detected'])}",
            f"Interpolation Success: {'Yes' if all(final < detected for final, detected in zip(bad_channel_breakdown['final'], bad_channel_breakdown['detected']) if detected > 0) else 'Partial/No'}"
        ]

        ax4.text(0.05, 0.95, '\n'.join(efficiency_metrics), transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Processing Efficiency Summary')
        ax4.axis('off')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_participant_processing_timeline(self, participant_data: Dict) -> str:
        """Processing timeline visualization"""
        conditions = participant_data['conditions']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Timeline of condition processing
        condition_names = []
        start_times = []
        durations = []
        success_status = []

        base_time = datetime.fromisoformat(participant_data['start_time'])

        for condition_name, condition_data in conditions.items():
            condition_names.append(condition_name)
            start_time = datetime.fromisoformat(condition_data['start_time'])
            end_time = datetime.fromisoformat(
                condition_data['completion'].get('end_time', condition_data['start_time']))

            start_offset = (start_time - base_time).total_seconds() / 60  # minutes from start
            duration = (end_time - start_time).total_seconds() / 60

            start_times.append(start_offset)
            durations.append(duration)
            success_status.append(condition_data['completion']['success'])

        # Gantt chart style timeline
        colors = ['green' if success else 'red' for success in success_status]
        y_pos = np.arange(len(condition_names))

        for i, (start, duration, color) in enumerate(zip(start_times, durations, colors)):
            ax1.barh(i, duration, left=start, height=0.6, color=color, alpha=0.7)
            ax1.text(start + duration / 2, i, f'{duration:.1f}m',
                     ha='center', va='center', fontweight='bold')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(condition_names)
        ax1.set_xlabel('Time from Start (minutes)')
        ax1.set_title('Condition Processing Timeline')
        ax1.grid(True, alpha=0.3)

        # Stage processing breakdown for each condition
        all_stages = set()
        for condition_data in conditions.values():
            all_stages.update(condition_data['stages'].keys())
        all_stages = sorted(list(all_stages))

        stage_counts = {stage: 0 for stage in all_stages}
        for condition_data in conditions.values():
            for stage in condition_data['stages'].keys():
                stage_counts[stage] += 1

        ax2.bar(range(len(all_stages)), list(stage_counts.values()), alpha=0.7, color='lightblue')
        ax2.set_xticks(range(len(all_stages)))
        ax2.set_xticklabels(all_stages, rotation=45)
        ax2.set_ylabel('Number of Executions')
        ax2.set_title('Stage Execution Frequency')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_participant_detailed_metrics(self, participant_data: Dict) -> str:
        """Detailed metrics table and summary"""
        conditions = participant_data['conditions']

        # Create comprehensive table data
        table_data = []
        headers = ['Condition', 'Status', 'Bad Ch. (O/D/F)', 'Rejection Rate', 'ICA Comp.', 'Processing Time', 'Stages']

        for condition_name, condition_data in conditions.items():
            stages = condition_data['stages']
            completion = condition_data['completion']

            status = "✓ Success" if completion['success'] else "✗ Failed"

            # Bad channels breakdown
            if 'detect_bad_channels' in stages:
                metrics = stages['detect_bad_channels']['metrics']
                bad_ch_str = f"{metrics.get('n_original', 0)}/{metrics.get('n_detected', 0)}/{metrics.get('n_final', 0)}"
            else:
                bad_ch_str = "N/A"

            # Rejection rate
            rejection_rate = stages['epoch']['metrics']['rejection_rate'] if 'epoch' in stages else 'N/A'
            if isinstance(rejection_rate, (int, float)):
                rejection_rate = f"{rejection_rate:.1f}%"

            # ICA components
            if 'blink_artifact' in stages:
                ica_metrics = stages['blink_artifact']['metrics']
                ica_components = ica_metrics.get('n_components_excluded', 'Applied')
            else:
                ica_components = 'N/A'

            # Processing time
            start_time = datetime.fromisoformat(condition_data['start_time'])
            end_time = datetime.fromisoformat(completion.get('end_time', condition_data['start_time']))
            duration = (end_time - start_time).total_seconds() / 60
            proc_time = f"{duration:.1f}m"

            # Stages completed
            stages_completed = len(stages)

            table_data.append(
                [condition_name, status, bad_ch_str, rejection_rate, ica_components, proc_time, stages_completed])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Detailed metrics table
        ax1.axis('tight')
        ax1.axis('off')

        table = ax1.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        # Color cells based on values
        for i, row in enumerate(table_data):
            # Status column
            if "Success" in row[1]:
                table[(i + 1, 1)].set_facecolor('lightgreen')
            else:
                table[(i + 1, 1)].set_facecolor('lightcoral')

            # Color other cells based on thresholds if applicable
            # Bad channels (parse the O/D/F format)
            if "/" in str(row[2]):
                try:
                    detected = int(row[2].split('/')[1])
                    if detected >= self.thresholds['bad_channels']['critical']:
                        table[(i + 1, 2)].set_facecolor('lightcoral')
                    elif detected >= self.thresholds['bad_channels']['warning']:
                        table[(i + 1, 2)].set_facecolor('lightyellow')
                except:
                    pass

            # Rejection rate
            if isinstance(row[3], str) and '%' in row[3]:
                try:
                    rate = float(row[3].replace('%', ''))
                    if rate >= self.thresholds['rejection_rate']['critical']:
                        table[(i + 1, 3)].set_facecolor('lightcoral')
                    elif rate >= self.thresholds['rejection_rate']['warning']:
                        table[(i + 1, 3)].set_facecolor('lightyellow')
                except:
                    pass

        ax1.set_title('Detailed Metrics Table', pad=20)

        # Summary statistics
        summary_stats = self._compute_participant_summary_stats(conditions)

        summary_text = [
            "PARTICIPANT SUMMARY STATISTICS",
            "=" * 35,
            f"Total Conditions: {summary_stats['total_conditions']}",
            f"Successful Conditions: {summary_stats['successful_conditions']}",
            f"Success Rate: {summary_stats['success_rate']:.1f}%",
            "",
            "QUALITY METRICS:",
            f"  Total Bad Channels Detected: {summary_stats['total_bad_channels']}",
            f"  Average Bad Channels per Condition: {summary_stats['avg_bad_channels']:.1f}",
            f"  Average Rejection Rate: {summary_stats['avg_rejection_rate']:.1f}%",
            f"  Total ICA Components Removed: {summary_stats['total_ica_components']}",
            "",
            "PROCESSING:",
            f"  Total Processing Time: {summary_stats['total_processing_time']:.1f} minutes",
            f"  Average Time per Condition: {summary_stats['avg_processing_time']:.1f} minutes",
            f"  Total Stages Executed: {summary_stats['total_stages']}",
            "",
            "QUALITY FLAGS:",
            f"  Critical Issues: {summary_stats['critical_flags']}",
            f"  Warning Issues: {summary_stats['warning_flags']}",
            f"  Overall Quality: {summary_stats['overall_quality']}"
        ]

        ax2.text(0.05, 0.95, '\n'.join(summary_text), transform=ax2.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax2.set_title('Participant Summary Statistics')
        ax2.axis('off')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _compute_participant_summary_stats(self, conditions: Dict) -> Dict:
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

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _create_enhanced_summary_html(self, dataset_info: Dict, completion_stats: Dict,
                                      quality_stats: Dict, processing_stats: Dict, plots: Dict[str, str]) -> str:
        """Create enhanced HTML content for summary report"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEG Processing Quality Summary Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .header {{ background: rgba(255,255,255,0.95); padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .container {{ max-width: 1200px; margin: 2rem auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }}
        .content {{ padding: 2rem; }}
        h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; font-weight: 700; }}
        h2 {{ color: #34495e; margin: 2rem 0 1rem 0; font-size: 1.8em; border-left: 4px solid #3498db; padding-left: 1rem; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin: 2rem 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 0.5rem; }}
        .stat-label {{ font-size: 1.1em; opacity: 0.9; }}
        .plot-container {{ margin: 2rem 0; text-align: center; background: #f8f9fa; padding: 1rem; border-radius: 12px; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .metrics-summary {{ background: #f8f9fa; padding: 1.5rem; border-radius: 12px; margin: 2rem 0; }}
        .quality-indicator {{ display: inline-block; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; margin: 0.25rem; }}
        .quality-good {{ background: #2ecc71; color: white; }}
        .quality-warning {{ background: #f39c12; color: white; }}
        .quality-critical {{ background: #e74c3c; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
        th {{ background: #3498db; color: white; padding: 1rem; font-weight: 600; }}
        td {{ padding: 0.75rem 1rem; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{ background: #2c3e50; color: white; padding: 2rem; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 EEG Processing Quality Summary Report</h1>
        <p style="font-size: 1.2em; color: #7f8c8d; margin: 1rem 0 0 0;">Comprehensive Analysis Dashboard</p>
    </div>

    <div class="container">
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{completion_stats['total_participants']}</div>
                    <div class="stat-label">Total Participants</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{completion_stats['completed_participants']}</div>
                    <div class="stat-label">Completed Successfully</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{completion_stats['completion_rate']:.1f}%</div>
                    <div class="stat-label">Completion Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{processing_stats['avg_processing_time']:.1f}m</div>
                    <div class="stat-label">Avg Processing Time</div>
                </div>
            </div>

            <div class="metrics-summary">
                <h3>🎯 Quality Indicators</h3>
                <span class="quality-indicator quality-good">Good Quality: {quality_stats['quality_flags']['good_participants']} participants</span>
                <span class="quality-indicator quality-warning">Warning Level: {quality_stats['quality_flags']['warning_participants']} participants</span>
                <span class="quality-indicator quality-critical">Critical Issues: {quality_stats['quality_flags']['critical_participants']} participants</span>
                <p style="margin-top: 1rem;"><strong>Interpolation Success Rate:</strong> {quality_stats['interpolation_success_rate']:.1f}%</p>
            </div>

            <h2>📊 Processing Completion Matrix</h2>
            <div class="plot-container">
                <img src="{plots['completion_matrix']}" alt="Completion Matrix">
            </div>

            <h2>📈 Quality Metrics Distributions</h2>
            <div class="plot-container">
                <img src="{plots['quality_distributions']}" alt="Quality Distributions">
            </div>

            <h2>🔍 Bad Channels Analysis</h2>
            <div class="plot-container">
                <img src="{plots['bad_channels_analysis']}" alt="Bad Channels Analysis">
            </div>

            <h2>⏱️ Processing Overview</h2>
            <div class="plot-container">
                <img src="{plots['processing_overview']}" alt="Processing Overview">
            </div>

            <h2>🎛️ Quality Summary Dashboard</h2>
            <div class="plot-container">
                <img src="{plots['quality_summary_dashboard']}" alt="Quality Dashboard">
            </div>

            <h2>📋 Detailed Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>
                <tr><td><strong>Bad Channels</strong></td><td>{quality_stats['bad_channels']['mean']:.1f}</td><td>{quality_stats['bad_channels']['median']:.1f}</td><td>{quality_stats['bad_channels']['std']:.1f}</td><td>{quality_stats['bad_channels']['min']:.0f}</td><td>{quality_stats['bad_channels']['max']:.0f}</td></tr>
                <tr><td><strong>Rejection Rate (%)</strong></td><td>{quality_stats['rejection_rates']['mean']:.1f}</td><td>{quality_stats['rejection_rates']['median']:.1f}</td><td>{quality_stats['rejection_rates']['std']:.1f}</td><td>{quality_stats['rejection_rates']['min']:.1f}</td><td>{quality_stats['rejection_rates']['max']:.1f}</td></tr>
                <tr><td><strong>ICA Components</strong></td><td>{quality_stats['ica_components']['mean']:.1f}</td><td>{quality_stats['ica_components']['median']:.1f}</td><td>{quality_stats['ica_components']['std']:.1f}</td><td>{quality_stats['ica_components']['min']:.0f}</td><td>{quality_stats['ica_components']['max']:.0f}</td></tr>
            </table>
        </div>
    </div>

    <div class="footer">
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Processing Period:</strong> {dataset_info['start_time']} to {dataset_info['end_time']}</p>
        <p>🔬 Advanced EEG Quality Control System</p>
    </div>
</body>
</html>"""

    def _create_enhanced_participant_html(self, participant_id: str, participant_data: Dict,
                                          plots: Dict[str, str]) -> str:
        """Create enhanced HTML content for individual participant report"""

        # Calculate summary stats for this participant
        summary_stats = self._compute_participant_summary_stats(participant_data['conditions'])

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report: {participant_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .header {{ background: rgba(255,255,255,0.95); padding: 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .container {{ max-width: 1200px; margin: 2rem auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); overflow: hidden; }}
        .content {{ padding: 2rem; }}
        h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; font-weight: 700; }}
        h2 {{ color: #34495e; margin: 2rem 0 1rem 0; font-size: 1.8em; border-left: 4px solid #3498db; padding-left: 1rem; }}
        .back-link {{ margin-bottom: 2rem; }}
        .back-link a {{ color: #3498db; text-decoration: none; font-size: 1.1em; font-weight: 500; }}
        .back-link a:hover {{ text-decoration: underline; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 12px; margin: 2rem 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 1rem; }}
        .summary-item {{ text-align: center; }}
        .summary-number {{ font-size: 2em; font-weight: bold; }}
        .summary-label {{ font-size: 0.9em; opacity: 0.9; }}
        .plot-container {{ margin: 2rem 0; text-align: center; background: #f8f9fa; padding: 1rem; border-radius: 12px; }}
        .plot-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .quality-badge {{ display: inline-block; padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; margin: 0.5rem; }}
        .quality-good {{ background: #2ecc71; color: white; }}
        .quality-warning {{ background: #f39c12; color: white; }}
        .quality-critical {{ background: #e74c3c; color: white; }}
        .footer {{ background: #2c3e50; color: white; padding: 2rem; text-align: center; }}
        .status-indicator {{ font-size: 1.5em; margin: 0.5rem; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="back-link">
            <a href="../quality_summary_report.html">← Back to Summary Report</a>
        </div>
        <h1>🧠 Quality Report: {participant_id}</h1>
        <div class="status-indicator">
            {'✅ Processing Completed' if participant_data['completed'] else '❌ Processing Incomplete'}
            <span class="quality-badge quality-{summary_stats['overall_quality'].lower()}">{summary_stats['overall_quality']} QUALITY</span>
        </div>
    </div>

    <div class="container">
        <div class="content">
            <div class="summary-card">
                <h3 style="margin-top: 0;">📊 Processing Summary</h3>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['total_conditions']}</div>
                        <div class="summary-label">Conditions Processed</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['success_rate']:.1f}%</div>
                        <div class="summary-label">Success Rate</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['total_processing_time']:.1f}m</div>
                        <div class="summary-label">Total Processing Time</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['total_bad_channels']}</div>
                        <div class="summary-label">Bad Channels Detected</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['total_ica_components']}</div>
                        <div class="summary-label">ICA Components Removed</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-number">{summary_stats['total_stages']}</div>
                        <div class="summary-label">Total Stages Executed</div>
                    </div>
                </div>
                <div style="margin-top: 1.5rem;">
                    <strong>Quality Flags:</strong>
                    <span class="quality-badge quality-critical">Critical: {summary_stats['critical_flags']}</span>
                    <span class="quality-badge quality-warning">Warning: {summary_stats['warning_flags']}</span>
                </div>
            </div>

            <h2>📈 Condition Overview</h2>
            <div class="plot-container">
                <img src="{plots['condition_overview']}" alt="Condition Overview">
            </div>

            <h2>🔍 Quality Details</h2>
            <div class="plot-container">
                <img src="{plots['quality_details']}" alt="Quality Details">
            </div>

            <h2>⏱️ Processing Timeline</h2>
            <div class="plot-container">
                <img src="{plots['processing_timeline']}" alt="Processing Timeline">
            </div>

            <h2>📋 Detailed Metrics</h2>
            <div class="plot-container">
                <img src="{plots['detailed_metrics']}" alt="Detailed Metrics">
            </div>
        </div>
    </div>

    <div class="footer">
        <p><strong>Individual Report for:</strong> {participant_id}</p>
        <p><strong>Processing Period:</strong> {participant_data['start_time']} to {participant_data.get('end_time', 'In Progress')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""


def generate_quality_reports(results_dir: Path):
    """Generate enhanced quality reports from saved metrics"""
    quality_dir = results_dir / "quality"
    metrics_file = quality_dir / "quality_metrics.json"

    reporter = QualityReporter(metrics_file)
    return reporter.generate_all_reports()