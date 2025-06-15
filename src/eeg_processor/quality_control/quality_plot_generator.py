# quality_control/quality_plot_generator.py

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import base64
from io import BytesIO
from loguru import logger

matplotlib.use('Agg')


class QualityPlotGenerator:
    """
    Generates all plots for quality control reports.

    Handles visualization of quality metrics including completion matrices,
    distributions, bad channels analysis, and processing timelines.
    """

    def __init__(self, thresholds: Dict = None):
        """
        Initialize plot generator with quality thresholds for reference lines.

        Args:
            thresholds: Dictionary of quality thresholds for warning/critical lines
        """
        self.thresholds = thresholds or {
            'bad_channels': {'warning': 4, 'critical': 8},
            'rejection_rate': {'warning': 15, 'critical': 30},
            'ica_components': {'warning': 16, 'critical': 24}
        }

    def generate_summary_plots(self, stats: Dict) -> Dict[str, str]:
        """
        Generate all summary plots for the main quality report.

        Args:
            stats: Complete statistics dictionary from QualityMetricsAnalyzer

        Returns:
            Dictionary mapping plot names to base64 encoded images
        """
        participants = stats['participants']

        return {
            'dashboard_summary': self.plot_dashboard_summary(stats),
            'completion_matrix': self.plot_completion_matrix(participants),
            'quality_distributions': self.plot_quality_distributions(stats['quality_stats']),
            'participant_bad_channels': self.plot_participant_bad_channels(participants),
            'processing_overview': self.plot_processing_overview(stats),
        }

    def generate_participant_plots(self, participant_id: str, participant_stats: Dict) -> Dict[str, str]:
        """
        Generate plots for individual participant reports.

        Args:
            participant_id: ID of the participant
            participant_stats: Participant data from QualityMetricsAnalyzer

        Returns:
            Dictionary mapping plot names to base64 encoded images
        """
        participant_data = participant_stats['participant_data']

        return {
            'condition_overview': self.plot_participant_condition_overview(participant_data),
            'quality_details': self.plot_participant_quality_details(participant_data),
            'processing_timeline': self.plot_participant_processing_timeline(participant_data),
            'detailed_metrics': self.plot_participant_detailed_metrics(participant_data,
                                                                       participant_stats['summary_stats'])
        }

    def plot_dashboard_summary(self, stats: Dict) -> str:
        """Create compact dashboard showing most critical global information"""
        quality_stats = stats['quality_stats']
        completion_stats = stats['completion_stats']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle('Quality Control Dashboard - Key Metrics at a Glance', fontsize=16, fontweight='bold')

        # 1. Threshold Violations (Top Left)
        violations = self._compute_threshold_violations(quality_stats)
        violation_names = list(violations.keys())
        violation_counts = list(violations.values())

        colors = ['red' if count > 0 else 'lightgreen' for count in violation_counts]
        bars = ax1.bar(violation_names, violation_counts, color=colors, alpha=0.8)

        # Add value labels
        for bar, count in zip(bars, violation_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     str(count), ha='center', va='bottom', fontweight='bold')

        ax1.set_title('Critical Threshold Violations', fontweight='bold')
        ax1.set_ylabel('Number of Participants')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Dataset Quality Assessment (Top Right)
        self._plot_quality_pie_chart(ax2, quality_stats, completion_stats)

        # 3. Key Quality Metrics Summary (Bottom Left)
        self._plot_metrics_summary_text(ax3, quality_stats)

        # 4. Processing Efficiency Summary (Bottom Right)
        self._plot_processing_summary_text(ax4, stats)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_completion_matrix(self, participants: Dict) -> str:
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
                text = '✓' if value == 1 else '✗' if value == -1 else '○'
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

    def plot_quality_distributions(self, quality_stats: Dict) -> str:
        """Enhanced quality metrics distribution plots"""
        fig, axes = plt.subplots(3, 2, figsize=(18, 10))
        fig.suptitle('Quality Metrics Distributions', fontsize=16, fontweight='bold')

        # Bad channels histogram with statistics
        self._plot_metric_histogram(axes[0, 0], quality_stats['bad_channels'], 'bad_channels',
                                    'Bad Channels Distribution', 'Number of Bad Channels')

        # Bad channels box plot
        self._plot_metric_boxplot(axes[0, 1], quality_stats['bad_channels'], 'Bad Channels Box Plot')

        # Rejection rates histogram
        self._plot_metric_histogram(axes[1, 0], quality_stats['rejection_rates'], 'rejection_rate',
                                    'Epoch Rejection Rate Distribution', 'Rejection Rate (%)')

        # Rejection rates box plot
        self._plot_metric_boxplot(axes[1, 1], quality_stats['rejection_rates'], 'Rejection Rates Box Plot')

        # ICA components histogram
        self._plot_metric_histogram(axes[2, 0], quality_stats['ica_components'], 'ica_components',
                                    'ICA Components Removed', 'Number of Components')

        # Quality flags pie chart
        self._plot_quality_flags_pie(axes[2, 1], quality_stats['quality_flags'])

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_participant_bad_channels(self, participants: Dict) -> str:
        """Participant-level bad channels bar chart grouped by condition"""
        # Collect data: participant -> condition -> bad_channel_count
        participant_data = {}
        all_conditions = set()

        for participant_id, participant_info in participants.items():
            participant_data[participant_id] = {}

            for condition_name, condition_data in participant_info['conditions'].items():
                all_conditions.add(condition_name)

                # Extract bad channel count
                if 'detect_bad_channels' in condition_data['stages']:
                    n_detected = condition_data['stages']['detect_bad_channels']['metrics']['n_detected']
                else:
                    n_detected = 0

                participant_data[participant_id][condition_name] = n_detected

        all_conditions = sorted(list(all_conditions))
        participant_ids = sorted(list(participant_data.keys()))

        if not participant_ids:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No participant data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Bad Channels by Participant')
            return self._plot_to_base64(fig)

        # Create the plot
        fig, ax = plt.subplots(figsize=(max(12, len(participant_ids) * 0.8), 6))

        # Set up bar positions
        n_conditions = len(all_conditions)
        bar_width = 0.8 / n_conditions if n_conditions > 0 else 0.8
        x_positions = np.arange(len(participant_ids))

        # Color palette for conditions
        colors = plt.cm.Set2(np.linspace(0, 1, n_conditions))

        # Plot bars for each condition
        for i, condition in enumerate(all_conditions):
            condition_counts = [participant_data[pid].get(condition, 0) for pid in participant_ids]
            bar_positions = x_positions + (i - n_conditions / 2 + 0.5) * bar_width

            bars = ax.bar(bar_positions, condition_counts, bar_width,
                          label=condition, color=colors[i], alpha=0.8)

            # Add value labels on bars
            for bar, count in zip(bars, condition_counts):
                if count > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize the plot
        ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Bad Channels Detected', fontsize=12, fontweight='bold')
        ax.set_title('Bad Channels Detected by Participant and Condition', fontsize=14, fontweight='bold')

        # Set x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels(participant_ids, rotation=45, ha='right')

        # Add threshold lines
        ax.axhline(y=self.thresholds['bad_channels']['warning'],
                   color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.axhline(y=self.thresholds['bad_channels']['critical'],
                   color='red', linestyle='--', alpha=0.7, label='Critical Threshold')

        # Legend and grid
        if n_conditions > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_processing_overview(self, stats: Dict) -> str:
        """Processing time and parameter overview"""
        processing_stats = stats['processing_stats']
        completion_stats = stats['completion_stats']

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
        self._plot_condition_success_rates(ax3, completion_stats)

        # Dataset overview stats
        self._plot_dataset_overview_text(ax4, stats)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_participant_condition_overview(self, participant_data: Dict) -> str:
        """Enhanced condition overview with more details"""
        conditions = participant_data['conditions']
        condition_names = list(conditions.keys())

        # Extract metrics for each condition
        bad_channels, rejection_rates, ica_components = self._extract_participant_condition_metrics(conditions)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        x = range(len(condition_names))

        # Bad channels with threshold coloring
        self._plot_metric_bars_with_thresholds(axes[0], x, bad_channels, condition_names,
                                               'bad_channels', 'Detected Bad Channels per Condition',
                                               'Number of Bad Channels')

        # Rejection rates with threshold coloring
        self._plot_metric_bars_with_thresholds(axes[1], x, rejection_rates, condition_names,
                                               'rejection_rate', 'Epoch Rejection Rate per Condition',
                                               'Rejection Rate (%)', is_percentage=True)

        # ICA components with threshold coloring
        self._plot_metric_bars_with_thresholds(axes[2], x, ica_components, condition_names,
                                               'ica_components', 'ICA Components Excluded per Condition',
                                               'Number of Components')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_participant_quality_details(self, participant_data: Dict) -> str:
        """Detailed quality analysis for participant"""
        conditions = participant_data['conditions']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Bad channels breakdown
        self._plot_bad_channels_breakdown(ax1, conditions)

        # Stage completion matrix
        self._plot_participant_stage_matrix(ax2, conditions)

        # Quality metrics vs thresholds
        self._plot_participant_thresholds_comparison(ax3, conditions)

        # Processing efficiency metrics
        self._plot_participant_efficiency_text(ax4, participant_data, conditions)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_participant_processing_timeline(self, participant_data: Dict) -> str:
        """Processing timeline visualization"""
        conditions = participant_data['conditions']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Timeline of condition processing (Gantt chart style)
        self._plot_processing_gantt(ax1, participant_data, conditions)

        # Stage processing breakdown
        self._plot_stage_execution_frequency(ax2, conditions)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def plot_participant_detailed_metrics(self, participant_data: Dict, summary_stats: Dict) -> str:
        """Detailed metrics table and summary"""
        conditions = participant_data['conditions']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # Detailed metrics table
        self._plot_participant_metrics_table(ax1, conditions)

        # Summary statistics text
        self._plot_participant_summary_text(ax2, summary_stats)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    # Helper methods for plot components
    def _compute_threshold_violations(self, quality_stats: Dict) -> Dict:
        """Count violations of critical thresholds"""
        violations = {'Bad Channels': 0, 'Rejection Rate': 0, 'ICA Components': 0}

        for metric_name, threshold_key in [('Bad Channels', 'bad_channels'),
                                           ('Rejection Rate', 'rejection_rates'),
                                           ('ICA Components', 'ica_components')]:
            if quality_stats[threshold_key]['distribution']:
                critical_threshold = self.thresholds[threshold_key.replace('rejection_rates', 'rejection_rate')][
                    'critical']
                violations[metric_name] = sum(1 for x in quality_stats[threshold_key]['distribution']
                                              if x >= critical_threshold)
        return violations

    def _plot_quality_pie_chart(self, ax, quality_stats: Dict, completion_stats: Dict):
        """Plot quality assessment pie chart"""
        total_participants = completion_stats['total_participants']
        quality_flags = quality_stats['quality_flags']

        critical_participants = quality_flags['critical_participants']
        warning_participants = quality_flags['warning_participants']
        good_participants = quality_flags['good_participants']

        # Overall quality assessment
        if critical_participants > total_participants * 0.3:
            overall_status = "POOR"
            status_color = 'red'
        elif critical_participants > total_participants * 0.1 or warning_participants > total_participants * 0.5:
            overall_status = "CONCERNING"
            status_color = 'orange'
        else:
            overall_status = "GOOD"
            status_color = 'green'

        # Pie chart of quality distribution
        sizes = [good_participants, warning_participants, critical_participants]
        labels = ['Good Quality', 'Warning Level', 'Critical Issues']
        colors_pie = ['lightgreen', 'orange', 'lightcoral']

        # Only include non-zero categories
        filtered_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors_pie) if size > 0]
        if filtered_data:
            sizes_filtered, labels_filtered, colors_filtered = zip(*filtered_data)
            ax.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered,
                   autopct='%1.0f', startangle=90)

        ax.set_title(f'Dataset Quality Status: {overall_status}',
                     fontweight='bold', color=status_color, fontsize=14)

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _plot_metrics_summary_text(self, ax, quality_stats: Dict):
        """Plot key quality metrics as text summary"""
        metrics_text = [
            "KEY QUALITY METRICS",
            "=" * 25,
            f"Bad Channels (mean ± std): {quality_stats['bad_channels']['mean']:.1f} ± {quality_stats['bad_channels']['std']:.1f}",
            f"Rejection Rate (mean ± std): {quality_stats['rejection_rates']['mean']:.1f}% ± {quality_stats['rejection_rates']['std']:.1f}%",
            f"ICA Components (mean ± std): {quality_stats['ica_components']['mean']:.1f} ± {quality_stats['ica_components']['std']:.1f}",
            "",
            "THRESHOLDS",
            f"Bad Channels: Warning ≥{self.thresholds['bad_channels']['warning']}, Critical ≥{self.thresholds['bad_channels']['critical']}",
            f"Rejection Rate: Warning ≥{self.thresholds['rejection_rate']['warning']}%, Critical ≥{self.thresholds['rejection_rate']['critical']}%",
            f"ICA Components: Warning ≥{self.thresholds['ica_components']['warning']}, Critical ≥{self.thresholds['ica_components']['critical']}",
            "",
            f"Interpolation Success Rate: {quality_stats['interpolation_success_rate']:.1f}%"
        ]

        ax.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Quality Metrics Summary', fontweight='bold')
        ax.axis('off')

    def _plot_processing_summary_text(self, ax, stats: Dict):
        """Plot processing efficiency summary"""
        completion_stats = stats['completion_stats']
        processing_stats = stats['processing_stats']
        quality_stats = stats['quality_stats']

        # Determine overall status
        critical_participants = quality_stats['quality_flags']['critical_participants']
        warning_participants = quality_stats['quality_flags']['warning_participants']
        total_participants = completion_stats['total_participants']

        if critical_participants > total_participants * 0.3:
            overall_status = "POOR"
        elif critical_participants > total_participants * 0.1 or warning_participants > total_participants * 0.5:
            overall_status = "CONCERNING"
        else:
            overall_status = "GOOD"

        processing_text = [
            "PROCESSING SUMMARY",
            "=" * 20,
            f"Total Participants: {completion_stats['total_participants']}",
            f"Successfully Completed: {completion_stats['completed_participants']}",
            f"Overall Success Rate: {completion_stats['completion_rate']:.1f}%",
            "",
            "CONDITION SUCCESS RATES:",
        ]

        # Add condition success rates
        for condition in completion_stats['condition_success'].keys():
            success = completion_stats['condition_success'][condition]
            failure = completion_stats['condition_failures'][condition]
            total = success + failure
            rate = (success / total * 100) if total > 0 else 0
            processing_text.append(f"  {condition}: {rate:.1f}% ({success}/{total})")

        processing_text.extend([
            "",
            f"Average Processing Time: {processing_stats['avg_processing_time']:.1f} min",
            "",
            f"Quality Assessment: {overall_status}",
            f"Ready for Analysis: {'✓ YES' if overall_status in ['GOOD', 'CONCERNING'] else '✗ REVIEW NEEDED'}"
        ])

        # Color code the text box based on overall status
        box_color = {'GOOD': 'lightgreen', 'CONCERNING': 'lightyellow', 'POOR': 'lightcoral'}[overall_status]

        ax.text(0.05, 0.95, '\n'.join(processing_text), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
        ax.set_title('Processing & Analysis Readiness', fontweight='bold')
        ax.axis('off')

    def _plot_metric_histogram(self, ax, metric_stats: Dict, threshold_key: str, title: str, xlabel: str):
        """Plot histogram for a quality metric with threshold lines"""
        if metric_stats['distribution']:
            ax.hist(metric_stats['distribution'], bins=10, alpha=0.7,
                    color='skyblue', edgecolor='black')

            # Add threshold lines if available
            if threshold_key in self.thresholds:
                ax.axvline(self.thresholds[threshold_key]['warning'], color='orange',
                           linestyle='--', linewidth=2, label='Warning')
                ax.axvline(self.thresholds[threshold_key]['critical'], color='red',
                           linestyle='--', linewidth=2, label='Critical')

            ax.axvline(metric_stats['mean'], color='green',
                       linestyle='-', linewidth=2, label='Mean')
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

    def _plot_metric_boxplot(self, ax, metric_stats: Dict, title: str):
        """Plot boxplot for a quality metric"""
        if metric_stats['distribution']:
            ax.boxplot(metric_stats['distribution'])
        ax.set_title(title)
        ax.set_ylabel('Values')

    def _plot_quality_flags_pie(self, ax, quality_flags: Dict):
        """Plot quality flags as pie chart"""
        labels = ['Good', 'Warning', 'Critical']
        sizes = [quality_flags['good_participants'],
                 quality_flags['warning_participants'],
                 quality_flags['critical_participants']]
        colors = ['lightgreen', 'orange', 'lightcoral']

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Participant Quality Flags')

    def _plot_condition_success_rates(self, ax, completion_stats: Dict):
        """Plot condition success rates"""
        condition_names = list(completion_stats['condition_success'].keys())
        success_rates = []

        for condition in condition_names:
            total = completion_stats['condition_success'][condition] + completion_stats['condition_failures'][condition]
            success_rate = (completion_stats['condition_success'][condition] / total * 100) if total > 0 else 0
            success_rates.append(success_rate)

        bars = ax.bar(condition_names, success_rates, alpha=0.7, color='lightcoral')
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        ax.set_title('Condition Success Rates')
        ax.set_ylabel('Success Rate (%)')
        ax.tick_params(axis='x', rotation=45)

    def _plot_dataset_overview_text(self, ax, stats: Dict):
        """Plot dataset overview statistics"""
        completion_stats = stats['completion_stats']
        processing_stats = stats['processing_stats']

        overview_data = [
            f"Total Participants: {completion_stats['total_participants']}",
            f"Completed: {completion_stats['completed_participants']}",
            f"Success Rate: {completion_stats['completion_rate']:.1f}%",
            f"Avg Processing Time: {processing_stats['avg_processing_time']:.1f} min",
            f"Total Conditions: {len(completion_stats['condition_success'])}",
            f"Total Stages: {len(completion_stats['stage_completion'])}"
        ]

        ax.text(0.1, 0.9, '\n'.join(overview_data), transform=ax.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Dataset Overview')
        ax.axis('off')

    def _extract_participant_condition_metrics(self, conditions: Dict) -> tuple:
        """Extract metrics for each condition of a participant"""
        condition_names = list(conditions.keys())
        bad_channels = []
        rejection_rates = []
        ica_components = []

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

        return bad_channels, rejection_rates, ica_components

    def _plot_metric_bars_with_thresholds(self, ax, x, values, labels, threshold_key: str,
                                          title: str, ylabel: str, is_percentage: bool = False):
        """Plot bars with threshold-based coloring"""
        # Color bars based on thresholds
        colors = []
        for val in values:
            if threshold_key in self.thresholds:
                if val >= self.thresholds[threshold_key]['critical']:
                    colors.append('red')
                elif val >= self.thresholds[threshold_key]['warning']:
                    colors.append('orange')
                else:
                    colors.append('green')
            else:
                colors.append('blue')

        bars = ax.bar(x, values, color=colors, alpha=0.7)

        # Add value labels on bars
        for i, v in enumerate(values):
            label_text = f'{v:.1f}%' if is_percentage else str(v)
            ax.text(i, v + (max(values) * 0.02), label_text, ha='center', va='bottom', fontweight='bold')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    def _plot_bad_channels_breakdown(self, ax, conditions: Dict):
        """Plot bad channels breakdown (original/detected/final)"""
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
                for key in bad_channel_breakdown:
                    bad_channel_breakdown[key].append(0)

        x = np.arange(len(condition_names))
        width = 0.25

        ax.bar(x - width, bad_channel_breakdown['original'], width, label='Original', alpha=0.7, color='lightgray')
        ax.bar(x, bad_channel_breakdown['detected'], width, label='Detected', alpha=0.7, color='orange')
        ax.bar(x + width, bad_channel_breakdown['final'], width, label='Final', alpha=0.7, color='red')

        ax.set_title('Bad Channels Breakdown')
        ax.set_ylabel('Number of Channels')
        ax.set_xticks(x)
        ax.set_xticklabels(condition_names, rotation=45)
        ax.legend()

    def _plot_participant_stage_matrix(self, ax, conditions: Dict):
        """Plot stage completion matrix for participant"""
        condition_names = list(conditions.keys())
        all_stages = set()
        for condition_data in conditions.values():
            all_stages.update(condition_data['stages'].keys())
        all_stages = sorted(list(all_stages))

        stage_matrix = np.zeros((len(condition_names), len(all_stages)))

        for i, condition_name in enumerate(condition_names):
            condition_stages = conditions[condition_name]['stages']
            for j, stage_name in enumerate(all_stages):
                stage_matrix[i, j] = 1 if stage_name in condition_stages else 0

        im = ax.imshow(stage_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_title('Stage Completion Matrix')
        ax.set_xticks(range(len(all_stages)))
        ax.set_xticklabels(all_stages, rotation=45)
        ax.set_yticks(range(len(condition_names)))
        ax.set_yticklabels(condition_names)

    def _plot_participant_thresholds_comparison(self, ax, conditions: Dict):
        """Plot participant metrics vs thresholds"""
        metrics_names = ['Bad Channels', 'Rejection Rate', 'ICA Components']
        participant_values = []

        # Get first condition's metrics (or average if multiple)
        for condition_data in conditions.values():
            stages = condition_data['stages']
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
        ax.bar(x_pos - 0.3, participant_values, 0.2, label='Participant', alpha=0.7, color='blue')
        ax.bar(x_pos - 0.1, threshold_warnings, 0.2, label='Warning Threshold', alpha=0.7, color='orange')
        ax.bar(x_pos + 0.1, threshold_criticals, 0.2, label='Critical Threshold', alpha=0.7, color='red')

        ax.set_title('Participant vs Thresholds')
        ax.set_ylabel('Values')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names, rotation=45)
        ax.legend()

    def _plot_participant_efficiency_text(self, ax, participant_data: Dict, conditions: Dict):
        """Plot processing efficiency metrics as text"""
        total_start = datetime.fromisoformat(participant_data['start_time'])
        total_end = datetime.fromisoformat(participant_data.get('end_time', participant_data['start_time']))
        total_duration = (total_end - total_start).total_seconds() / 60

        # Calculate total bad channels detected
        total_bad_channels = 0
        for condition_data in conditions.values():
            if 'detect_bad_channels' in condition_data['stages']:
                total_bad_channels += condition_data['stages']['detect_bad_channels']['metrics']['n_detected']

        efficiency_metrics = [
            f"Total Processing Time: {total_duration:.1f} minutes",
            f"Conditions Processed: {len(conditions)}",
            f"Average Time per Condition: {total_duration / len(conditions):.1f} min",
            f"Success Rate: {sum(1 for c in conditions.values() if c['completion']['success']) / len(conditions) * 100:.1f}%",
            f"Total Bad Channels Detected: {total_bad_channels}",
            f"Interpolation Success: {'Yes' if total_bad_channels > 0 else 'N/A'}"
        ]

        ax.text(0.05, 0.95, '\n'.join(efficiency_metrics), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Processing Efficiency Summary')
        ax.axis('off')

    def _plot_processing_gantt(self, ax, participant_data: Dict, conditions: Dict):
        """Plot processing timeline as Gantt chart"""
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
            ax.barh(i, duration, left=start, height=0.6, color=color, alpha=0.7)
            ax.text(start + duration / 2, i, f'{duration:.1f}m',
                    ha='center', va='center', fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(condition_names)
        ax.set_xlabel('Time from Start (minutes)')
        ax.set_title('Condition Processing Timeline')
        ax.grid(True, alpha=0.3)

    def _plot_stage_execution_frequency(self, ax, conditions: Dict):
        """Plot how many times each stage was executed"""
        all_stages = set()
        for condition_data in conditions.values():
            all_stages.update(condition_data['stages'].keys())
        all_stages = sorted(list(all_stages))

        stage_counts = {stage: 0 for stage in all_stages}
        for condition_data in conditions.values():
            for stage in condition_data['stages'].keys():
                stage_counts[stage] += 1

        ax.bar(range(len(all_stages)), list(stage_counts.values()), alpha=0.7, color='lightblue')
        ax.set_xticks(range(len(all_stages)))
        ax.set_xticklabels(all_stages, rotation=45)
        ax.set_ylabel('Number of Executions')
        ax.set_title('Stage Execution Frequency')

    def _plot_participant_metrics_table(self, ax, conditions: Dict):
        """Plot detailed metrics as a table"""
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

        # Create table
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
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

            # Color based on thresholds if applicable
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

        ax.set_title('Detailed Metrics Table', pad=20)

    def _plot_participant_summary_text(self, ax, summary_stats: Dict):
        """Plot participant summary statistics as text"""
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

        ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Participant Summary Statistics')
        ax.axis('off')