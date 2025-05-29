# utils/quality_reporter.py

from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for batch processing
import seaborn as sns
from loguru import logger
import base64
from io import BytesIO


class QualityReporter:
    """Generate HTML quality control reports from tracked metrics"""

    def __init__(self, metrics_file: Path):
        self.metrics_file = Path(metrics_file)
        self.quality_dir = self.metrics_file.parent
        self.data = self._load_metrics()

        # Flagging thresholds
        self.thresholds = {
            'bad_channels': {'warning': 3, 'critical': 5},
            'rejection_rate': {'warning': 15, 'critical': 30},
            'ica_components': {'warning': 5, 'critical': 10}
        }

    def _load_metrics(self) -> Dict:
        """Load quality metrics from JSON file"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics file: {e}")
            return {}

    def generate_all_reports(self):
        """Generate both summary and individual participant reports"""
        logger.info("Generating quality control reports...")

        # Generate summary report
        summary_path = self._generate_summary_report()

        # Generate individual participant reports
        participant_paths = self._generate_participant_reports()

        logger.success(f"Reports generated:")
        logger.success(f"  Summary: {summary_path}")
        logger.success(f"  Individual reports: {len(participant_paths)} files")

        return summary_path, participant_paths

    def _generate_summary_report(self) -> Path:
        """Generate dataset overview HTML report"""
        output_path = self.quality_dir / "quality_summary_report.html"

        # Extract summary statistics
        participants = self.data.get('participants', {})
        dataset_info = self.data.get('dataset_info', {})

        # Generate summary statistics
        completion_stats = self._get_completion_statistics(participants)
        quality_stats = self._get_quality_statistics(participants)

        # Generate plots
        plots = self._generate_summary_plots(participants)

        # Create HTML content
        html_content = self._create_summary_html(
            dataset_info, completion_stats, quality_stats, plots
        )

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_participant_reports(self) -> List[Path]:
        """Generate individual HTML reports for each participant"""
        participants = self.data.get('participants', {})
        report_paths = []

        # Create individual reports directory
        individual_dir = self.quality_dir / "individual_reports"
        individual_dir.mkdir(exist_ok=True)

        for participant_id, participant_data in participants.items():
            output_path = individual_dir / f"{participant_id}_quality_report.html"

            # Generate participant-specific plots
            plots = self._generate_participant_plots(participant_id, participant_data)

            # Create HTML content
            html_content = self._create_participant_html(
                participant_id, participant_data, plots
            )

            # Write HTML file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            report_paths.append(output_path)

        return report_paths

    def _get_completion_statistics(self, participants: Dict) -> Dict:
        """Extract completion statistics across all participants"""
        total_participants = len(participants)
        completed_participants = sum(1 for p in participants.values() if p.get('completed', False))

        # Count condition completions
        condition_success = {}
        condition_failures = {}

        for participant_data in participants.values():
            conditions = participant_data.get('conditions', {})
            for condition_name, condition_data in conditions.items():
                success = condition_data.get('completion', {}).get('success', False)

                if condition_name not in condition_success:
                    condition_success[condition_name] = 0
                    condition_failures[condition_name] = 0

                if success:
                    condition_success[condition_name] += 1
                else:
                    condition_failures[condition_name] += 1

        return {
            'total_participants': total_participants,
            'completed_participants': completed_participants,
            'completion_rate': (completed_participants / total_participants * 100) if total_participants > 0 else 0,
            'condition_success': condition_success,
            'condition_failures': condition_failures
        }

    def _get_quality_statistics(self, participants: Dict) -> Dict:
        """Extract quality statistics across all participants"""
        all_bad_channels = []
        all_rejection_rates = []
        all_ica_components = []

        for participant_id, participant_data in participants.items():
            conditions = participant_data.get('conditions', {})

            for condition_name, condition_data in conditions.items():
                stages = condition_data.get('stages', {})

                # Bad channels
                if 'detect_bad_channels' in stages:
                    n_bad = stages['detect_bad_channels']['metrics'].get('n_bad_channels', 0)
                    all_bad_channels.append(n_bad)

                # Epoch rejection
                if 'epoch' in stages:
                    rejection_rate = stages['epoch']['metrics'].get('rejection_rate', 0)
                    all_rejection_rates.append(rejection_rate)

                # ICA components
                if 'blink_artifact' in stages:
                    n_components = stages['blink_artifact']['metrics'].get('components_removed', 0)
                    all_ica_components.append(n_components)

        return {
            'bad_channels': {
                'mean': np.mean(all_bad_channels) if all_bad_channels else 0,
                'std': np.std(all_bad_channels) if all_bad_channels else 0,
                'max': max(all_bad_channels) if all_bad_channels else 0,
                'distribution': all_bad_channels
            },
            'rejection_rates': {
                'mean': np.mean(all_rejection_rates) if all_rejection_rates else 0,
                'std': np.std(all_rejection_rates) if all_rejection_rates else 0,
                'max': max(all_rejection_rates) if all_rejection_rates else 0,
                'distribution': all_rejection_rates
            },
            'ica_components': {
                'mean': np.mean(all_ica_components) if all_ica_components else 0,
                'std': np.std(all_ica_components) if all_ica_components else 0,
                'max': max(all_ica_components) if all_ica_components else 0,
                'distribution': all_ica_components
            }
        }

    def _generate_summary_plots(self, participants: Dict) -> Dict[str, str]:
        """Generate summary plots for the overview report"""
        plots = {}

        # 1. Completion Matrix Plot
        plots['completion_matrix'] = self._plot_completion_matrix(participants)

        # 2. Quality Metrics Distribution
        plots['quality_distributions'] = self._plot_quality_distributions(participants)

        # 3. Bad Channels Frequency Map
        plots['bad_channels_frequency'] = self._plot_bad_channels_frequency(participants)

        return plots

    def _plot_completion_matrix(self, participants: Dict) -> str:
        """Create completion matrix heatmap"""
        # Extract completion data
        participant_ids = list(participants.keys())
        conditions = set()

        for participant_data in participants.values():
            conditions.update(participant_data.get('conditions', {}).keys())

        conditions = sorted(list(conditions))

        # Create completion matrix
        completion_matrix = np.zeros((len(participant_ids), len(conditions)))

        for i, participant_id in enumerate(participant_ids):
            participant_data = participants[participant_id]
            participant_conditions = participant_data.get('conditions', {})

            for j, condition_name in enumerate(conditions):
                if condition_name in participant_conditions:
                    success = participant_conditions[condition_name].get('completion', {}).get('success', False)
                    completion_matrix[i, j] = 1 if success else -1
                else:
                    completion_matrix[i, j] = 0  # Not processed

        # Create plot
        fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 0.8), max(6, len(participant_ids) * 0.3)))

        # Custom colormap: Green=Success, Red=Failed, Gray=Not processed
        colors = ['red', 'lightgray', 'green']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(completion_matrix, cmap=cmap, norm=norm, aspect='auto')

        # Set labels
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_yticks(range(len(participant_ids)))
        ax.set_yticklabels(participant_ids)

        ax.set_title('Processing Completion Matrix\n(Green=Success, Red=Failed, Gray=Not Processed)')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_quality_distributions(self, participants: Dict) -> str:
        """Create quality metrics distribution plots"""
        quality_stats = self._get_quality_statistics(participants)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Bad channels distribution
        if quality_stats['bad_channels']['distribution']:
            axes[0].hist(quality_stats['bad_channels']['distribution'], bins=10, alpha=0.7, color='skyblue')
            axes[0].axvline(self.thresholds['bad_channels']['warning'], color='orange', linestyle='--', label='Warning')
            axes[0].axvline(self.thresholds['bad_channels']['critical'], color='red', linestyle='--', label='Critical')
            axes[0].set_title('Bad Channels Distribution')
            axes[0].set_xlabel('Number of Bad Channels')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()

        # Rejection rate distribution
        if quality_stats['rejection_rates']['distribution']:
            axes[1].hist(quality_stats['rejection_rates']['distribution'], bins=10, alpha=0.7, color='lightgreen')
            axes[1].axvline(self.thresholds['rejection_rate']['warning'], color='orange', linestyle='--',
                            label='Warning')
            axes[1].axvline(self.thresholds['rejection_rate']['critical'], color='red', linestyle='--',
                            label='Critical')
            axes[1].set_title('Epoch Rejection Rate Distribution')
            axes[1].set_xlabel('Rejection Rate (%)')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()

        # ICA components distribution
        if quality_stats['ica_components']['distribution']:
            axes[2].hist(quality_stats['ica_components']['distribution'], bins=10, alpha=0.7, color='salmon')
            axes[2].set_title('ICA Components Removed Distribution')
            axes[2].set_xlabel('Number of Components')
            axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_bad_channels_frequency(self, participants: Dict) -> str:
        """Create bad channels frequency map"""
        bad_channel_counts = {}

        for participant_data in participants.values():
            conditions = participant_data.get('conditions', {})
            for condition_data in conditions.values():
                stages = condition_data.get('stages', {})
                if 'detect_bad_channels' in stages:
                    bad_channels = stages['detect_bad_channels']['metrics'].get('bad_channels', [])
                    for channel in bad_channels:
                        bad_channel_counts[channel] = bad_channel_counts.get(channel, 0) + 1

        if not bad_channel_counts:
            # No bad channels found
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, 'No bad channels detected across dataset',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Bad Channels Frequency')
            return self._plot_to_base64(fig)

        # Sort by frequency
        channels = list(bad_channel_counts.keys())
        counts = [bad_channel_counts[ch] for ch in channels]

        # Create bar plot
        fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.5), 6))
        bars = ax.bar(channels, counts, alpha=0.7, color='coral')

        # Color bars based on frequency
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count >= 3:  # Frequently bad channel
                bar.set_color('red')
            elif count >= 2:
                bar.set_color('orange')

        ax.set_title('Bad Channels Frequency Across Dataset')
        ax.set_xlabel('Channel Name')
        ax.set_ylabel('Number of Occurrences')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._plot_to_base64(fig)

    def _generate_participant_plots(self, participant_id: str, participant_data: Dict) -> Dict[str, str]:
        """Generate plots for individual participant report"""
        plots = {}

        # 1. Condition Summary Bar Chart
        plots['condition_summary'] = self._plot_participant_condition_summary(participant_data)

        # 2. Quality Metrics Table (as plot)
        plots['quality_metrics'] = self._plot_participant_quality_metrics(participant_data)

        return plots

    def _plot_participant_condition_summary(self, participant_data: Dict) -> str:
        """Create condition summary plot for participant"""
        conditions = participant_data.get('conditions', {})

        condition_names = []
        bad_channels = []
        rejection_rates = []
        success_status = []

        for condition_name, condition_data in conditions.items():
            condition_names.append(condition_name)
            stages = condition_data.get('stages', {})

            # Extract metrics
            n_bad = 0
            if 'detect_bad_channels' in stages:
                n_bad = stages['detect_bad_channels']['metrics'].get('n_bad_channels', 0)
            bad_channels.append(n_bad)

            rejection_rate = 0
            if 'epoch' in stages:
                rejection_rate = stages['epoch']['metrics'].get('rejection_rate', 0)
            rejection_rates.append(rejection_rate)

            success = condition_data.get('completion', {}).get('success', False)
            success_status.append(success)

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        x = range(len(condition_names))

        # Bad channels plot
        colors1 = ['red' if n >= self.thresholds['bad_channels']['critical']
                   else 'orange' if n >= self.thresholds['bad_channels']['warning']
        else 'green' for n in bad_channels]
        ax1.bar(x, bad_channels, color=colors1, alpha=0.7)
        ax1.set_title('Bad Channels per Condition')
        ax1.set_ylabel('Number of Bad Channels')
        ax1.set_xticks(x)
        ax1.set_xticklabels(condition_names, rotation=45)

        # Rejection rate plot
        colors2 = ['red' if r >= self.thresholds['rejection_rate']['critical']
                   else 'orange' if r >= self.thresholds['rejection_rate']['warning']
        else 'green' for r in rejection_rates]
        ax2.bar(x, rejection_rates, color=colors2, alpha=0.7)
        ax2.set_title('Epoch Rejection Rate per Condition')
        ax2.set_ylabel('Rejection Rate (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(condition_names, rotation=45)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_participant_quality_metrics(self, participant_data: Dict) -> str:
        """Create quality metrics summary table as plot"""
        conditions = participant_data.get('conditions', {})

        # Prepare data for table
        table_data = []
        headers = ['Condition', 'Status', 'Bad Channels', 'Rejection Rate (%)', 'ICA Components']

        for condition_name, condition_data in conditions.items():
            stages = condition_data.get('stages', {})
            completion = condition_data.get('completion', {})

            # Extract data
            status = "✓ Success" if completion.get('success', False) else "✗ Failed"

            n_bad = stages.get('detect_bad_channels', {}).get('metrics', {}).get('n_bad_channels', 'N/A')
            rejection_rate = stages.get('epoch', {}).get('metrics', {}).get('rejection_rate', 'N/A')
            ica_components = stages.get('blink_artifact', {}).get('metrics', {}).get('components_removed', 'N/A')

            table_data.append([condition_name, status, n_bad, rejection_rate, ica_components])

        # Create table plot
        fig, ax = plt.subplots(figsize=(12, max(4, len(table_data) * 0.5)))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color cells based on values
        for i, row in enumerate(table_data):
            # Status column (index 1)
            if "Success" in row[1]:
                table[(i + 1, 1)].set_facecolor('lightgreen')
            else:
                table[(i + 1, 1)].set_facecolor('lightcoral')

            # Bad channels column (index 2)
            if isinstance(row[2], (int, float)):
                if row[2] >= self.thresholds['bad_channels']['critical']:
                    table[(i + 1, 2)].set_facecolor('lightcoral')
                elif row[2] >= self.thresholds['bad_channels']['warning']:
                    table[(i + 1, 2)].set_facecolor('lightyellow')

            # Rejection rate column (index 3)
            if isinstance(row[3], (int, float)):
                if row[3] >= self.thresholds['rejection_rate']['critical']:
                    table[(i + 1, 3)].set_facecolor('lightcoral')
                elif row[3] >= self.thresholds['rejection_rate']['warning']:
                    table[(i + 1, 3)].set_facecolor('lightyellow')

        plt.title(f'Quality Metrics Summary', pad=20)
        return self._plot_to_base64(fig)

    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _get_flag_class(self, value: float, thresholds: Dict[str, float]) -> str:
        """Get CSS class for flagging based on thresholds"""
        if value >= thresholds['critical']:
            return 'critical'
        elif value >= thresholds['warning']:
            return 'warning'
        else:
            return 'good'

    def _create_summary_html(self, dataset_info: Dict, completion_stats: Dict,
                             quality_stats: Dict, plots: Dict[str, str]) -> str:
        """Create HTML content for summary report"""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEG Processing Quality Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .good {{ color: #27ae60; }}
        .plot-container {{ margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>EEG Processing Quality Summary Report</h1>

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
        </div>

        <h2>Processing Completion Matrix</h2>
        <div class="plot-container">
            <img src="{plots['completion_matrix']}" alt="Completion Matrix">
        </div>

        <h2>Quality Metrics Distributions</h2>
        <div class="plot-container">
            <img src="{plots['quality_distributions']}" alt="Quality Distributions">
        </div>

        <h2>Bad Channels Frequency</h2>
        <div class="plot-container">
            <img src="{plots['bad_channels_frequency']}" alt="Bad Channels Frequency">
        </div>

        <h2>Condition Success Rates</h2>
        <table>
            <tr>
                <th>Condition</th>
                <th>Successful</th>
                <th>Failed</th>
                <th>Success Rate</th>
            </tr>
        """

        for condition_name in completion_stats['condition_success'].keys():
            success = completion_stats['condition_success'][condition_name]
            failed = completion_stats['condition_failures'][condition_name]
            total = success + failed
            success_rate = (success / total * 100) if total > 0 else 0

            html += f"""
            <tr>
                <td>{condition_name}</td>
                <td>{success}</td>
                <td>{failed}</td>
                <td>{success_rate:.1f}%</td>
            </tr>
            """

        html += f"""
        </table>

        <h2>Quality Statistics Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Maximum</th>
            </tr>
            <tr>
                <td>Bad Channels</td>
                <td>{quality_stats['bad_channels']['mean']:.1f}</td>
                <td>{quality_stats['bad_channels']['std']:.1f}</td>
                <td>{quality_stats['bad_channels']['max']}</td>
            </tr>
            <tr>
                <td>Rejection Rate (%)</td>
                <td>{quality_stats['rejection_rates']['mean']:.1f}</td>
                <td>{quality_stats['rejection_rates']['std']:.1f}</td>
                <td>{quality_stats['rejection_rates']['max']:.1f}</td>
            </tr>
            <tr>
                <td>ICA Components</td>
                <td>{quality_stats['ica_components']['mean']:.1f}</td>
                <td>{quality_stats['ica_components']['std']:.1f}</td>
                <td>{quality_stats['ica_components']['max']}</td>
            </tr>
        </table>

        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Processing started: {dataset_info.get('start_time', 'N/A')}</p>
            <p>Processing completed: {dataset_info.get('end_time', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
        """

        return html

    def _create_participant_html(self, participant_id: str, participant_data: Dict,
                                 plots: Dict[str, str]) -> str:
        """Create HTML content for individual participant report"""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report: {participant_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .good {{ color: #27ae60; }}
        .plot-container {{ margin: 20px 0; text-align: center; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        .back-link {{ margin-bottom: 20px; }}
        .back-link a {{ color: #3498db; text-decoration: none; }}
        .back-link a:hover {{ text-decoration: underline; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="../quality_summary_report.html">← Back to Summary Report</a>
        </div>

        <h1>Quality Report: {participant_id}</h1>

        <h2>Condition Summary</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{plots['condition_summary'].split(',')[1]}" alt="Condition Summary">
        </div>

        <h2>Quality Metrics</h2>
        <div class="plot-container">
            <img src="data:image/png;base64,{plots['quality_metrics'].split(',')[1]}" alt="Quality Metrics">
        </div>

        <div class="footer">
            <p>Individual report for participant: {participant_id}</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """

        return html


# Helper function to generate reports after pipeline completion
def generate_quality_reports(results_dir: Path):
    """Generate quality reports from saved metrics"""
    quality_dir = results_dir / "quality"
    metrics_file = quality_dir / "quality_metrics.json"

    if not metrics_file.exists():
        logger.error(f"No quality metrics file found: {metrics_file}")
        return None

    reporter = QualityReporter(metrics_file)
    return reporter.generate_all_reports()