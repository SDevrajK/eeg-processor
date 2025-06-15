from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import json

from .quality_reporter import QualityReporter


def generate_session_quality_reports(session_dir: Path = None, metrics_file: Path = None) -> Tuple[Path, List[Path]]:
    """
    Generate session quality reports from existing JSON data files
    
    This function works independently of the pipeline and reads quality metrics
    from previously saved JSON files.
    
    Args:
        session_dir: Path to session directory containing quality/ subdirectory
        metrics_file: Direct path to quality_metrics.json file (alternative to session_dir)
        
    Returns:
        Tuple of (summary_report_path, list_of_individual_report_paths)
        
    Raises:
        FileNotFoundError: If quality metrics JSON file cannot be found
        ValueError: If neither session_dir nor metrics_file is provided
    """
    if metrics_file is None:
        if session_dir is None:
            raise ValueError("Either session_dir or metrics_file must be provided")
        
        quality_dir = session_dir / "quality"
        metrics_file = quality_dir / "quality_metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Session quality metrics not found: {metrics_file}")

    logger.info(f"Generating session quality reports from: {metrics_file}")
    
    # Use SessionQualityReporter with JSON-based initialization
    reporter = SessionQualityReporter(metrics_file)
    return reporter.generate_all_reports()


class SessionQualityReporter(QualityReporter):
    """
    Extends existing QualityReporter to handle session data
    Reuses all existing plotting and HTML generation components
    Works independently by reading data from JSON files
    """

    def __init__(self, metrics_file: Path):
        # Initialize basic attributes without calling super().__init__()
        self.metrics_file = metrics_file
        self.quality_dir = metrics_file.parent

        # Load session-specific data from JSON
        with open(metrics_file, 'r') as f:
            self.session_data = json.load(f)

        # Extract core data components
        self.data = self._reconstruct_data_from_json()
        self.session_info = self.data['participants'].get('session_info', {})
        self.availability = self.session_data.get('participant_availability', {})

        # Initialize components that don't depend on pipeline state
        from .quality_metrics_analyzer import QualityMetricsAnalyzer
        from .quality_plot_generator import QualityPlotGenerator  
        from .quality_html_generator import QualityHTMLGenerator

        # Create cleaned data for analyzer (without session_info in participants)
        analyzer_data = {
            'dataset_info': self.data.get('dataset_info', {}),
            'participants': {k: v for k, v in self.data['participants'].items() if k != 'session_info'}
        }
        
        self.analyzer = QualityMetricsAnalyzer(analyzer_data, self.quality_dir)
        self.plotter = QualityPlotGenerator(self.quality_dir)
        self.html_generator = QualityHTMLGenerator()

        logger.info(f"Session quality reporter initialized from JSON: {self.session_info.get('session_name', 'Unknown')}")

    def _reconstruct_data_from_json(self) -> Dict:
        """Reconstruct the data structure expected by QualityReporter from JSON"""
        reconstructed_data = {
            'dataset_info': self.session_data.get('dataset_info', {}),
            'participants': {}
        }

        # Process each participant from the JSON structure
        for participant_id, participant_data in self.session_data.get('participants', {}).items():
            if participant_id == 'session_info':
                # Keep session_info in participants dict as expected by SessionQualityReporter
                reconstructed_data['participants']['session_info'] = participant_data
                continue

            # Convert participant data to expected format
            reconstructed_data['participants'][participant_id] = {
                'conditions': {}
            }

            conditions_data = participant_data.get('conditions', {})
            for condition_name, condition_data in conditions_data.items():
                reconstructed_data['participants'][participant_id]['conditions'][condition_name] = {
                    'completion': self._extract_completion_info(condition_data),
                    'stages': condition_data.get('stages', {}),
                    'config_name': condition_data.get('config_name', 'unknown')
                }

        return reconstructed_data

    def _extract_completion_info(self, condition_data: Dict) -> Dict:
        """Extract completion status from condition data"""
        stages = condition_data.get('stages', {})
        
        # Check if all stages completed successfully
        all_completed = True
        errors = []
        
        for stage_name, stage_data in stages.items():
            stage_metrics = stage_data.get('metrics', {})
            if not stage_metrics.get('stage_completed', False):
                all_completed = False
                error_msg = stage_metrics.get('error', f'Stage {stage_name} failed')
                errors.append(error_msg)

        return {
            'success': all_completed,
            'status': 'completed' if all_completed else 'failed',
            'error': '; '.join(errors) if errors else None,
            'total_stages': len(stages),
            'completed_stages': sum(1 for s in stages.values() if s.get('metrics', {}).get('stage_completed', False))
        }

    def generate_all_reports(self) -> Tuple[Path, List[Path]]:
        """Generate session reports - overrides parent method"""
        logger.info("Generating session quality reports...")

        # Enhance the existing data with session context
        self._enhance_data_with_session_context()

        # Use parent's report generation with enhanced data
        summary_path = self._generate_summary_report_with_session_context()
        participant_paths = self._generate_participant_reports_with_session_context()

        logger.success(f"Session reports generated: summary + {len(participant_paths)} individual reports")
        return summary_path, participant_paths

    def _enhance_data_with_session_context(self):
        """Add session context to existing data structure"""
        # Add session info to dataset_info
        if 'dataset_info' not in self.data:
            self.data['dataset_info'] = {}

        self.data['dataset_info'].update({
            'session_name': self.session_info.get('session_name', 'Unknown'),
            'total_configs': len(self.session_info.get('configs_processed', [])),
            'configs_processed': self.session_info.get('configs_processed', [])
        })

        # Mark missing participants in the main data
        self._mark_missing_participants()

    def _mark_missing_participants(self):
        """Mark conditions as missing vs failed based on availability data"""
        for participant_id, availability_data in self.availability.items():
            if participant_id in self.data['participants']:
                participant_data = self.data['participants'][participant_id]

                # Add availability info to each condition
                for config_name, avail_info in availability_data.items():
                    if avail_info['status'] == 'missing':
                        # Mark any conditions from this config as missing rather than failed
                        for condition_name, condition_data in participant_data.get('conditions', {}).items():
                            if condition_data.get('config_name') == config_name:
                                condition_data['completion']['status'] = 'missing_data'
                                condition_data['completion']['success'] = False
                                condition_data['completion']['error'] = 'Participant data not available for this config'

    def _generate_summary_report_with_session_context(self) -> Path:
        """Generate summary report with session enhancements"""
        # Use existing analyzer and plotter but with session-enhanced data
        stats = self.analyzer.compute_all_statistics()

        # Add session-specific statistics
        stats['session_info'] = self.session_info
        stats['availability_analysis'] = self._compute_availability_analysis()

        # Generate plots using existing plotter
        summary_plots = self.plotter.generate_summary_plots(stats)

        # Add session-specific plots
        session_plots = self._generate_session_specific_plots(stats)
        summary_plots.update(session_plots)

        # Use existing HTML generator with enhanced data
        output_path = self.quality_dir / "session_quality_summary_report.html"
        return self.html_generator.create_summary_report(
            output_path=output_path,
            stats=stats,
            plots=summary_plots
        )

    def _compute_availability_analysis(self) -> Dict:
        """Compute session-specific availability statistics"""
        analysis = {
            'total_configs': len(self.session_info.get('configs_processed', [])),
            'participant_coverage': {},
            'missing_data_summary': {}
        }

        configs = self.session_info.get('configs_processed', [])

        for participant_id, availability_data in self.availability.items():
            available_configs = sum(1 for data in availability_data.values() if data['status'] == 'available')
            analysis['participant_coverage'][participant_id] = {
                'available_configs': available_configs,
                'total_configs': len(configs),
                'coverage_rate': (available_configs / len(configs) * 100) if configs else 0
            }

        # Summary of missing data patterns
        for config in configs:
            missing_count = sum(
                1 for avail_data in self.availability.values()
                if avail_data.get(config, {}).get('status') == 'missing'
            )
            analysis['missing_data_summary'][config] = {
                'missing_participants': missing_count,
                'total_participants': len(self.availability),
                'availability_rate': ((len(self.availability) - missing_count) / len(
                    self.availability) * 100) if self.availability else 0
            }

        return analysis

    def _generate_session_specific_plots(self, stats: Dict) -> Dict[str, str]:
        """Generate additional plots specific to session analysis"""
        return {
            'availability_matrix': self._plot_availability_matrix(),
            'config_comparison': self._plot_config_comparison()
        }

    def _plot_availability_matrix(self) -> str:
        """Plot participant availability across configs - minimal implementation"""
        import matplotlib.pyplot as plt
        import numpy as np

        participants = list(self.availability.keys())
        configs = self.session_info.get('configs_processed', [])

        if not participants or not configs:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No availability data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Participant Availability Matrix')
            return self.plotter._plot_to_base64(fig)

        # Create availability matrix
        matrix = np.zeros((len(participants), len(configs)))
        for i, participant in enumerate(participants):
            for j, config in enumerate(configs):
                status = self.availability[participant].get(config, {}).get('status', 'missing')
                matrix[i, j] = 1 if status == 'available' else 0

        fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.2), max(6, len(participants) * 0.4)))

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45)
        ax.set_yticks(range(len(participants)))
        ax.set_yticklabels(participants)
        ax.set_title('Participant Data Availability Matrix\n(Green=Available, Red=Missing)')

        plt.tight_layout()
        return self.plotter._plot_to_base64(fig)

    def _plot_config_comparison(self) -> str:
        """Simple config comparison plot"""
        import matplotlib.pyplot as plt

        configs = self.session_info.get('configs_processed', [])

        fig, ax = plt.subplots(figsize=(10, 6))

        if configs:
            # Count participants per config
            config_counts = []
            for config in configs:
                count = sum(
                    1 for avail_data in self.availability.values()
                    if avail_data.get(config, {}).get('status') == 'available'
                )
                config_counts.append(count)

            bars = ax.bar(configs, config_counts, color='skyblue', alpha=0.7)
            for bar, count in zip(bars, config_counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')

            ax.set_title('Participants Available per Config')
            ax.set_ylabel('Number of Participants')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No config data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Config Comparison')

        plt.tight_layout()
        return self.plotter._plot_to_base64(fig)