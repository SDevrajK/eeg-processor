from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

from .quality_metrics_analyzer import QualityMetricsAnalyzer
from .quality_plot_generator import QualityPlotGenerator
from .quality_html_generator import QualityHTMLGenerator


class QualityReporter:
    """
    Main orchestrator for quality report generation.

    Coordinates between data analysis, plot generation, and HTML creation
    to produce comprehensive quality reports. Much smaller and focused
    after refactoring into specialized components.
    """

    def __init__(self, metrics_file: Path):
        """
        Initialize quality reporter with metrics file.

        Args:
            metrics_file: Path to the quality metrics JSON file
        """
        self.metrics_file = Path(metrics_file)
        self.quality_dir = self.metrics_file.parent

        # Initialize specialized components
        self.analyzer = QualityMetricsAnalyzer(metrics_file)
        self.plotter = QualityPlotGenerator(thresholds=self.analyzer.thresholds)
        self.html_generator = QualityHTMLGenerator()

        logger.info("QualityReporter initialized with refactored components")

    def generate_all_reports(self) -> Tuple[Path, List[Path]]:
        """
        Generate both summary and individual participant reports.

        This is the main public API method that preserves exact same usage
        as the original QualityReporter.

        Returns:
            Tuple of (summary_report_path, list_of_participant_report_paths)
        """
        logger.info("Generating enhanced quality reports...")

        # Step 1: Analyze all data and compute statistics
        stats = self.analyzer.compute_all_statistics()

        # Step 2: Generate summary report
        summary_path = self._generate_summary_report(stats)

        # Step 3: Generate individual participant reports
        participant_paths = self._generate_participant_reports(stats)

        logger.success(f"Enhanced reports generated: summary + {len(participant_paths)} individual reports")
        return summary_path, participant_paths

    def _generate_summary_report(self, stats: Dict) -> Path:
        """Generate comprehensive dataset overview HTML report"""
        logger.info("Creating summary report...")

        output_path = self.quality_dir / "quality_summary_report.html"

        # Generate all plots for summary report
        summary_plots = self.plotter.generate_summary_plots(stats)

        # Create HTML report
        return self.html_generator.create_summary_report(
            output_path=output_path,
            stats=stats,
            plots=summary_plots
        )

    def _generate_participant_reports(self, stats: Dict) -> List[Path]:
        """Generate enhanced individual participant reports"""
        logger.info("Creating individual participant reports...")

        participants = stats['participants']
        individual_dir = self.quality_dir / "individual_reports"

        # Prepare participant data and generate all plots
        participants_data = {}
        participant_plots = {}

        for participant_id in participants.keys():
            # Get participant-specific data
            participant_stats = self.analyzer.get_participant_condition_data(participant_id)
            participants_data[participant_id] = participant_stats

            # Generate participant-specific plots
            participant_plots[participant_id] = self.plotter.generate_participant_plots(
                participant_id, participant_stats
            )

        # Create all HTML reports
        return self.html_generator.create_participant_reports(
            individual_dir=individual_dir,
            participants_data=participants_data,
            participant_plots=participant_plots
        )

    # Mock data generation for testing (simplified from original)
    def generate_mock_report(self, output_dir: Path) -> Path:
        """
        Generate a test report with simple hardcoded mock data.

        Args:
            output_dir: Directory where to save the mock report

        Returns:
            Path to the generated mock report
        """
        from .mock_data import get_simple_mock_data

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Use simple hardcoded mock data instead of complex randomization
        mock_data = get_simple_mock_data()

        # Temporarily replace analyzer data for mock generation
        original_data = self.analyzer.data
        self.analyzer.data = mock_data

        try:
            # Generate mock report using same pipeline
            stats = self.analyzer.compute_all_statistics()
            summary_plots = self.plotter.generate_summary_plots(stats)

            output_path = output_dir / "mock_quality_summary_report.html"
            result_path = self.html_generator.create_summary_report(
                output_path=output_path,
                stats=stats,
                plots=summary_plots
            )

            logger.info(f"Mock quality report generated: {result_path}")
            return result_path

        finally:
            # Restore original data
            self.analyzer.data = original_data

    @classmethod
    def create_mock_reporter(cls, output_dir: Path):
        """
        Class method to create a QualityReporter instance for mock report generation.

        Creates a reporter without requiring real metrics file.
        """
        # Create a temporary mock reporter without real metrics file
        reporter = cls.__new__(cls)  # Create instance without calling __init__

        # Set required attributes
        reporter.quality_dir = Path(output_dir)
        reporter.quality_dir.mkdir(exist_ok=True)

        # Initialize components with mock data
        from .mock_data import get_simple_mock_data
        mock_data = get_simple_mock_data()

        # Create a temporary mock analyzer
        reporter.analyzer = QualityMetricsAnalyzer.__new__(QualityMetricsAnalyzer)
        reporter.analyzer.data = mock_data
        reporter.analyzer.thresholds = {
            'bad_channels': {'warning': 4, 'critical': 8},
            'rejection_rate': {'warning': 15, 'critical': 30},
            'ica_components': {'warning': 16, 'critical': 24}
        }

        # Initialize other components
        reporter.plotter = QualityPlotGenerator(thresholds=reporter.analyzer.thresholds)
        reporter.html_generator = QualityHTMLGenerator()

        return reporter

    # Legacy compatibility methods (if needed)
    def get_thresholds(self) -> Dict:
        """Get quality thresholds used by the reporter"""
        return self.analyzer.thresholds

    def get_data_summary(self) -> Dict:
        """Get basic summary of loaded data"""
        return {
            'total_participants': len(self.analyzer.data['participants']),
            'dataset_start': self.analyzer.data['dataset_info']['start_time'],
            'dataset_end': self.analyzer.data['dataset_info']['end_time']
        }


# Standalone function to maintain exact same API as original
def generate_quality_reports(results_dir: Path) -> Tuple[Path, List[Path]]:
    """
    Generate enhanced quality reports from saved metrics.

    This function maintains the exact same API as the original implementation
    for backward compatibility.

    Args:
        results_dir: Results directory containing quality/quality_metrics.json

    Returns:
        Tuple of (summary_report_path, list_of_participant_report_paths)
    """
    quality_dir = results_dir / "quality"
    metrics_file = quality_dir / "quality_metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Quality metrics file not found: {metrics_file}")

    reporter = QualityReporter(metrics_file)
    return reporter.generate_all_reports()