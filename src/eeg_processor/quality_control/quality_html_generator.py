# quality_control/quality_html_generator.py

from pathlib import Path
from typing import Dict, List
from datetime import datetime
from loguru import logger


class QualityHTMLGenerator:
    """
    Generates HTML reports for quality control data.

    Handles all HTML template generation and file creation for both
    summary and individual participant reports.
    """

    def __init__(self):
        """Initialize HTML generator with common CSS styles"""
        self.common_css = self._get_common_css()

    def create_summary_report(self, output_path: Path, stats: Dict, plots: Dict[str, str]) -> Path:
        """
        Create the main summary HTML report.

        Args:
            output_path: Path where to save the HTML file
            stats: Complete statistics from QualityMetricsAnalyzer
            plots: Dictionary of plot names to base64 encoded images

        Returns:
            Path to the created HTML file
        """
        html_content = self._create_summary_html_content(stats, plots)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Summary report created: {output_path}")
        return output_path

    def create_participant_reports(self, individual_dir: Path, participants_data: Dict,
                                   participant_plots: Dict[str, Dict[str, str]]) -> List[Path]:
        """
        Create individual participant HTML reports.

        Args:
            individual_dir: Directory where to save individual reports
            participants_data: Dictionary of participant_id -> participant stats
            participant_plots: Dictionary of participant_id -> plots

        Returns:
            List of paths to created HTML files
        """
        individual_dir.mkdir(exist_ok=True)
        report_paths = []

        for participant_id, participant_stats in participants_data.items():
            output_path = individual_dir / f"{participant_id}_quality_report.html"
            plots = participant_plots.get(participant_id, {})

            html_content = self._create_participant_html_content(
                participant_id, participant_stats, plots
            )

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            report_paths.append(output_path)

        logger.info(f"Created {len(report_paths)} individual participant reports")
        return report_paths

    def _create_summary_html_content(self, stats: Dict, plots: Dict[str, str]) -> str:
        """Create the main summary report HTML content"""
        dataset_info = stats['dataset_info']
        completion_stats = stats['completion_stats']
        quality_stats = stats['quality_stats']
        processing_stats = stats['processing_stats']

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEG Processing Quality Summary Report</title>
    <style>
        {self.common_css}
        {self._get_summary_specific_css()}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 EEG Processing Quality Summary Report</h1>
        <p style="font-size: 1.2em; color: #7f8c8d; margin: 1rem 0 0 0;">Comprehensive Analysis Dashboard</p>
    </div>

    <div class="container">
        <div class="content">
            {self._create_basic_stats_section(completion_stats, processing_stats)}

            {self._create_quality_indicators_section(quality_stats)}

            {self._create_dashboard_section(plots)}

            {self._create_detailed_analysis_section(plots)}

            {self._create_bad_channels_section(plots)}

            {self._create_processing_overview_section(plots)}
        </div>
    </div>

    {self._create_footer(dataset_info)}
</body>
</html>"""

    def _create_participant_html_content(self, participant_id: str, participant_stats: Dict,
                                         plots: Dict[str, str]) -> str:
        """Create individual participant report HTML content"""
        participant_data = participant_stats['participant_data']
        summary_stats = participant_stats['summary_stats']

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report: {participant_id}</title>
    <style>
        {self.common_css}
        {self._get_participant_specific_css()}
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
            {self._create_participant_summary_card(summary_stats)}

            {self._create_participant_plots_sections(plots)}
        </div>
    </div>

    {self._create_participant_footer(participant_id, participant_data)}
</body>
</html>"""

    def _create_basic_stats_section(self, completion_stats: Dict, processing_stats: Dict) -> str:
        """Create the basic statistics grid section"""
        return f"""
            <!-- Basic Statistics -->
            <div class="basic-stats">
                <div class="stat-item">
                    <div class="stat-number">{completion_stats['total_participants']}</div>
                    <div class="stat-label">Total Participants</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{completion_stats['completed_participants']}</div>
                    <div class="stat-label">Completed Successfully</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{completion_stats['completion_rate']:.1f}%</div>
                    <div class="stat-label">Completion Rate</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{processing_stats['avg_processing_time']:.1f}m</div>
                    <div class="stat-label">Avg Processing Time</div>
                </div>
            </div>"""

    def _create_quality_indicators_section(self, quality_stats: Dict) -> str:
        """Create the quality indicators section"""
        return f"""
            <!-- Quality Indicators -->
            <div class="quality-indicators">
                <h3 style="margin-top: 0;">🎯 Quality Status</h3>
                <span class="quality-indicator quality-good">Good Quality: {quality_stats['quality_flags']['good_participants']} participants</span>
                <span class="quality-indicator quality-warning">Warning Level: {quality_stats['quality_flags']['warning_participants']} participants</span>
                <span class="quality-indicator quality-critical">Critical Issues: {quality_stats['quality_flags']['critical_participants']} participants</span>
                <span style="margin-left: 1rem;"><strong>Interpolation Success:</strong> {quality_stats['interpolation_success_rate']:.1f}%</span>
            </div>"""

    def _create_dashboard_section(self, plots: Dict[str, str]) -> str:
        """Create the quality control dashboard section"""
        return f"""
            <!-- Critical Information Dashboard - First and Prominent -->
            <h2>🎛️ Quality Control Dashboard</h2>
            <div class="plot-grid single-column">
                <div class="plot-container">
                    <img src="{plots['dashboard_summary']}" alt="Quality Dashboard">
                </div>
            </div>"""

    def _create_detailed_analysis_section(self, plots: Dict[str, str]) -> str:
        """Create the detailed processing analysis section"""
        return f"""
            <!-- Main Analysis Plots - 2x2 Grid -->
            <h2>📊 Detailed Processing Analysis</h2>
            <div class="plot-grid">
                <div class="plot-container">
                    <h3>Processing Completion Matrix</h3>
                    <img src="{plots['completion_matrix']}" alt="Completion Matrix">
                </div>
                <div class="plot-container">
                    <h3>Quality Metrics Distributions</h3>
                    <img src="{plots['quality_distributions']}" alt="Quality Distributions">
                </div>
            </div>"""

    def _create_bad_channels_section(self, plots: Dict[str, str]) -> str:
        """Create the bad channels analysis section"""
        return f"""
            <!-- Bad Channels Analysis - Full Width -->
            <h2>🔍 Bad Channels by Participant</h2>
            <div class="plot-grid single-column">
                <div class="plot-container">
                    <img src="{plots['participant_bad_channels']}" alt="Bad Channels by Participant">
                </div>
            </div>"""

    def _create_processing_overview_section(self, plots: Dict[str, str]) -> str:
        """Create the processing overview section"""
        return f"""
            <!-- Processing Overview - Full Width at End -->
            <h2>⏱️ Processing Overview</h2>
            <div class="plot-grid single-column">
                <div class="plot-container">
                    <img src="{plots['processing_overview']}" alt="Processing Overview">
                </div>
            </div>"""

    def _create_participant_summary_card(self, summary_stats: Dict) -> str:
        """Create the participant summary card"""
        return f"""
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
            </div>"""

    def _create_participant_plots_sections(self, plots: Dict[str, str]) -> str:
        """Create all participant plot sections"""
        return f"""
            <h2>📈 Condition Overview</h2>
            <div class="plot-container">
                <img src="{plots.get('condition_overview', '')}" alt="Condition Overview">
            </div>

            <h2>🔍 Quality Details</h2>
            <div class="plot-container">
                <img src="{plots.get('quality_details', '')}" alt="Quality Details">
            </div>

            <h2>⏱️ Processing Timeline</h2>
            <div class="plot-container">
                <img src="{plots.get('processing_timeline', '')}" alt="Processing Timeline">
            </div>

            <h2>📋 Detailed Metrics</h2>
            <div class="plot-container">
                <img src="{plots.get('detailed_metrics', '')}" alt="Detailed Metrics">
            </div>"""

    def _create_footer(self, dataset_info: Dict) -> str:
        """Create the main report footer"""
        return f"""
    <div class="footer">
        <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Processing Period:</strong> {dataset_info['start_time']} to {dataset_info['end_time']}</p>
        <p>🔬 Advanced EEG Quality Control System</p>
    </div>"""

    def _create_participant_footer(self, participant_id: str, participant_data: Dict) -> str:
        """Create the participant report footer"""
        return f"""
    <div class="footer">
        <p><strong>Individual Report for:</strong> {participant_id}</p>
        <p><strong>Processing Period:</strong> {participant_data['start_time']} to {participant_data.get('end_time', 'In Progress')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>"""

    def _get_common_css(self) -> str:
        """Get common CSS styles used in both report types"""
        return """
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .header { 
            background: rgba(255,255,255,0.95); 
            padding: 2rem; 
            text-align: center; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }
        .container { 
            max-width: 1400px; 
            margin: 2rem auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            overflow: hidden; 
        }
        .content { 
            padding: 2rem; 
        }
        h1 { 
            color: #2c3e50; 
            margin: 0; 
            font-size: 2.5em; 
            font-weight: 700; 
        }
        h2 { 
            color: #34495e; 
            margin: 2rem 0 1rem 0; 
            font-size: 1.8em; 
            border-left: 4px solid #3498db; 
            padding-left: 1rem; 
        }
        h3 {
            color: #34495e;
            margin: 1rem 0;
            font-size: 1.2em;
        }
        .plot-container { 
            text-align: center; 
            background: #f8f9fa; 
            padding: 1.5rem; 
            border-radius: 12px; 
            margin: 1rem 0;
        }
        .plot-container img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        }
        .quality-indicator { 
            display: inline-block; 
            padding: 0.5rem 1rem; 
            border-radius: 20px; 
            font-weight: bold; 
            margin: 0.25rem; 
        }
        .quality-badge { 
            display: inline-block; 
            padding: 0.5rem 1rem; 
            border-radius: 20px; 
            font-weight: bold; 
            margin: 0.5rem; 
        }
        .quality-good { 
            background: #2ecc71; 
            color: white; 
        }
        .quality-warning { 
            background: #f39c12; 
            color: white; 
        }
        .quality-critical { 
            background: #e74c3c; 
            color: white; 
        }
        .footer { 
            background: #2c3e50; 
            color: white; 
            padding: 2rem; 
            text-align: center; 
        }
        .status-indicator { 
            font-size: 1.5em; 
            margin: 0.5rem; 
        }"""

    def _get_summary_specific_css(self) -> str:
        """Get CSS specific to summary reports"""
        return """
        /* Horizontal Grid Layout */
        .plot-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 2rem; 
            margin: 2rem 0; 
        }
        .plot-grid.single-column { 
            grid-template-columns: 1fr; 
        }

        /* Basic stats summary */
        .basic-stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 1rem; 
            margin: 2rem 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 1.5rem; 
            border-radius: 12px; 
        }
        .stat-item { 
            text-align: center; 
        }
        .stat-number { 
            font-size: 2em; 
            font-weight: bold; 
        }
        .stat-label { 
            font-size: 0.9em; 
            opacity: 0.9; 
        }

        /* Quality indicators */
        .quality-indicators { 
            background: #f8f9fa; 
            padding: 1.5rem; 
            border-radius: 12px; 
            margin: 2rem 0; 
        }"""

    def _get_participant_specific_css(self) -> str:
        """Get CSS specific to participant reports"""
        return """
        .back-link { 
            margin-bottom: 2rem; 
        }
        .back-link a { 
            color: #3498db; 
            text-decoration: none; 
            font-size: 1.1em; 
            font-weight: 500; 
        }
        .back-link a:hover { 
            text-decoration: underline; 
        }
        .summary-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 2rem; 
            border-radius: 12px; 
            margin: 2rem 0; 
        }
        .summary-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 1.5rem; 
            margin-top: 1rem; 
        }
        .summary-item { 
            text-align: center; 
        }
        .summary-number { 
            font-size: 2em; 
            font-weight: bold; 
        }
        .summary-label { 
            font-size: 0.9em; 
            opacity: 0.9; 
        }"""