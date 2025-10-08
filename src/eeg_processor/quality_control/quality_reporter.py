"""
Quality Reporter

Main orchestrator for the EEG quality reporting system.
Automatically detects pipeline stages and generates relevant, research-focused reports.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

from .pipeline_detector import PipelineDetector
from .quality_flagging import QualityFlagger
from .quality_plot_generator import QualityPlotGenerator
from .quality_html_generator import QualityHTMLGenerator


class QualityReporter:
    """
    Main adaptive quality reporter that:
    1. Detects which processing stages were used
    2. Flags participants based on actual data quality issues
    3. Generates only relevant plots and sections
    4. Creates clean, research-focused HTML reports
    
    Maintains the same API as the original QualityReporter for seamless integration.
    """
    
    def __init__(self, metrics_file: Path):
        """
        Initialize quality reporter with metrics file.
        
        Args:
            metrics_file: Path to the quality metrics JSON file
        """
        self.metrics_file = Path(metrics_file)
        self.quality_dir = self.metrics_file.parent
        
        # Load and analyze the metrics data
        import json
        with open(self.metrics_file, 'r') as f:
            self.data = json.load(f)
            
        # Initialize adaptive components
        self.pipeline_detector = PipelineDetector(self.data['participants'])
        self.pipeline_info = self.pipeline_detector.pipeline_info
        
        # Get quality thresholds based on detected pipeline
        self.quality_thresholds = self.pipeline_detector.get_quality_thresholds()
        
        # Initialize quality flagger
        self.quality_flagger = QualityFlagger(self.pipeline_info, self.quality_thresholds)
        
        # Initialize plot generator
        self.plot_generator = QualityPlotGenerator(self.pipeline_info, self.quality_thresholds)
        
        # Initialize HTML generator
        self.html_generator = QualityHTMLGenerator(self.pipeline_info)
        
        logger.info(f"QualityReporter initialized for {self.pipeline_info['data_type']} "
                   f"pipeline with {len(self.pipeline_info['stages_used'])} stages")
    
    def generate_all_reports(self) -> Tuple[Path, List[Path]]:
        """
        Generate quality reports.
        
        Returns all quality reports including summary and individual participant reports.
        
        Returns:
            Tuple of (summary_report_path, list_of_participant_report_paths)
        """
        logger.info("Generating quality reports...")
        
        # Step 1: Flag all participants based on data quality
        flagged_participants = self.quality_flagger.flag_all_participants(self.data['participants'])
        
        # Step 2: Compute basic statistics
        stats = self._compute_basic_statistics()
        
        # Step 3: Generate adaptive plots
        plots = self.plot_generator.generate_adaptive_plots(stats, flagged_participants)
        
        # Step 4: Create summary report
        summary_path = self._generate_summary_report(stats, plots, flagged_participants)
        
        # Step 5: Generate simplified individual reports (placeholder for now)
        participant_paths = self._generate_participant_reports(flagged_participants)
        
        logger.success(f"Quality reports generated: summary + {len(participant_paths)} individual reports")
        logger.info(f"Quality summary: {len(flagged_participants['critical'])} critical, "
                   f"{len(flagged_participants['warning'])} warning, "
                   f"{len(flagged_participants['good'])} good")
        
        return summary_path, participant_paths
    
    def _compute_basic_statistics(self) -> Dict:
        """Compute basic statistics needed for reporting."""
        participants = self.data['participants']
        dataset_info = self.data['dataset_info']
        
        # Basic completion statistics
        total_participants = len(participants)
        completed_participants = sum(1 for p in participants.values() if p['completed'])
        
        return {
            'dataset_info': dataset_info,
            'participants': participants,
            'completion_stats': {
                'total_participants': total_participants,
                'completed_participants': completed_participants,
                'completion_rate': (completed_participants / total_participants * 100) if total_participants > 0 else 0
            }
        }
    
    def _generate_summary_report(self, stats: Dict, plots: Dict[str, str], 
                               flagged_participants: Dict) -> Path:
        """Generate the main summary report."""
        logger.info("Creating summary report...")
        
        output_path = self.quality_dir / "quality_summary_report.html"
        
        return self.html_generator.create_summary_report(
            output_path=output_path,
            stats=stats,
            plots=plots,
            flagged_participants=flagged_participants
        )
    
    def _generate_participant_reports(self, flagged_participants: Dict) -> List[Path]:
        """Generate simplified individual participant reports."""
        logger.info("Creating simplified individual participant reports...")
        
        individual_dir = self.quality_dir / "individual_reports"
        individual_dir.mkdir(exist_ok=True)
        
        report_paths = []
        
        # For now, create simple text-based individual reports
        # This can be enhanced later with full HTML reports if needed
        for level, participants in flagged_participants.items():
            for participant_info in participants:
                participant_id = participant_info['participant_id']
                output_path = individual_dir / f"{participant_id}_quality_report.html"
                
                # Create simple individual report
                html_content = self._create_simple_participant_report(participant_info)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
                report_paths.append(output_path)
        
        logger.info(f"Created {len(report_paths)} individual participant reports")
        return report_paths
    
    def _create_simple_participant_report(self, participant_info: Dict) -> str:
        """Create a detailed individual participant report."""
        participant_id = participant_info['participant_id']
        flag_level = participant_info['flag_level']
        reasons = participant_info['reasons']
        participant_data = participant_info['participant_data']
        
        # Get detailed quality information
        bad_channel_details = self._get_bad_channel_details(participant_data)
        epoch_rejection_details = self._get_epoch_rejection_details(participant_data)
        
        # Color coding based on flag level
        colors = {
            'critical': '#dc3545',
            'warning': '#ffc107', 
            'good': '#28a745'
        }
        
        status_icons = {
            'critical': '🚨',
            'warning': '⚠️',
            'good': '✅'
        }
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Report: {participant_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 2rem; line-height: 1.6; color: #333; }}
        .header {{ background: white; padding: 2rem; border-radius: 8px; 
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
        .status {{ padding: 1rem; border-radius: 8px; margin: 1rem 0; 
                   background-color: {colors[flag_level]}20; border-left: 4px solid {colors[flag_level]}; }}
        .back-link {{ margin-bottom: 1rem; }}
        .back-link a {{ color: #007bff; text-decoration: none; }}
        .back-link a:hover {{ text-decoration: underline; }}
        ul {{ margin: 1rem 0; }}
        li {{ margin: 0.5rem 0; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="back-link">
            <a href="../quality_summary_report.html">← Back to Summary Report</a>
        </div>
        <h1>{status_icons[flag_level]} Quality Report: {participant_id}</h1>
        
        <div class="status">
            <h2>Quality Status: {flag_level.title()}</h2>
            {self._format_participant_reasons(reasons)}
        </div>
        
        {self._format_detailed_quality_info(bad_channel_details, epoch_rejection_details)}
        
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 6px;">
            <h3>Pipeline Information</h3>
            <p><strong>Data Type:</strong> {self.pipeline_info['data_type'].title()}</p>
            <p><strong>Stages Used:</strong> {len(self.pipeline_info['stages_used'])}</p>
            <p><strong>Key Features:</strong> {', '.join(self.pipeline_detector._get_key_features())}</p>
        </div>
    </div>
</body>
</html>"""
    
    def _format_participant_reasons(self, reasons: List[str]) -> str:
        """Format the list of quality issues for display."""
        if not reasons:
            return "<p>No quality issues detected. Processing completed successfully.</p>"
        
        html = "<p>Quality issues identified:</p><ul>"
        for reason in reasons:
            html += f"<li>{reason}</li>"
        html += "</ul>"
        
        return html
    
    def _get_bad_channel_details(self, participant_data: Dict) -> Dict:
        """Get detailed bad channel information for individual reports."""
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'detect_bad_channels' in stages:
                metrics = stages['detect_bad_channels'].get('metrics', {})
                interpolation_details = metrics.get('interpolation_details', {})
                return {
                    'bad_channels': metrics.get('detected_bads', []),
                    'interpolated_channels': interpolation_details.get('successfully_interpolated', []),
                    'still_bad_channels': interpolation_details.get('still_noisy', []),
                    'interpolation_successful': metrics.get('interpolation_successful', False),
                    'n_detected': metrics.get('n_detected', 0),
                    'bad_percentage': interpolation_details.get('bad_percentage_before', 0)
                }
        return {}
    
    def _get_epoch_rejection_details(self, participant_data: Dict) -> Dict:
        """Get detailed epoch rejection information for individual reports."""
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'epoch' in stages:
                metrics = stages['epoch'].get('metrics', {})
                return {
                    'rejection_rate': metrics.get('rejection_rate', 0),
                    'rejection_reasons_by_channel': metrics.get('rejection_reasons', {}),  # Note: tracker provides reason->count, not channel->count
                    'n_epochs_rejected': metrics.get('rejected_epochs', 0),
                    'n_epochs_total': metrics.get('total_epochs', 0)
                }
        return {}
    
    def _format_detailed_quality_info(self, bad_channel_details: Dict, epoch_rejection_details: Dict) -> str:
        """Format detailed quality information for individual reports."""
        html = ""
        
        # Bad channel details
        if bad_channel_details and self.pipeline_info['has_bad_channels']:
            html += '''
        <div style="margin-top: 2rem; padding: 1rem; background-color: #fff; border-radius: 6px; border-left: 3px solid #007bff;">
            <h3>Bad Channel Analysis</h3>'''
            
            bad_channels = bad_channel_details.get('bad_channels', [])
            interpolated = bad_channel_details.get('interpolated_channels', [])
            still_bad = bad_channel_details.get('still_bad_channels', [])
            
            if bad_channels:
                html += f'''
            <p><strong>Channels detected as bad:</strong> {', '.join(bad_channels)} ({len(bad_channels)} total)</p>
            <p><strong>Bad channel percentage:</strong> {bad_channel_details.get('bad_percentage', 0):.1f}%</p>'''
                
                if interpolated:
                    html += f'''
            <p><strong>Successfully interpolated:</strong> {', '.join(interpolated)} ({len(interpolated)} channels)</p>'''
                
                if still_bad:
                    html += f'''
            <p style="color: #dc3545;"><strong>Still bad after interpolation:</strong> {', '.join(still_bad)} ({len(still_bad)} channels)</p>'''
                else:
                    html += f'''
            <p style="color: #28a745;"><strong>All bad channels successfully interpolated</strong></p>'''
            else:
                html += '''
            <p style="color: #28a745;"><strong>No bad channels detected</strong></p>'''
            
            html += '''
        </div>'''
        
        # Epoch rejection details
        if epoch_rejection_details and self.pipeline_info['has_epoching']:
            html += '''
        <div style="margin-top: 2rem; padding: 1rem; background-color: #fff; border-radius: 6px; border-left: 3px solid #28a745;">
            <h3>Epoch Rejection Analysis</h3>'''
            
            rejection_rate = epoch_rejection_details.get('rejection_rate', 0)
            n_rejected = epoch_rejection_details.get('n_epochs_rejected', 0)
            n_total = epoch_rejection_details.get('n_epochs_total', 0)
            rejection_reasons = epoch_rejection_details.get('rejection_reasons_by_channel', {})
            
            html += f'''
            <p><strong>Rejection rate:</strong> {rejection_rate:.1f}% ({n_rejected}/{n_total} epochs)</p>'''
            
            if rejection_reasons:
                # Show rejection reasons (not channels, as tracker provides reason->count)
                html += '''
            <p><strong>Rejection reasons:</strong></p>
            <ul>'''
                for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                    html += f'''
                <li>{reason}: {count} epoch rejections</li>'''
                html += '''
            </ul>'''
            else:
                html += '''
            <p style="color: #28a745;"><strong>No specific rejection reasons tracked</strong></p>'''
            
            html += '''
        </div>'''
        
        return html
    
    # Utility methods for external access
    def get_pipeline_info(self) -> Dict:
        """Get information about the detected pipeline."""
        return self.pipeline_info
    
    def get_quality_summary(self) -> Dict:
        """Get a summary of quality flags for external use."""
        flagged_participants = self.quality_flagger.flag_all_participants(self.data['participants'])
        return self.quality_flagger.get_quality_summary(flagged_participants)


# Standalone function for generating quality reports
def generate_quality_reports(results_dir: Path, set_backend: bool = True) -> Tuple[Path, List[Path]]:
    """
    Generate quality reports from saved metrics.
    
    Main entry point for quality report generation from the pipeline.
    
    Args:
        results_dir: Results directory containing quality/quality_metrics.json
        set_backend: Whether to set matplotlib backend to 'Agg' (default: True)
        
    Returns:
        Tuple of (summary_report_path, list_of_participant_report_paths)
    """
    # Set matplotlib backend for non-interactive plot generation if requested
    if set_backend:
        from .quality_plot_generator import QualityPlotGenerator
        QualityPlotGenerator.set_matplotlib_backend('Agg')
    
    results_dir = Path(results_dir)
    quality_dir = results_dir / "quality"
    metrics_file = quality_dir / "quality_metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Quality metrics file not found: {metrics_file}")
    
    reporter = QualityReporter(metrics_file)
    return reporter.generate_all_reports()