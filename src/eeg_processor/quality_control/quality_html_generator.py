"""
Quality HTML Generator

Creates clean, research-focused HTML reports with critical issues prominently displayed.
Adapts content based on processing pipeline used.
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime
from loguru import logger


class QualityHTMLGenerator:
    """
    Generates clean, minimal HTML reports focused on EEG research priorities.
    
    Features:
    - Critical issues displayed prominently at top
    - Adaptive content based on pipeline stages used
    - Clean, scannable layout without visual clutter
    - Research-focused information hierarchy
    """
    
    def __init__(self, pipeline_info: Dict):
        """
        Initialize HTML generator.
        
        Args:
            pipeline_info: Information about which stages were used
        """
        self.pipeline_info = pipeline_info
        
    def create_summary_report(self, output_path: Path, stats: Dict, plots: Dict[str, str], 
                            flagged_participants: Dict) -> Path:
        """
        Create the main summary HTML report with critical alerts at top.
        
        Args:
            output_path: Path where to save the HTML file
            stats: Complete statistics from quality analyzer
            plots: Dictionary of plot names to base64 encoded images
            flagged_participants: Participants organized by flag level
            
        Returns:
            Path to the created HTML file
        """
        html_content = self._create_summary_html_content(stats, plots, flagged_participants)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Summary report created: {output_path}")
        return output_path
    
    def _create_summary_html_content(self, stats: Dict, plots: Dict[str, str], 
                                   flagged_participants: Dict) -> str:
        """Create the main summary report HTML content."""
        dataset_info = stats['dataset_info']
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEG Quality Control Report</title>
    <style>
        {self._get_clean_css()}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 EEG Quality Control Report</h1>
        <div class="header-info">
            <span>Dataset: {dataset_info.get('total_participants', 0)} participants</span>
            <span>Pipeline: {self.pipeline_info['data_type'].title()} data processing</span>
            <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>

    <div class="container">
        {self._create_critical_alerts_section(flagged_participants)}
        
        {self._create_pipeline_summary_section()}
        
        {self._create_quality_overview_section(plots)}
        
        {self._create_detailed_analysis_section(plots)}
    </div>

    <div class="footer">
        <p>EEG Processor Quality Control System | Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""

    def _create_critical_alerts_section(self, flagged_participants: Dict) -> str:
        """Create the critical alerts section at the top of the report."""
        critical_participants = flagged_participants['critical']
        warning_participants = flagged_participants['warning']
        
        html = '<div class="alerts-section">'
        
        # Critical issues (if any)
        if critical_participants:
            html += '''
            <div class="alert critical-alert">
                <h2>🚨 Critical Quality Issues</h2>
                <p>The following participants require immediate attention:</p>
                <table class="alert-table">
                    <thead>
                        <tr>
                            <th>Participant</th>
                            <th>Issues</th>
                        </tr>
                    </thead>
                    <tbody>'''
            
            for participant in critical_participants:
                participant_id = participant['participant_id']
                issues = '; '.join(participant['reasons'])
                html += f'''
                        <tr>
                            <td><strong>{participant_id}</strong></td>
                            <td>{issues}</td>
                        </tr>'''
            
            html += '''
                    </tbody>
                </table>
            </div>'''
        
        # Warning issues (if any)
        if warning_participants:
            html += '''
            <div class="alert warning-alert">
                <h2>⚠️ Quality Warnings</h2>
                <p>The following participants have quality concerns:</p>
                <table class="alert-table">
                    <thead>
                        <tr>
                            <th>Participant</th>
                            <th>Concerns</th>
                        </tr>
                    </thead>
                    <tbody>'''
            
            for participant in warning_participants:
                participant_id = participant['participant_id']
                issues = '; '.join(participant['reasons'])
                html += f'''
                        <tr>
                            <td><strong>{participant_id}</strong></td>
                            <td>{issues}</td>
                        </tr>'''
            
            html += '''
                    </tbody>
                </table>
            </div>'''
        
        # All good message (if no issues)
        if not critical_participants and not warning_participants:
            html += '''
            <div class="alert success-alert">
                <h2>✅ All Participants Processed Successfully</h2>
                <p>No critical quality issues detected. All participants meet quality standards.</p>
            </div>'''
        
        html += '</div>'
        return html
    
    def _create_pipeline_summary_section(self) -> str:
        """Create pipeline summary section."""
        pipeline_summary = self._get_pipeline_summary()
        
        return f'''
        <div class="section">
            <h2>📋 Processing Pipeline Summary</h2>
            <div class="pipeline-info">
                <div class="pipeline-item">
                    <strong>Data Type:</strong> {pipeline_summary['data_type']}
                </div>
                <div class="pipeline-item">
                    <strong>Total Stages:</strong> {pipeline_summary['total_stages']}
                </div>
                <div class="pipeline-item">
                    <strong>Key Features:</strong> {', '.join(pipeline_summary['key_features'])}
                </div>
                <div class="pipeline-item">
                    <strong>Processing Flow:</strong> {' → '.join(pipeline_summary['processing_flow'])}
                </div>
            </div>
        </div>'''
    
    def _create_quality_overview_section(self, plots: Dict[str, str]) -> str:
        """Create quality overview section with key plots."""
        html = '''
        <div class="section">
            <h2>📊 Quality Overview</h2>
            <div class="plot-grid">'''
        
        # Quality summary plot (always present)
        if 'quality_summary' in plots:
            html += f'''
                <div class="plot-container">
                    <h3>Overall Quality Distribution</h3>
                    <img src="{plots['quality_summary']}" alt="Quality Summary">
                </div>'''
        
        # Completion overview plot (always present)
        if 'completion_overview' in plots:
            html += f'''
                <div class="plot-container full-width">
                    <h3>Processing Stage Completion</h3>
                    <img src="{plots['completion_overview']}" alt="Completion Overview">
                </div>'''
        
        html += '''
            </div>
        </div>'''
        return html
    
    def _create_detailed_analysis_section(self, plots: Dict[str, str]) -> str:
        """Create detailed analysis section with stage-specific plots."""
        html = '''
        <div class="section">
            <h2>🔬 Detailed Quality Analysis</h2>
            <div class="plot-grid">'''
        
        # Add stage-specific plots only if they exist
        stage_plots = {
            'bad_channels': ('Bad Channels Analysis', 'Bad Channel Detection Results'),
            'epoch_rejection': ('Epoch Rejection Analysis', 'Epoch Quality Assessment'),
            'ica_components': ('ICA Analysis', 'Independent Component Analysis Results')
        }
        
        for plot_key, (title, description) in stage_plots.items():
            if plot_key in plots:
                html += f'''
                <div class="plot-container">
                    <h3>{title}</h3>
                    <p class="plot-description">{description}</p>
                    <img src="{plots[plot_key]}" alt="{title}">
                </div>'''
        
        html += '''
            </div>
        </div>'''
        return html
    
    def _get_pipeline_summary(self) -> Dict:
        """Get pipeline summary information for display."""
        # Key features based on detected stages
        features = []
        if self.pipeline_info['has_epoching']:
            features.append("Event-related analysis")
        else:
            features.append("Continuous data analysis")
            
        if self.pipeline_info['has_ica']:
            features.append("ICA artifact removal")
        if self.pipeline_info['has_asr']:
            features.append("ASR artifact correction")
        if self.pipeline_info['has_bad_channels']:
            features.append("Bad channel detection")
        
        # Processing flow - use correct order directly from stages_used
        # Define the correct processing order
        standard_order = [
            'crop',
            'segment_condition', 
            'compute_eog',
            'filter',
            'detect_bad_channels',
            'clean_rawdata_asr',
            'rereference',
            'blink_artifact',
            'epoch',
            'time_frequency'
        ]
        
        # Get only stages that were actually used, in the correct order
        used_stages = self.pipeline_info['stages_used']
        ordered_stages = [stage for stage in standard_order if stage in used_stages]
        
        # Map to display labels
        stage_labels = {
            'crop': 'Crop',
            'segment_condition': 'Segment',
            'compute_eog': 'EOG Computation',
            'filter': 'Filter', 
            'detect_bad_channels': 'Bad Channels',
            'clean_rawdata_asr': 'ASR',
            'rereference': 'Re-reference',
            'blink_artifact': 'ICA',
            'epoch': 'Epoch',
            'time_frequency': 'Time-Frequency'
        }
        
        flow_stages = [stage_labels.get(stage, stage.title()) for stage in ordered_stages]
        
        return {
            'data_type': self.pipeline_info['data_type'].title(),
            'total_stages': len(self.pipeline_info['stages_used']),
            'key_features': features,
            'processing_flow': flow_stages
        }
    
    def _get_clean_css(self) -> str:
        """Get clean, minimal CSS styling focused on readability."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .header {
            background: white;
            padding: 2rem;
            text-align: center;
            border-bottom: 3px solid #007bff;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 2.2em;
        }
        
        .header-info {
            color: #6c757d;
            font-size: 1.1em;
        }
        
        .header-info span {
            margin: 0 1rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem 2rem 2rem;
        }
        
        .section {
            background: white;
            margin-bottom: 2rem;
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .alerts-section {
            margin-bottom: 2rem;
        }
        
        .alert {
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        
        .critical-alert {
            background-color: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        
        .warning-alert {
            background-color: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
        }
        
        .success-alert {
            background-color: #d1edda;
            border-left-color: #28a745;
            color: #155724;
        }
        
        .alert h2 {
            margin-bottom: 1rem;
            font-size: 1.3em;
        }
        
        .alert-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .alert-table th,
        .alert-table td {
            text-align: left;
            padding: 0.8rem;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }
        
        .alert-table th {
            background-color: rgba(0,0,0,0.05);
            font-weight: bold;
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 1.5rem;
            font-size: 1.8em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 0.5rem;
        }
        
        h3 {
            color: #495057;
            margin-bottom: 1rem;
            font-size: 1.3em;
        }
        
        .pipeline-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .pipeline-item {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 6px;
            border-left: 3px solid #007bff;
        }
        
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-top: 1.5rem;
        }
        
        .plot-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
        }
        
        .plot-container.full-width {
            grid-column: 1 / -1;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            margin-top: 1rem;
        }
        
        .plot-description {
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 1rem;
        }
        
        .footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 1.5rem;
            margin-top: 3rem;
        }
        
        @media (max-width: 768px) {
            .plot-grid {
                grid-template-columns: 1fr;
            }
            
            .pipeline-info {
                grid-template-columns: 1fr;
            }
            
            .header-info span {
                display: block;
                margin: 0.5rem 0;
            }
        }
        """