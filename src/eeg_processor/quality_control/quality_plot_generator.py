"""
Quality Plot Generator

Generates relevant plots based on the processing pipeline that was used.
Focuses on EEG research priorities: stage completion, data quality issues.
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
from loguru import logger

matplotlib.use('Agg')


class QualityPlotGenerator:
    """
    Generates plots adaptively based on which processing stages were used.
    
    Creates clean, research-focused visualizations that highlight:
    1. Processing completion status
    2. Data quality issues
    3. Stage-specific quality metrics (only for stages that were used)
    """
    
    def __init__(self, pipeline_info: Dict, quality_thresholds: Dict):
        """
        Initialize quality plot generator.
        
        Args:
            pipeline_info: Information about which stages were used
            quality_thresholds: Threshold values for quality metrics
        """
        self.pipeline_info = pipeline_info
        self.thresholds = quality_thresholds
        
        # Configure matplotlib for clean, minimal plots
        plt.style.use('default')
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Configure clean, minimal plot styling."""
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
    
    def generate_adaptive_plots(self, stats: Dict, flagged_participants: Dict) -> Dict[str, str]:
        """
        Generate all relevant plots based on pipeline stages used.
        
        Args:
            stats: Complete statistics from quality analyzer
            flagged_participants: Participants organized by flag level
            
        Returns:
            Dictionary mapping plot names to base64 encoded images
        """
        plots = {}
        
        # Always generate: Processing completion overview
        plots['completion_overview'] = self.plot_completion_overview(stats, flagged_participants)
        
        # Always generate: Quality status summary  
        plots['quality_summary'] = self.plot_quality_summary(flagged_participants)
        
        # Generate stage-specific plots only if stages were used
        if self.pipeline_info['has_bad_channels']:
            plots['bad_channels'] = self.plot_bad_channels_analysis(stats['participants'])
            
        if self.pipeline_info['has_epoching']:
            plots['epoch_rejection'] = self.plot_epoch_rejection_analysis(stats['participants'])
            
        if self.pipeline_info['has_ica']:
            plots['ica_components'] = self.plot_ica_components_analysis(stats['participants'])
        
        logger.info(f"Generated {len(plots)} quality plots for pipeline")
        return plots
    
    def plot_completion_overview(self, stats: Dict, flagged_participants: Dict) -> str:
        """
        Create processing completion overview matrix.
        Shows which participants completed which stages successfully.
        """
        participants = stats['participants']
        
        # Create completion matrix
        participant_ids = list(participants.keys())
        stages = self.pipeline_info['stages_used']
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(participant_ids) * 0.3)))
        
        # Build completion matrix
        completion_matrix = []
        for participant_id in participant_ids:
            participant_row = []
            participant_data = participants[participant_id]
            
            for stage in stages:
                # Check if stage completed successfully across all conditions
                stage_success = True
                for condition_data in participant_data['conditions'].values():
                    if stage not in condition_data.get('stages', {}):
                        stage_success = False
                        break
                    stage_metrics = condition_data['stages'][stage].get('metrics', {})
                    if 'error' in stage_metrics or not stage_metrics.get('stage_completed', True):
                        stage_success = False
                        break
                        
                participant_row.append(1 if stage_success else 0)
            completion_matrix.append(participant_row)
        
        # Plot matrix
        completion_array = np.array(completion_matrix)
        im = ax.imshow(completion_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_yticks(range(len(participant_ids)))
        ax.set_yticklabels(participant_ids)
        
        # Color-code participant labels by flag level
        for i, participant_id in enumerate(participant_ids):
            flag_level = self._get_participant_flag_level(participant_id, flagged_participants)
            color = {'critical': 'red', 'warning': 'orange', 'good': 'black'}[flag_level]
            ax.get_yticklabels()[i].set_color(color)
        
        ax.set_title('Processing Stage Completion Matrix\n(Red participant = critical issues, Orange = warnings)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Processing Stages')
        ax.set_ylabel('Participants')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Stage Completed', rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Failed', 'Success'])
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_quality_summary(self, flagged_participants: Dict) -> str:
        """Create clean quality status summary chart."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Quality distribution bar chart (removed redundant pie chart)
        counts = [len(flagged_participants[level]) for level in ['good', 'warning', 'critical']]
        labels = ['Good Quality', 'Warning Issues', 'Critical Issues']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        # Only plot if there's data
        if sum(counts) > 0:
            bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title('Dataset Quality Distribution', fontweight='bold', fontsize=14)
            ax.set_ylabel('Number of Participants', fontsize=12)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Add percentage labels
            total = sum(counts)
            for i, count in enumerate(counts):
                percentage = (count / total * 100) if total > 0 else 0
                ax.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=10)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_bad_channels_analysis(self, participants: Dict) -> str:
        """Plot bad channel analysis showing electrode removal frequency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract bad channel data and count electrode occurrences
        electrode_removal_count = {}
        bad_channel_percentages = []
        interpolation_success = []
        participant_ids = []
        
        for participant_id, participant_data in participants.items():
            # Get bad channel metrics
            bad_channel_metrics = self._extract_bad_channel_metrics(participant_data)
            if bad_channel_metrics:
                bad_channel_percentages.append(bad_channel_metrics.get('bad_percentage_before', 0))
                interpolation_success.append(bad_channel_metrics.get('interpolation_successful', True))
                participant_ids.append(participant_id)
                
                # Count electrode removals
                detected_bads = bad_channel_metrics.get('detected_bads', [])
                for electrode in detected_bads:
                    electrode_removal_count[electrode] = electrode_removal_count.get(electrode, 0) + 1
        
        # Generate plots even if we don't have electrode-level data
        if bad_channel_percentages:  # Changed condition to check for any bad channel data
            
            # Left plot: Electrode removal frequency OR bad channel count distribution
            if electrode_removal_count:
                # Electrode removal frequency (only show electrodes removed > 0 times)
                electrodes = list(electrode_removal_count.keys())
                removal_counts = list(electrode_removal_count.values())
                
                # Sort by removal frequency
                sorted_data = sorted(zip(electrodes, removal_counts), key=lambda x: x[1], reverse=True)
                electrodes, removal_counts = zip(*sorted_data)
                
                bars = ax1.bar(range(len(electrodes)), removal_counts, 
                              alpha=0.7, color='steelblue', edgecolor='black')
                ax1.set_xlabel('Electrodes')
                ax1.set_ylabel('Number of Times Removed')
                ax1.set_title('Electrode Removal Frequency Across Participants', fontweight='bold')
                ax1.set_xticks(range(len(electrodes)))
                ax1.set_xticklabels(electrodes, rotation=45, ha='right')
                
                # Add count labels on bars
                for i, count in enumerate(removal_counts):
                    ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
            else:
                # Fallback: Show bad channel count distribution if no electrode-level data
                bad_channel_counts = []
                for participant_id, participant_data in participants.items():
                    bad_channel_metrics = self._extract_bad_channel_metrics(participant_data)
                    if bad_channel_metrics:
                        bad_channel_counts.append(bad_channel_metrics.get('n_detected', 0))
                
                if bad_channel_counts:
                    ax1.hist(bad_channel_counts, bins=max(5, len(set(bad_channel_counts))), 
                            alpha=0.7, color='steelblue', edgecolor='black')
                    ax1.axvline(self.thresholds['bad_channels']['max_reasonable'], 
                               color='orange', linestyle='--', linewidth=2, label='Warning threshold')
                    ax1.set_xlabel('Number of Bad Channels Detected')
                    ax1.set_ylabel('Number of Participants')
                    ax1.set_title('Bad Channel Count Distribution', fontweight='bold')
                    ax1.legend()
            
            # Right plot: Bad channel percentage by participant
            colors = ['red' if not success else 'steelblue' for success in interpolation_success]
            bars = ax2.bar(range(len(bad_channel_percentages)), bad_channel_percentages, 
                          color=colors, alpha=0.7)
            
            # Add threshold lines
            ax2.axhline(self.thresholds['bad_channels']['warning_percentage'], 
                       color='orange', linestyle='--', linewidth=2, label='Warning (10%)')
            ax2.axhline(self.thresholds['bad_channels']['critical_percentage'], 
                       color='red', linestyle='--', linewidth=2, label='Critical (25%)')
            
            ax2.set_xlabel('Participants')
            ax2.set_ylabel('Bad Channel Percentage')
            ax2.set_title('Bad Channel Percentage by Participant\n(Red bars = interpolation failed)', 
                         fontweight='bold')
            ax2.legend()
            ax2.set_xticks(range(len(participant_ids)))
            ax2.set_xticklabels(participant_ids, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_epoch_rejection_analysis(self, participants: Dict) -> str:
        """Plot epoch rejection analysis showing electrode-based rejection frequency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract epoch rejection data and count electrode-based rejections
        electrode_rejection_count = {}
        rejection_rates = []
        participant_ids = []
        
        for participant_id, participant_data in participants.items():
            # Get epoch metrics
            epoch_metrics = self._extract_epoch_metrics(participant_data)
            if epoch_metrics:
                rejection_rates.append(epoch_metrics.get('rejection_rate', 0))
                participant_ids.append(participant_id)
                
                # Count electrode-based rejections
                rejection_reasons = epoch_metrics.get('rejection_reasons', {})
                for electrode, count in rejection_reasons.items():
                    if count > 0 and electrode != 'IGNORED':  # Skip the IGNORED category
                        electrode_rejection_count[electrode] = electrode_rejection_count.get(electrode, 0) + count
        
        # Generate plots even if we don't have electrode-level rejection data
        if rejection_rates:  # Changed condition to check for any epoch data
            
            # Left plot: Electrode rejection frequency OR rejection rate distribution
            if electrode_rejection_count:
                # Electrode rejection frequency (only show electrodes that caused rejections > 0)
                electrodes = list(electrode_rejection_count.keys())
                rejection_counts = list(electrode_rejection_count.values())
                
                # Sort by rejection frequency
                sorted_data = sorted(zip(electrodes, rejection_counts), key=lambda x: x[1], reverse=True)
                electrodes, rejection_counts = zip(*sorted_data)
                
                bars = ax1.bar(range(len(electrodes)), rejection_counts, 
                              alpha=0.7, color='steelblue', edgecolor='black')
                ax1.set_xlabel('Electrodes')
                ax1.set_ylabel('Number of Epoch Rejections Caused')
                ax1.set_title('Electrode-Based Epoch Rejection Frequency', fontweight='bold')
                ax1.set_xticks(range(len(electrodes)))
                ax1.set_xticklabels(electrodes, rotation=45, ha='right')
                
                # Add count labels on bars
                for i, count in enumerate(rejection_counts):
                    ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
            else:
                # Fallback: Show rejection rate distribution if no electrode-level data
                ax1.hist(rejection_rates, bins=10, alpha=0.7, color='steelblue', edgecolor='black')
                ax1.axvline(self.thresholds['epoch_rejection']['warning_rate'], 
                           color='orange', linestyle='--', linewidth=2, label='Warning (20%)')
                ax1.axvline(self.thresholds['epoch_rejection']['critical_rate'], 
                           color='red', linestyle='--', linewidth=2, label='Critical (40%)')
                ax1.set_xlabel('Epoch Rejection Rate (%)')
                ax1.set_ylabel('Number of Participants')
                ax1.set_title('Epoch Rejection Rate Distribution', fontweight='bold')
                ax1.legend()
            
            # Right plot: Rejection rate by participant
            colors = ['red' if rate > self.thresholds['epoch_rejection']['critical_rate'] 
                     else 'orange' if rate > self.thresholds['epoch_rejection']['warning_rate']
                     else 'steelblue' for rate in rejection_rates]
            
            bars = ax2.bar(range(len(rejection_rates)), rejection_rates, 
                          color=colors, alpha=0.7)
            
            # Add threshold lines
            ax2.axhline(self.thresholds['epoch_rejection']['warning_rate'], 
                       color='orange', linestyle='--', linewidth=2, alpha=0.7)
            ax2.axhline(self.thresholds['epoch_rejection']['critical_rate'], 
                       color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            ax2.set_xlabel('Participants')
            ax2.set_ylabel('Epoch Rejection Rate (%)')
            ax2.set_title('Epoch Rejection Rate by Participant', fontweight='bold')
            ax2.set_xticks(range(len(participant_ids)))
            ax2.set_xticklabels(participant_ids, rotation=45, ha='right')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def plot_ica_components_analysis(self, participants: Dict) -> str:
        """Plot ICA component analysis if ICA was used."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract ICA data
        excluded_counts = []
        fitted_counts = []
        participant_ids = []
        
        for participant_id, participant_data in participants.items():
            # Get ICA metrics
            ica_metrics = self._extract_ica_metrics(participant_data)
            if ica_metrics:
                excluded_counts.append(ica_metrics.get('n_components_excluded', 0))
                fitted_counts.append(ica_metrics.get('n_components_fitted', 0))
                participant_ids.append(participant_id)
        
        if excluded_counts:
            # Components excluded distribution
            ax1.hist(excluded_counts, bins=max(5, len(set(excluded_counts))), 
                    alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Number of ICA Components Excluded')
            ax1.set_ylabel('Number of Participants')
            ax1.set_title('ICA Components Excluded Distribution', fontweight='bold')
            
            # Components excluded by participant
            bars = ax2.bar(range(len(excluded_counts)), excluded_counts, 
                          color='steelblue', alpha=0.7)
            
            ax2.set_xlabel('Participants')
            ax2.set_ylabel('Components Excluded')
            ax2.set_title('ICA Components Excluded by Participant', fontweight='bold')
            ax2.set_xticks(range(len(participant_ids)))
            ax2.set_xticklabels(participant_ids, rotation=45, ha='right')
            
            # Add values on bars
            for i, count in enumerate(excluded_counts):
                ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _extract_bad_channel_metrics(self, participant_data: Dict) -> Dict:
        """Extract bad channel metrics from participant data."""
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'detect_bad_channels' in stages:
                metrics = stages['detect_bad_channels'].get('metrics', {})
                
                # Enhance with interpolation details if available
                if 'interpolation_details' in metrics:
                    interpolation_details = metrics['interpolation_details']
                    metrics.update({
                        'interpolation_successful': interpolation_details.get('success_rate', 0) > 0.8,
                        'bad_percentage_before': interpolation_details.get('bad_percentage_before', 0)
                    })
                
                return metrics
        return {}
    
    def _extract_epoch_metrics(self, participant_data: Dict) -> Dict:
        """Extract epoch metrics from participant data."""
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'epoch' in stages:
                return stages['epoch'].get('metrics', {})
        return {}
    
    def _extract_ica_metrics(self, participant_data: Dict) -> Dict:
        """Extract ICA metrics from participant data."""
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'blink_artifact' in stages:
                return stages['blink_artifact'].get('metrics', {})
        return {}
    
    def _get_participant_flag_level(self, participant_id: str, flagged_participants: Dict) -> str:
        """Get flag level for a specific participant."""
        for level, participants in flagged_participants.items():
            if any(p['participant_id'] == participant_id for p in participants):
                return level
        return 'good'
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"