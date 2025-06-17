"""
Pipeline Detection Module

Analyzes quality metrics to determine which processing stages were used
and configures adaptive reporting accordingly.
"""

from typing import Dict, Set, List
from loguru import logger


class PipelineDetector:
    """
    Detects which processing stages were used in the EEG pipeline
    by analyzing the quality metrics data.
    """
    
    def __init__(self, participants_data: Dict):
        """
        Initialize detector with participants data.
        
        Args:
            participants_data: Dictionary of participant metrics from quality tracker
        """
        self.participants_data = participants_data
        self.pipeline_info = self._detect_pipeline_stages()
        
    def _detect_pipeline_stages(self) -> Dict:
        """
        Analyze all participants to detect which stages were used.
        
        Returns:
            Dictionary with boolean flags for each potential stage type
        """
        all_stages = set()
        
        # Collect all stage names used across all participants/conditions
        for participant_data in self.participants_data.values():
            for condition_data in participant_data['conditions'].values():
                all_stages.update(condition_data['stages'].keys())
        
        logger.info(f"Detected pipeline stages: {sorted(all_stages)}")
        
        # Map stages to their functional categories
        pipeline_info = {
            'stages_used': sorted(all_stages),
            'has_bad_channels': 'detect_bad_channels' in all_stages,
            'has_epoching': 'epoch' in all_stages,
            'has_ica': 'blink_artifact' in all_stages,
            'has_filtering': 'filter' in all_stages,
            'has_asr': 'clean_rawdata_asr' in all_stages,
            'has_cropping': 'crop' in all_stages,
            'has_segmentation': 'segment_condition' in all_stages,
            'has_rereferencing': 'rereference' in all_stages,
        }
        
        # Determine pipeline type
        if pipeline_info['has_epoching']:
            pipeline_info['data_type'] = 'epoched'
        else:
            pipeline_info['data_type'] = 'continuous'
            
        # Log pipeline characteristics
        logger.info(f"Pipeline type: {pipeline_info['data_type']}")
        logger.info(f"Key stages: epoching={pipeline_info['has_epoching']}, "
                   f"ICA={pipeline_info['has_ica']}, bad_channels={pipeline_info['has_bad_channels']}")
        
        return pipeline_info
    
    def get_required_plots(self) -> List[str]:
        """
        Determine which plots are relevant for this pipeline.
        
        Returns:
            List of plot names that should be generated
        """
        required_plots = [
            'completion_matrix',  # Always show stage completion status
            'processing_overview'  # Always show overall processing status
        ]
        
        # Add stage-specific plots only if stages were used
        if self.pipeline_info['has_bad_channels']:
            required_plots.append('bad_channels_analysis')
            
        if self.pipeline_info['has_epoching']:
            required_plots.append('epoch_rejection_analysis')
            
        if self.pipeline_info['has_ica']:
            required_plots.append('ica_analysis')
            
        if self.pipeline_info['has_asr']:
            required_plots.append('asr_analysis')
            
        logger.info(f"Required plots for this pipeline: {required_plots}")
        return required_plots
    
    def get_quality_thresholds(self) -> Dict:
        """
        Get appropriate quality thresholds based on pipeline type.
        
        Returns:
            Dictionary of quality thresholds for flagging participants
        """
        thresholds = {
            'bad_channels': {
                'warning_percentage': 10.0,  # >10% of channels bad
                'critical_percentage': 25.0,  # >25% of channels bad
                'max_reasonable': 8  # >8 bad channels for typical 32-ch system
            }
        }
        
        # Add epoch-specific thresholds if epoching is used
        if self.pipeline_info['has_epoching']:
            thresholds['epoch_rejection'] = {
                'warning_rate': 20.0,  # >20% rejection rate
                'critical_rate': 40.0   # >40% rejection rate
            }
        
        return thresholds
    
    def get_pipeline_summary(self) -> Dict:
        """
        Get a summary of the detected pipeline for reporting.
        
        Returns:
            Dictionary with pipeline summary information
        """
        stage_count = len(self.pipeline_info['stages_used'])
        
        return {
            'total_stages': stage_count,
            'data_type': self.pipeline_info['data_type'],
            'key_features': self._get_key_features(),
            'processing_flow': self._get_processing_flow()
        }
    
    def _get_key_features(self) -> List[str]:
        """Get list of key pipeline features for display."""
        features = []
        
        if self.pipeline_info['has_epoching']:
            features.append("Event-related analysis (epoched)")
        else:
            features.append("Continuous data analysis")
            
        if self.pipeline_info['has_ica']:
            features.append("ICA artifact removal")
            
        if self.pipeline_info['has_asr']:
            features.append("ASR artifact correction")
            
        if self.pipeline_info['has_bad_channels']:
            features.append("Bad channel detection & interpolation")
            
        return features
    
    def _get_processing_flow(self) -> List[str]:
        """Get ordered list of processing stages for display."""
        # Common stage order (stages not present will be filtered out)
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
        
        # Return only stages that were actually used, in standard order
        used_stages = self.pipeline_info['stages_used']
        return [stage for stage in standard_order if stage in used_stages]