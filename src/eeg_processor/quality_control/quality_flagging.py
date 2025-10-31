"""
Quality Flagging Module

Identifies participants with data quality issues based on EEG research standards.
Focuses on actual data quality problems, not methodological choices.
"""

from typing import Dict, List, Tuple
from loguru import logger


class QualityFlagger:
    """
    Flags participants based on actual EEG data quality issues.
    
    Designed for pipelines where all stages are required - any stage failure
    is a critical issue. Focuses on data quality metrics that indicate
    problems with the recorded EEG data itself.
    """
    
    def __init__(self, pipeline_info: Dict, quality_thresholds: Dict):
        """
        Initialize quality flagger.
        
        Args:
            pipeline_info: Information about which stages were used
            quality_thresholds: Threshold values for quality metrics
        """
        self.pipeline_info = pipeline_info
        self.thresholds = quality_thresholds
        
    def flag_all_participants(self, participants_data: Dict) -> Dict:
        """
        Flag all participants based on data quality.
        
        Args:
            participants_data: Dictionary of all participant metrics
            
        Returns:
            Dictionary with participants organized by flag level
        """
        flagged_participants = {
            'critical': [],
            'warning': [], 
            'good': []
        }
        
        for participant_id, participant_data in participants_data.items():
            flag_result = self.flag_participant(participant_id, participant_data)
            flag_level = flag_result['flag_level']
            flagged_participants[flag_level].append(flag_result)
            
        # Log summary
        logger.info(f"Quality flagging complete: "
                   f"{len(flagged_participants['critical'])} critical, "
                   f"{len(flagged_participants['warning'])} warning, "
                   f"{len(flagged_participants['good'])} good")
        
        return flagged_participants
    
    def flag_participant(self, participant_id: str, participant_data: Dict) -> Dict:
        """
        Flag a single participant based on data quality.
        
        Args:
            participant_id: ID of the participant
            participant_data: Participant's quality metrics
            
        Returns:
            Dictionary with flag level and detailed reasons
        """
        flags = []
        flag_level = 'good'
        
        # Check for processing failures (critical since all stages are required)
        processing_flags = self._check_processing_completion(participant_data)
        if processing_flags:
            flags.extend(processing_flags)
            flag_level = 'critical'
            
        # Check bad channel issues (if stage was used)
        if self.pipeline_info['has_bad_channels']:
            bad_channel_flags, bad_channel_level = self._check_bad_channels(participant_data)
            if bad_channel_flags:
                flags.extend(bad_channel_flags)
                flag_level = self._escalate_flag_level(flag_level, bad_channel_level)
                
        # Check epoch rejection issues (if epoching was used)
        if self.pipeline_info['has_epoching']:
            epoch_flags, epoch_level = self._check_epoch_rejection(participant_data)
            if epoch_flags:
                flags.extend(epoch_flags)
                flag_level = self._escalate_flag_level(flag_level, epoch_level)
        
        # Check ICA issues (if ICA was used)
        if self.pipeline_info['has_ica']:
            ica_flags, ica_level = self._check_ica_components(participant_data)
            if ica_flags:
                flags.extend(ica_flags)
                flag_level = self._escalate_flag_level(flag_level, ica_level)
        
        # Check EMCP issues (if EMCP was used)
        if self.pipeline_info['has_emcp']:
            emcp_flags, emcp_level = self._check_emcp_correction(participant_data)
            if emcp_flags:
                flags.extend(emcp_flags)
                flag_level = self._escalate_flag_level(flag_level, emcp_level)

        # Check regression artifact removal issues (if regression was used)
        if self.pipeline_info.get('has_regression', False):
            regression_flags, regression_level = self._check_regression_correction(participant_data)
            if regression_flags:
                flags.extend(regression_flags)
                flag_level = self._escalate_flag_level(flag_level, regression_level)

        return {
            'participant_id': participant_id,
            'flag_level': flag_level,
            'reasons': flags,
            'participant_data': participant_data
        }
    
    def _check_processing_completion(self, participant_data: Dict) -> List[str]:
        """Check for processing failures (critical in required-stage pipeline)."""
        failures = []
        
        for condition_name, condition_data in participant_data['conditions'].items():
            # Check overall condition completion
            if not condition_data['completion']['success']:
                error_msg = condition_data['completion'].get('error', 'Unknown error')
                failures.append(f"Processing failed: {condition_name} ({error_msg})")
            
            # Check for stage-level failures
            stages = condition_data['stages']
            for stage_name, stage_data in stages.items():
                stage_metrics = stage_data.get('metrics', {})
                
                # Look for error indicators in stage metrics
                if 'error' in stage_metrics:
                    failures.append(f"Stage failed: {stage_name} in {condition_name}")
                elif not stage_metrics.get('stage_completed', True):
                    failures.append(f"Stage incomplete: {stage_name} in {condition_name}")
                    
        return failures
    
    def _check_bad_channels(self, participant_data: Dict) -> Tuple[List[str], str]:
        """Check for bad channel quality issues."""
        flags = []
        flag_level = 'good'
        
        # Extract bad channel metrics from any condition
        bad_channel_metrics = self._get_bad_channel_metrics(participant_data)
        
        if bad_channel_metrics:
            n_detected = bad_channel_metrics.get('n_detected', 0)
            interpolation_details = bad_channel_metrics.get('interpolation_details', {})
            
            # Use the actual bad percentage calculated during interpolation
            bad_percentage = interpolation_details.get('bad_percentage_before', 0)
            
            # If no interpolation details, fallback to a simple calculation
            if bad_percentage == 0 and n_detected > 0:
                # Estimate total EEG channels (excluding EOG, etc.)
                total_eeg_channels = 32  # Default assumption for typical systems
                bad_percentage = (n_detected / total_eeg_channels) * 100
            
            # Check if interpolation failed: final_bads should equal detected_bads if interpolation failed
            detected_bads = set(bad_channel_metrics.get('detected_bads', []))
            final_bads = set(bad_channel_metrics.get('final_bads', []))
            interpolation_attempted = bad_channel_metrics.get('interpolation_attempted', False)
            
            # Interpolation failed if it was attempted but final_bads == detected_bads
            interpolation_failed = interpolation_attempted and (final_bads == detected_bads) and len(detected_bads) > 0
            interpolation_reliable = interpolation_details.get('interpolation_reliable', True)
            
            # Critical issues
            if interpolation_failed:
                flags.append("Bad channel interpolation failed - no channels successfully interpolated")
                flag_level = 'critical'
            elif not interpolation_reliable:
                flags.append("Bad channel interpolation unreliable")
                flag_level = 'critical'
            elif bad_percentage >= self.thresholds['bad_channels']['critical_percentage']:
                flags.append(f"Severe channel noise: {bad_percentage:.1f}% bad channels")
                flag_level = 'critical'
            
            # Warning issues  
            elif bad_percentage >= self.thresholds['bad_channels']['warning_percentage']:
                flags.append(f"Moderate channel noise: {bad_percentage:.1f}% bad channels")
                flag_level = 'warning'
                
        return flags, flag_level
    
    def _check_epoch_rejection(self, participant_data: Dict) -> Tuple[List[str], str]:
        """Check for epoch rejection quality issues."""
        flags = []
        flag_level = 'good'
        
        # Extract epoch metrics from any condition
        epoch_metrics = self._get_epoch_metrics(participant_data)
        
        if epoch_metrics:
            rejection_rate = epoch_metrics.get('rejection_rate', 0)
            
            # Critical rejection rate
            if rejection_rate > self.thresholds['epoch_rejection']['critical_rate']:
                flags.append(f"Extreme epoch rejection: {rejection_rate:.1f}%")
                flag_level = 'critical'
            
            # Warning rejection rate
            elif rejection_rate > self.thresholds['epoch_rejection']['warning_rate']:
                flags.append(f"High epoch rejection: {rejection_rate:.1f}%")
                flag_level = 'warning'
                
        return flags, flag_level
    
    def _check_ica_components(self, participant_data: Dict) -> Tuple[List[str], str]:
        """Check for ICA component issues."""
        flags = []
        flag_level = 'good'
        
        # Extract ICA metrics from any condition
        ica_metrics = self._get_ica_metrics(participant_data)
        
        if ica_metrics:
            n_components_excluded = ica_metrics.get('n_components_excluded', 0)
            
            # Warning if no components were removed (suggests no artifacts found, which is unusual)
            if n_components_excluded == 0:
                flags.append("No ICA components removed (unusual - no artifacts detected)")
                flag_level = 'warning'
                
        return flags, flag_level
    
    def _get_bad_channel_metrics(self, participant_data: Dict) -> Dict:
        """Extract bad channel metrics from participant data."""
        # Look for bad channel metrics in any condition
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'detect_bad_channels' in stages:
                metrics = stages['detect_bad_channels'].get('metrics', {})
                
                # Also check interpolation details if available
                if 'interpolation_details' in metrics:
                    interpolation_details = metrics['interpolation_details']
                    metrics.update({
                        'interpolation_successful': interpolation_details.get('success_rate', 0) > 0.8,
                        'interpolation_reliable': interpolation_details.get('interpolation_reliable', True),
                        'bad_percentage_before': interpolation_details.get('bad_percentage_before', 0)
                    })
                
                return metrics
        return {}
    
    def _get_epoch_metrics(self, participant_data: Dict) -> Dict:
        """Extract epoch rejection metrics from participant data."""
        # Look for epoch metrics in any condition
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'epoch' in stages:
                return stages['epoch'].get('metrics', {})
        return {}
    
    def _get_ica_metrics(self, participant_data: Dict) -> Dict:
        """Extract ICA metrics from participant data."""
        # Look for ICA metrics in any condition
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'blink_artifact' in stages:
                return stages['blink_artifact'].get('metrics', {})
        return {}
    
    def _check_emcp_correction(self, participant_data: Dict) -> Tuple[List[str], str]:
        """Check for EMCP blink correction issues."""
        flags = []
        flag_level = 'good'
        
        # Extract EMCP metrics from any condition
        emcp_metrics = self._get_emcp_metrics(participant_data)
        
        if emcp_metrics:
            # Check for essential quality flags that indicate real problems
            quality_flags = emcp_metrics.get('quality_flags', {})
            
            # Critical: No blinks detected when EMCP was used (suggests method failure)
            if quality_flags.get('no_blinks_detected', False):
                flags.append("EMCP found no blinks to correct (possible EOG channel issue)")
                flag_level = 'critical'
            
            # Critical: Very low correlation suggests severe overcorrection
            mean_correlation = emcp_metrics.get('mean_correlation', 1.0)
            if mean_correlation < 0.7:
                flags.append(f"EMCP severe overcorrection (correlation: {mean_correlation:.2f})")
                flag_level = 'critical'
            
            # Warning: Low correlation suggests possible overcorrection
            elif mean_correlation < 0.8:
                flags.append(f"EMCP possible overcorrection (correlation: {mean_correlation:.2f})")
                flag_level = 'warning'
            
            # Warning: Extreme regression coefficients (Gratton & Coles only)
            if quality_flags.get('extreme_coefficients', False):
                max_coeff = emcp_metrics.get('max_regression_coefficient', 0)
                flags.append(f"EMCP extreme regression coefficients (max: {max_coeff:.3f})")
                if flag_level == 'good':
                    flag_level = 'warning'
                
        return flags, flag_level
    
    def _get_emcp_metrics(self, participant_data: Dict) -> Dict:
        """Extract EMCP metrics from participant data."""
        # Look for EMCP metrics in any condition
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'remove_blinks_emcp' in stages:
                return stages['remove_blinks_emcp'].get('metrics', {})
        return {}

    def _check_regression_correction(self, participant_data: Dict) -> Tuple[List[str], str]:
        """Check for regression-based artifact removal issues."""
        flags = []
        flag_level = 'good'

        # Extract regression metrics from any condition
        regression_metrics = self._get_regression_metrics(participant_data)

        if regression_metrics:
            # Check quality flags from the regression metrics
            quality_flags = regression_metrics.get('quality_flags', {})

            # Critical: Very low correlation suggests severe overcorrection
            if quality_flags.get('low_correlation', False):
                mean_corr = regression_metrics.get('artifact_reduction', {}).get('mean_correlation_preserved', 0)
                flags.append(f"Regression low correlation (signal preservation: {mean_corr:.2f})")
                flag_level = 'critical'

            # Warning: Extreme regression coefficients
            if quality_flags.get('extreme_coefficients', False):
                max_coeff = regression_metrics.get('regression_coefficients', {}).get('max_coeff', 0)
                flags.append(f"Regression extreme coefficients (max: {max_coeff:.3f})")
                if flag_level == 'good':
                    flag_level = 'warning'

            # Info: Minimal correction applied (might indicate no artifacts present)
            if quality_flags.get('minimal_correction', False):
                flags.append("Regression minimal correction applied (check if artifacts were present)")
                # Don't escalate flag level - this is informational

        return flags, flag_level

    def _get_regression_metrics(self, participant_data: Dict) -> Dict:
        """Extract regression artifact removal metrics from participant data."""
        # Look for regression metrics in any condition
        for condition_data in participant_data['conditions'].values():
            stages = condition_data.get('stages', {})
            if 'remove_artifacts' in stages:
                metrics = stages['remove_artifacts'].get('metrics', {})
                # Check if this is regression method (not ICA)
                if metrics.get('method') == 'regression':
                    return metrics
        return {}

    def _escalate_flag_level(self, current_level: str, new_level: str) -> str:
        """Escalate flag level to the more severe one."""
        priority = {'good': 0, 'warning': 1, 'critical': 2}
        
        current_priority = priority.get(current_level, 0)
        new_priority = priority.get(new_level, 0)
        
        if new_priority > current_priority:
            return new_level
        return current_level
    
    def get_quality_summary(self, flagged_participants: Dict) -> Dict:
        """Generate summary statistics of quality flags."""
        total = sum(len(participants) for participants in flagged_participants.values())
        
        return {
            'total_participants': total,
            'critical_count': len(flagged_participants['critical']),
            'warning_count': len(flagged_participants['warning']),
            'good_count': len(flagged_participants['good']),
            'critical_percentage': (len(flagged_participants['critical']) / total * 100) if total > 0 else 0,
            'warning_percentage': (len(flagged_participants['warning']) / total * 100) if total > 0 else 0,
            'good_percentage': (len(flagged_participants['good']) / total * 100) if total > 0 else 0
        }