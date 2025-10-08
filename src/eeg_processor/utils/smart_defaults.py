"""
Smart defaults and progressive disclosure system for EEG Processor configuration.

This module provides intelligent default values and progressive option disclosure
based on user choices and data characteristics.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re


class SmartDefaultsEngine:
    """Engine for providing intelligent defaults based on context."""
    
    def __init__(self):
        self.context = {}
    
    def update_context(self, key: str, value: Any):
        """Update context information for smarter defaults."""
        self.context[key] = value
    
    def get_file_extension_suggestions(self, data_dir: Optional[str] = None) -> List[Tuple[str, str, bool]]:
        """Get file extension suggestions based on directory contents or context.
        
        Returns:
            List of (extension, description, is_recommended) tuples
        """
        suggestions = [
            ('.vhdr', 'BrainVision (most common)', True),
            ('.edf', 'European Data Format', False),
            ('.fif', 'MNE-Python native format', False),
            ('.set', 'EEGLAB format', False)
        ]
        
        # If data directory is provided, try to detect formats
        if data_dir and Path(data_dir).exists():
            detected_formats = self._detect_formats_in_directory(data_dir)
            if detected_formats:
                # Update recommendations based on detected files
                for i, (ext, desc, _) in enumerate(suggestions):
                    is_detected = ext in detected_formats
                    suggestions[i] = (ext, desc, is_detected)
        
        return suggestions
    
    def get_smart_processing_defaults(self, preset_name: Optional[str] = None,
                                    data_characteristics: Optional[Dict] = None) -> Dict[str, Any]:
        """Get smart defaults for processing parameters.
        
        Args:
            preset_name: Name of selected preset
            data_characteristics: Detected data characteristics
            
        Returns:
            Dictionary of smart default parameters
        """
        defaults = {}
        
        # Filtering defaults based on context
        if data_characteristics:
            sampling_rate = data_characteristics.get('sampling_rate', 1000)
            n_channels = data_characteristics.get('n_channels', 64)
            
            # Adjust filter defaults based on sampling rate
            if sampling_rate >= 1000:
                defaults['filter'] = {
                    'l_freq': 0.1,
                    'h_freq': min(40.0, sampling_rate / 3),  # Conservative Nyquist
                    'notch': 50 if data_characteristics.get('region') == 'EU' else 60
                }
            else:
                defaults['filter'] = {
                    'l_freq': 0.1,
                    'h_freq': min(30.0, sampling_rate / 3),
                    'notch': None  # Skip notch for low sampling rates
                }
            
            # Bad channel detection defaults based on channel count
            if n_channels >= 64:
                defaults['detect_bad_channels'] = {
                    'n_neighbors': 12,
                    'threshold': 1.5
                }
            elif n_channels >= 32:
                defaults['detect_bad_channels'] = {
                    'n_neighbors': 8,
                    'threshold': 1.8
                }
            else:
                defaults['detect_bad_channels'] = {
                    'n_neighbors': 6,
                    'threshold': 2.0
                }
        
        # Epoch defaults based on experiment type
        experiment_type = self.context.get('experiment_type', 'erp')
        if experiment_type == 'erp':
            defaults['epoch'] = {
                'tmin': -0.2,
                'tmax': 0.8,
                'baseline': [-0.2, 0]
            }
        elif experiment_type == 'time_frequency':
            defaults['epoch'] = {
                'tmin': -0.5,
                'tmax': 2.0,
                'baseline': [-0.5, -0.1]
            }
        elif experiment_type == 'resting_state':
            # No epoching for resting state
            pass
        
        return defaults
    
    def get_progressive_options(self, user_level: str, section: str) -> Dict[str, bool]:
        """Get which options to show based on user experience level.
        
        Args:
            user_level: 'beginner', 'intermediate', or 'advanced'
            section: Configuration section name
            
        Returns:
            Dictionary mapping option names to whether they should be shown
        """
        if user_level == 'beginner':
            return self._get_beginner_options(section)
        elif user_level == 'intermediate':
            return self._get_intermediate_options(section)
        else:  # advanced
            return self._get_advanced_options(section)
    
    def _get_beginner_options(self, section: str) -> Dict[str, bool]:
        """Options to show for beginners."""
        options = {
            'paths': {
                'raw_data_dir': True,
                'results_dir': True,
                'file_extension': True,
                'dataset_name': False,
                'interim_dir': False,
                'figures_dir': False
            },
            'stages': {
                'filter': True,
                'detect_bad_channels': True,
                'rereference': True,
                'epoch': True,
                'remove_artifacts': False,
                'clean_rawdata_asr': False,
                'time_frequency': False
            },
            'filter': {
                'l_freq': True,
                'h_freq': True,
                'notch': True,
                'filter_length': False,
                'phase': False,
                'fir_window': False
            },
            'epoch': {
                'tmin': True,
                'tmax': True,
                'baseline': True,
                'reject': False,
                'flat': False,
                'preload': False
            }
        }
        return options.get(section, {})
    
    def _get_intermediate_options(self, section: str) -> Dict[str, bool]:
        """Options to show for intermediate users."""
        options = {
            'paths': {
                'raw_data_dir': True,
                'results_dir': True,
                'file_extension': True,
                'dataset_name': True,
                'interim_dir': False,
                'figures_dir': False
            },
            'stages': {
                'filter': True,
                'detect_bad_channels': True,
                'rereference': True,
                'epoch': True,
                'remove_artifacts': True,
                'clean_rawdata_asr': True,
                'time_frequency': True,
                'remove_blinks_emcp': True
            },
            'filter': {
                'l_freq': True,
                'h_freq': True,
                'notch': True,
                'filter_length': True,
                'phase': False,
                'fir_window': False
            },
            'epoch': {
                'tmin': True,
                'tmax': True,
                'baseline': True,
                'reject': True,
                'flat': False,
                'preload': True
            }
        }
        return options.get(section, {})
    
    def _get_advanced_options(self, section: str) -> Dict[str, bool]:
        """Options to show for advanced users (all options)."""
        # Advanced users see everything
        return {}  # Empty dict means show all options
    
    def suggest_conditions_from_markers(self, detected_markers: List[Any]) -> List[Dict[str, Any]]:
        """Suggest condition configurations based on detected event markers.
        
        Args:
            detected_markers: List of detected event markers in the data
            
        Returns:
            List of suggested condition configurations
        """
        suggestions = []
        
        # Convert all markers to strings for pattern matching
        str_markers = [str(m) for m in detected_markers]
        
        # Common ERP paradigm patterns
        patterns = [
            # Oddball paradigm
            {
                'pattern': ['1', '2'],
                'conditions': [
                    {'name': 'Standard', 'condition_markers': [1]},
                    {'name': 'Target', 'condition_markers': [2]}
                ]
            },
            # Face/house paradigm
            {
                'pattern': ['10', '20'],
                'conditions': [
                    {'name': 'Faces', 'condition_markers': [10]},
                    {'name': 'Houses', 'condition_markers': [20]}
                ]
            },
            # Go/NoGo paradigm
            {
                'pattern': ['100', '200'],
                'conditions': [
                    {'name': 'Go', 'condition_markers': [100]},
                    {'name': 'NoGo', 'condition_markers': [200]}
                ]
            }
        ]
        
        # Check for pattern matches
        for pattern_info in patterns:
            pattern_markers = pattern_info['pattern']
            if all(marker in str_markers for marker in pattern_markers):
                suggestions.extend(pattern_info['conditions'])
                break
        
        # If no patterns match, create generic conditions
        if not suggestions and detected_markers:
            # Group similar markers
            numeric_markers = [m for m in detected_markers if isinstance(m, int)]
            string_markers = [m for m in detected_markers if isinstance(m, str)]
            
            if len(numeric_markers) >= 2:
                suggestions = [
                    {'name': f'Condition{i+1}', 'condition_markers': [marker]}
                    for i, marker in enumerate(sorted(numeric_markers)[:4])  # Max 4 conditions
                ]
            elif len(string_markers) >= 2:
                suggestions = [
                    {'name': marker.replace('S', 'Condition'), 'condition_markers': [marker]}
                    for marker in sorted(string_markers)[:4]
                ]
        
        return suggestions
    
    def _detect_formats_in_directory(self, data_dir: str) -> List[str]:
        """Detect EEG file formats in a directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            return []
        
        detected_formats = set()
        
        # Check for common EEG file extensions
        format_patterns = {
            '.vhdr': r'\.vhdr$',
            '.edf': r'\.edf$',
            '.fif': r'\.fif$',
            '.set': r'\.set$',
            '.eeg': r'\.eeg$',
            '.bdf': r'\.bdf$'
        }
        
        try:
            for file_path in data_path.iterdir():
                if file_path.is_file():
                    file_name = file_path.name.lower()
                    for ext, pattern in format_patterns.items():
                        if re.search(pattern, file_name):
                            detected_formats.add(ext)
                            break
        except (PermissionError, OSError):
            # Can't access directory
            pass
        
        return list(detected_formats)
    
    def get_context_based_suggestions(self, context_key: str, **kwargs) -> List[str]:
        """Get suggestions based on current context."""
        if context_key == 'experiment_types':
            return [
                'ERP (Event-Related Potentials)',
                'Time-frequency analysis',
                'Resting state',
                'Connectivity analysis',
                'Source localization prep'
            ]
        
        elif context_key == 'participant_populations':
            return [
                'Healthy adults',
                'Clinical population',
                'Pediatric (children)',
                'Elderly participants',
                'Mixed population'
            ]
        
        elif context_key == 'recording_environments':
            return [
                'Controlled lab environment',
                'Clinical setting',
                'Field/mobile recording',
                'High-noise environment'
            ]
        
        return []
    
    def adjust_defaults_for_population(self, population: str) -> Dict[str, Any]:
        """Adjust processing defaults based on participant population."""
        adjustments = {}
        
        if 'pediatric' in population.lower() or 'children' in population.lower():
            # More conservative settings for children
            adjustments['detect_bad_channels'] = {'threshold': 2.0}
            adjustments['filter'] = {'l_freq': 0.5}  # Higher high-pass for movement
            adjustments['epoch'] = {'reject': {'eeg': 150e-6}}  # More lenient rejection
            
        elif 'clinical' in population.lower():
            # More aggressive artifact removal for clinical data
            adjustments['detect_bad_channels'] = {'threshold': 1.2}
            adjustments['filter'] = {'l_freq': 1.0}
            adjustments['clean_rawdata_asr'] = {'cutoff': 15}
            
        elif 'elderly' in population.lower():
            # Account for potential increased artifacts
            adjustments['detect_bad_channels'] = {'threshold': 1.8}
            adjustments['filter'] = {'l_freq': 0.5}
            
        return adjustments
    
    def validate_parameter_compatibility(self, config: Dict[str, Any]) -> List[str]:
        """Validate parameter compatibility and suggest fixes."""
        warnings = []
        
        # Check filter parameters
        if 'stages' in config:
            filter_params = None
            for stage in config['stages']:
                if isinstance(stage, dict) and 'filter' in stage:
                    filter_params = stage['filter']
                    break
            
            if filter_params:
                l_freq = filter_params.get('l_freq', 0)
                h_freq = filter_params.get('h_freq', 1000)
                
                if l_freq and h_freq and l_freq >= h_freq:
                    warnings.append("High-pass frequency should be lower than low-pass frequency")
                
                if h_freq and h_freq > 200:
                    warnings.append("Very high low-pass frequency may include excessive noise")
        
        # Check epoch parameters
        epoch_params = None
        for stage in config.get('stages', []):
            if isinstance(stage, dict) and 'epoch' in stage:
                epoch_params = stage['epoch']
                break
        
        if epoch_params:
            tmin = epoch_params.get('tmin', 0)
            tmax = epoch_params.get('tmax', 1)
            
            if tmin >= tmax:
                warnings.append("Epoch start time should be before end time")
            
            baseline = epoch_params.get('baseline')
            if baseline and len(baseline) == 2:
                if baseline[0] < tmin or baseline[1] > tmax:
                    warnings.append("Baseline period extends outside epoch window")
        
        return warnings