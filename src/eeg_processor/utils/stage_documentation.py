"""
Stage documentation system for EEG Processor.

This module provides utilities for extracting and formatting docstring information
from processing stages, organizing them by category, and providing comprehensive
help for CLI users.
"""

import inspect
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import importlib
from loguru import logger


@dataclass
class StageInfo:
    """Information about a processing stage."""
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    returns: str
    notes: List[str]
    examples: List[str]
    function_source: str
    module_path: str


class StageDocumentationExtractor:
    """Extract comprehensive documentation from processing stages."""
    
    def __init__(self):
        self.stage_categories = {
            # Data handling stages
            "crop": "data_handling",
            "adjust_events": "data_handling", 
            "correct_triggers": "data_handling",
            
            # Pre-processing stages
            "filter": "preprocessing",
            "compute_eog": "preprocessing",
            "detect_bad_channels": "preprocessing",
            "rereference": "preprocessing", 
            "remove_artifacts": "preprocessing",
            "remove_blinks_emcp": "preprocessing",
            "clean_rawdata_asr": "preprocessing",
            "blink_artifact": "preprocessing",  # Legacy alias
            
            # Condition handling stages
            "segment_condition": "condition_handling",
            "epoch": "condition_handling",
            
            # Post-epoching analysis
            "time_frequency": "post_epoching",
            "time_frequency_raw": "post_epoching",
            "time_frequency_average": "post_epoching",

            # Experimental/advanced analysis
            "custom_cwt": "experimental",

            # Visualization and other
            "view": "visualization"
        }

        self.category_descriptions = {
            "data_handling": "Data preparation and event management",
            "preprocessing": "Signal filtering and artifact removal",
            "condition_handling": "Experimental condition processing and epoching",
            "post_epoching": "Analysis of epoched data",
            "experimental": "Experimental and advanced analysis methods",
            "visualization": "Data visualization and inspection"
        }
    
    def get_all_stages(self) -> Dict[str, List[str]]:
        """Get all stages organized by category.
        
        Returns:
            Dictionary mapping category names to lists of stage names
        """
        categorized_stages = {}
        for stage, category in self.stage_categories.items():
            if category not in categorized_stages:
                categorized_stages[category] = []
            categorized_stages[category].append(stage)
        
        return categorized_stages
    
    def get_stage_info(self, stage_name: str) -> Optional[StageInfo]:
        """Extract comprehensive information about a specific stage.
        
        Args:
            stage_name: Name of the processing stage
            
        Returns:
            StageInfo object with extracted documentation, or None if stage not found
        """
        if stage_name not in self.stage_categories:
            return None
        
        try:
            # Get the actual implementation function
            from ..state_management.data_processor import DataProcessor
            processor = DataProcessor()
            
            if stage_name not in processor.stage_registry:
                return None
                
            stage_func = processor.stage_registry[stage_name]
            
            # Get the target function (handle method wrappers)
            target_func = self._get_target_function(stage_name, stage_func)
            if not target_func:
                return None
            
            # Extract documentation
            docstring = inspect.getdoc(target_func) or ""
            parameters = self._extract_parameters(target_func)
            description, returns, notes, examples = self._parse_docstring(docstring)
            
            # Add practical usage examples if not found in docstring
            if not examples:
                examples = self._get_usage_examples(stage_name)
            
            return StageInfo(
                name=stage_name,
                category=self.stage_categories[stage_name],
                description=description,
                parameters=parameters,
                returns=returns,
                notes=notes,
                examples=examples,
                function_source=target_func.__module__ if hasattr(target_func, '__module__') else "",
                module_path=self._get_module_path(stage_name)
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract documentation for stage '{stage_name}': {e}")
            return None
    
    def _get_target_function(self, stage_name: str, stage_func) -> Optional[callable]:
        """Get the actual target function for documentation extraction."""
        try:
            # Map stage names to their implementation modules and functions
            stage_implementations = {
                "crop": ("..utils.raw_data_tools", "crop_data"),
                "adjust_events": ("..utils.raw_data_tools", "adjust_event_times"),
                "correct_triggers": ("..utils.correct_triggers", "correct_triggers"),
                "filter": ("..processing.filtering", "filter_data"),
                "compute_eog": ("..processing.montages", "compute_eog_channels"),
                "detect_bad_channels": ("..processing.bad_channels", "detect_bad_channels"),
                "rereference": ("..processing.rereferencing", "set_reference"),
                "remove_artifacts": ("..processing.ica", "remove_artifacts_ica"),
                "remove_blinks_emcp": ("..processing.emcp", "remove_blinks_eog_regression"),
                "clean_rawdata_asr": ("..processing.artifact", "clean_rawdata_asr"),
                "segment_condition": ("..processing.segmentation", "segment_raw_by_conditions"),
                "epoch": ("..processing.epoching", "create_epochs"),
                "time_frequency": ("..processing.time_frequency", "compute_epochs_tfr_average"),
                "time_frequency_raw": ("..processing.time_frequency", "compute_baseline_spectrum"),
                "time_frequency_average": ("..processing.time_frequency", "compute_raw_tfr_average"),
                "view": ("..processing.visualization", "plot_stage")
            }
            
            if stage_name not in stage_implementations:
                return None
                
            module_name, func_name = stage_implementations[stage_name]
            
            # Import the actual implementation function
            try:
                # Convert relative import to absolute
                if module_name.startswith(".."):
                    base_module = "eeg_processor"
                    relative_parts = module_name[2:].split(".")
                    absolute_module = base_module + "." + ".".join(relative_parts)
                else:
                    absolute_module = module_name
                
                module = importlib.import_module(absolute_module)
                return getattr(module, func_name, None)
                
            except (ImportError, AttributeError):
                # Fallback: try to get documentation from the wrapper function
                return stage_func
                
        except Exception as e:
            logger.debug(f"Could not get target function for {stage_name}: {e}")
            return stage_func
    
    def _get_module_path(self, stage_name: str) -> str:
        """Get the module path for a stage."""
        module_map = {
            "crop": "utils/raw_data_tools.py",
            "adjust_events": "utils/raw_data_tools.py", 
            "correct_triggers": "utils/correct_triggers.py",
            "filter": "processing/filtering.py",
            "compute_eog": "processing/montages.py",
            "detect_bad_channels": "processing/bad_channels.py",
            "rereference": "processing/rereferencing.py",
            "remove_artifacts": "processing/ica.py",
            "remove_blinks_emcp": "processing/emcp.py",
            "clean_rawdata_asr": "processing/artifact.py",
            "segment_condition": "processing/segmentation.py",
            "epoch": "processing/epoching.py",
            "time_frequency": "processing/time_frequency.py",
            "time_frequency_raw": "processing/time_frequency.py",
            "time_frequency_average": "processing/time_frequency.py",
            "view": "processing/visualization.py"
        }
        return module_map.get(stage_name, "state_management/data_processor.py")
    
    def _extract_parameters(self, func) -> Dict[str, Any]:
        """Extract parameter information from function signature."""
        try:
            sig = inspect.signature(func)
            parameters = {}
            
            for name, param in sig.parameters.items():
                if name in ['self', 'data', 'inplace']:  # Skip common parameters
                    continue
                    
                param_info = {
                    'name': name,
                    'type': self._get_type_string(param.annotation),
                    'default': self._get_default_string(param.default),
                    'required': param.default == inspect.Parameter.empty
                }
                parameters[name] = param_info
                
            return parameters
            
        except Exception as e:
            logger.debug(f"Could not extract parameters: {e}")
            return {}
    
    def _get_type_string(self, annotation) -> str:
        """Convert type annotation to readable string."""
        if annotation == inspect.Parameter.empty:
            return "Any"
        
        # Handle typing module annotations first (before __name__ check)
        annotation_str = str(annotation)
        
        # Check if this is a typing annotation
        if 'typing.' in annotation_str or annotation_str.startswith(('Optional[', 'Union[')):
            # Process typing annotations
            pass  # Continue to typing processing below
        elif hasattr(annotation, '__name__'):
            # Handle simple class types
            return annotation.__name__
        
        # Clean up module names
        annotation_str = annotation_str.replace('typing.', '')
        annotation_str = annotation_str.replace('mne.io.base.', '')
        annotation_str = annotation_str.replace('mne.', '')
        annotation_str = annotation_str.replace('NoneType', 'None')
        annotation_str = annotation_str.replace("<class '", '').replace("'>", '')
        
        # Handle specific complex types first
        if 'BaseRaw' in annotation_str:
            return 'BaseRaw'
        elif 'Epochs' in annotation_str:
            return 'Epochs'
        
        # Handle Optional types - these are Union[T, None]
        if annotation_str.startswith('Optional['):
            inner_type = annotation_str[9:-1]  # Remove Optional[ and ]
            return f"{self._simplify_type(inner_type)} | None"
        
        # Handle Union types
        if annotation_str.startswith('Union['):
            inner_types = annotation_str[6:-1]  # Remove Union[ and ]
            # Split by comma and clean up
            types = [t.strip() for t in inner_types.split(',')]
            simplified_types = [self._simplify_type(t) for t in types]
            return ' | '.join(simplified_types)
        
        return self._simplify_type(annotation_str)
    
    def _simplify_type(self, type_str: str) -> str:
        """Simplify a single type string."""
        type_str = type_str.strip()
        
        # Handle specific types
        if 'dict' in type_str.lower():
            return 'dict'
        elif 'list' in type_str.lower():
            return 'list'
        elif 'bool' in type_str.lower():
            return 'bool'
        elif 'float' in type_str.lower():
            return 'float'
        elif 'int' in type_str.lower():
            return 'int'
        elif 'str' in type_str.lower():
            return 'str'
        elif type_str == 'None':
            return 'None'
        
        return type_str
    
    def _get_default_string(self, default) -> str:
        """Convert default value to readable string."""
        if default == inspect.Parameter.empty:
            return "Required"
        
        if default is None:
            return "None"
        
        if isinstance(default, str):
            return f'"{default}"'
        
        return str(default)
    
    def _parse_docstring(self, docstring: str) -> Tuple[str, str, List[str], List[str]]:
        """Parse docstring into components.
        
        Returns:
            Tuple of (description, returns, notes, examples)
        """
        if not docstring:
            return "No description available", "", [], []
        
        lines = docstring.strip().split('\n')
        description = ""
        returns = ""
        notes = []
        examples = []
        
        current_section = "description"
        
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                current_section = "args"
                continue
            elif line.lower().startswith(('returns:', 'return:')):
                current_section = "returns"
                continue
            elif line.lower().startswith(('notes:', 'note:')):
                current_section = "notes"
                continue
            elif line.lower().startswith(('examples:', 'example:')):
                current_section = "examples"
                continue
            elif line.lower().startswith(('raises:', 'raise:')):
                current_section = "raises"
                continue
            
            if current_section == "description" and line:
                if not description:
                    description = line
                else:
                    description += " " + line
            elif current_section == "returns" and line:
                if not returns:
                    returns = line
                else:
                    returns += " " + line
            elif current_section == "notes" and line:
                if line.startswith('-') or line.startswith('*'):
                    notes.append(line[1:].strip())
                elif line:
                    notes.append(line)
            elif current_section == "examples" and line:
                examples.append(line)
        
        return description or "No description available", returns, notes, examples
    
    def _get_usage_examples(self, stage_name: str) -> List[str]:
        """Get practical usage examples for a stage."""
        examples = {
            "filter": [
                "# Basic bandpass filtering",
                "- filter: {l_freq: 0.1, h_freq: 40}",
                "",
                "# High-pass only for baseline correction",
                "- filter: {l_freq: 1.0, h_freq: null}",
                "",
                "# Notch filter for line noise",
                "- filter: {l_freq: 0.1, h_freq: 40, notch: 50}"
            ],
            "detect_bad_channels": [
                "# Basic bad channel detection with interpolation",
                "- detect_bad_channels: {interpolate: true}",
                "",
                "# Conservative detection (fewer channels marked bad)",
                "- detect_bad_channels: {threshold: 2.0, interpolate: true}",
                "",
                "# Detection only, no interpolation",
                "- detect_bad_channels: {interpolate: false, show_plot: true}"
            ],
            "rereference": [
                "# Average reference (most common)",
                "- rereference: {method: 'average'}",
                "",
                "# Mastoid reference",
                "- rereference: {method: 'single', ref_channel: 'M1'}",
                "",
                "# Exclude bad channels from reference",
                "- rereference: {method: 'average', exclude: ['Fp1', 'Fp2']}"
            ],
            "remove_artifacts": [
                "# ICA artifact removal (recommended)",
                "- remove_artifacts: {method: 'ica'}",
                "",
                "# ICA with specific number of components",
                "- remove_artifacts: {method: 'ica', n_components: 0.9}",
                "",
                "# Show ICA component plots for review",
                "- remove_artifacts: {method: 'ica', show_plots: true}"
            ],
            "clean_rawdata_asr": [
                "# Conservative ASR (recommended)",
                "- clean_rawdata_asr: {cutoff: 20}",
                "",
                "# More aggressive artifact removal",
                "- clean_rawdata_asr: {cutoff: 10, method: 'euclid'}",
                "",
                "# ASR with visualization",
                "- clean_rawdata_asr: {cutoff: 20, show_plot: true}"
            ],
            "remove_blinks_emcp": [
                "# EOG regression method (standard)",
                "- remove_blinks_emcp: {method: 'eog_regression', eog_channels: ['HEOG', 'VEOG']}",
                "",
                "# Gratton & Coles method (reference-agnostic)",
                "- remove_blinks_emcp: {method: 'gratton_coles', eog_channels: ['HEOG', 'VEOG']}",
                "",
                "# With evoked response subtraction",
                "- remove_blinks_emcp: {method: 'gratton_coles', subtract_evoked: true}"
            ],
            "epoch": [
                "# Basic ERP epoching",
                "- epoch: {tmin: -0.2, tmax: 0.8, baseline: [-0.2, 0]}",
                "",
                "# Long epochs for time-frequency analysis",
                "- epoch: {tmin: -1.0, tmax: 2.0, baseline: [-1.0, -0.5]}",
                "",
                "# No baseline correction",
                "- epoch: {tmin: -0.2, tmax: 0.8, baseline: null}"
            ],
            "time_frequency": [
                "# Basic time-frequency analysis",
                "- time_frequency: {freq_range: [1, 50], n_freqs: 50}",
                "",
                "# High-resolution frequency analysis",
                "- time_frequency: {freq_range: [4, 100], n_freqs: 96, method: 'morlet'}",
                "",
                "# Include inter-trial coherence",
                "- time_frequency: {freq_range: [1, 50], compute_itc: true}"
            ],
            "crop": [
                "# Crop to specific time window",
                "- crop: {tmin: 10.0, tmax: 600.0}",
                "",
                "# Crop using event markers",
                "- crop: {event_start: 'start_recording', event_end: 'end_recording'}",
                "",
                "# Crop from beginning to specific time",
                "- crop: {tmax: 300.0}"
            ],
            "compute_eog": [
                "# Compute bipolar EOG channels",
                "- compute_eog: {heog_pair: ['F7', 'F8'], veog_pair: ['Fp1', 'F3']}",
                "",
                "# Standard 10-20 EOG computation",
                "- compute_eog: {heog_pair: ['F7', 'F8'], veog_pair: ['Fp1', 'Fp2']}"
            ]
        }
        
        return examples.get(stage_name, [f"# Configuration example for {stage_name}", f"- {stage_name}: {{}}"])


def format_stage_help(stage_info: StageInfo, include_examples: bool = False) -> str:
    """Format stage information for CLI display.
    
    Args:
        stage_info: StageInfo object with stage documentation
        include_examples: Whether to include usage examples
        
    Returns:
        Formatted help string
    """
    output = []
    
    # Header
    output.append(f"Stage: {stage_info.name}")
    output.append(f"Category: {stage_info.category}")
    output.append("=" * 50)
    
    # Description
    output.append(f"\nDescription:")
    output.append(f"  {stage_info.description}")
    
    # Parameters
    if stage_info.parameters:
        output.append(f"\nParameters:")
        for param_name, param_info in stage_info.parameters.items():
            required_str = "(required)" if param_info['required'] else f"(default: {param_info['default']})"
            type_str = f"[{param_info['type']}]" if param_info['type'] != "Any" else ""
            output.append(f"  {param_name} {type_str} {required_str}")
    
    # Returns
    if stage_info.returns:
        output.append(f"\nReturns:")
        output.append(f"  {stage_info.returns}")
    
    # Notes
    if stage_info.notes:
        output.append(f"\nNotes:")
        for note in stage_info.notes:
            output.append(f"  - {note}")
    
    # Examples
    if include_examples and stage_info.examples:
        output.append(f"\nExamples:")
        for example in stage_info.examples:
            output.append(f"  {example}")
    
    # Source info
    output.append(f"\nSource: {stage_info.module_path}")
    
    return "\n".join(output)


def format_stage_list(categorized_stages: Dict[str, List[str]], 
                     category_filter: Optional[str] = None) -> str:
    """Format list of stages for CLI display.
    
    Args:
        categorized_stages: Dictionary mapping categories to stage lists
        category_filter: Optional category to filter by
        
    Returns:
        Formatted stage list string
    """
    extractor = StageDocumentationExtractor()
    output = []
    
    output.append("Available Processing Stages")
    output.append("=" * 50)
    
    for category, stages in categorized_stages.items():
        if category_filter and category != category_filter:
            continue
            
        category_desc = extractor.category_descriptions.get(category, category.replace('_', ' ').title())
        output.append(f"\n{category_desc.upper()}")
        output.append("-" * len(category_desc))
        
        for stage in sorted(stages):
            # Add basic description if available
            stage_info = extractor.get_stage_info(stage)
            if stage_info and stage_info.description:
                desc = stage_info.description.split('.')[0]  # First sentence only
                output.append(f"  {stage:<20} - {desc}")
            else:
                output.append(f"  {stage}")
    
    output.append(f"\nUse 'eeg-processor help-stage <stage_name>' for detailed help")
    output.append(f"Use 'eeg-processor list-stages --category <category>' to filter by category")
    
    return "\n".join(output)