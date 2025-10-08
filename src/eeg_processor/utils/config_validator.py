"""
Smart configuration validation with fix suggestions for EEG Processor.

This module provides intelligent validation of configuration files with
helpful error messages and automatic fix suggestions.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import re

from .exceptions import ValidationError, ConfigurationError
from .smart_defaults import SmartDefaultsEngine


@dataclass
class ValidationIssue:
    """Represents a configuration validation issue."""
    level: str  # 'error', 'warning', 'info'
    section: str
    field: str
    message: str
    suggestion: Optional[str] = None
    auto_fix: Optional[Any] = None


class ConfigValidator:
    """Smart configuration validator with fix suggestions."""
    
    def __init__(self):
        self.smart_defaults = SmartDefaultsEngine()
        self.issues = []
        
        # Define validation rules
        self.validation_rules = {
            'paths': self._validate_paths,
            'participants': self._validate_participants,
            'stages': self._validate_stages,
            'conditions': self._validate_conditions,
            'study': self._validate_study,
            'output': self._validate_output
        }
        
        # Define parameter ranges and constraints
        self.parameter_constraints = {
            'filter': {
                'l_freq': {'type': (int, float, type(None)), 'min': 0.001, 'max': 1000},
                'h_freq': {'type': (int, float, type(None)), 'min': 0.1, 'max': 5000},
                'notch': {'type': (int, float, type(None)), 'min': 30, 'max': 100}
            },
            'epoch': {
                'tmin': {'type': (int, float), 'min': -10.0, 'max': 5.0},
                'tmax': {'type': (int, float), 'min': 0.1, 'max': 60.0},
                'baseline': {'type': (list, type(None))}
            },
            'detect_bad_channels': {
                'threshold': {'type': (int, float), 'min': 0.1, 'max': 10.0},
                'n_neighbors': {'type': int, 'min': 3, 'max': 50}
            },
            'clean_rawdata_asr': {
                'cutoff': {'type': (int, float), 'min': 5, 'max': 100},
                'method': {'type': str, 'choices': ['euclid', 'riemann']}
            }
        }
    
    def validate_config(self, config: Dict[str, Any], 
                       config_path: Optional[str] = None) -> List[ValidationIssue]:
        """Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Optional path to config file for path validation
            
        Returns:
            List of ValidationIssue objects
        """
        self.issues = []
        self.config_path = config_path
        
        # Validate each section
        for section, validator in self.validation_rules.items():
            if section in config:
                try:
                    validator(config[section], config)
                except Exception as e:
                    self.issues.append(ValidationIssue(
                        level='error',
                        section=section,
                        field='',
                        message=f"Error validating {section}: {e}"
                    ))
        
        # Cross-section validation
        self._validate_cross_dependencies(config)
        
        # Parameter-specific validation
        self._validate_stage_parameters(config)
        
        return self.issues
    
    def _validate_paths(self, paths: Dict[str, Any], config: Dict[str, Any]):
        """Validate path configuration."""
        required_paths = ['raw_data_dir', 'results_dir', 'file_extension']
        
        for required in required_paths:
            if required not in paths:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='paths',
                    field=required,
                    message=f"Required path '{required}' is missing",
                    suggestion=f"Add '{required}' to your paths configuration"
                ))
            elif not paths[required]:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='paths',
                    field=required,
                    message=f"Path '{required}' cannot be empty",
                    suggestion=f"Provide a valid path for '{required}'"
                ))
        
        # Validate raw data directory exists or can be created
        if 'raw_data_dir' in paths and paths['raw_data_dir']:
            raw_dir = Path(paths['raw_data_dir'])
            if not raw_dir.exists():
                if self.config_path:
                    # Try relative to config file
                    config_dir = Path(self.config_path).parent
                    relative_path = config_dir / raw_dir
                    if relative_path.exists():
                        raw_dir = relative_path
                
                if not raw_dir.exists():
                    self.issues.append(ValidationIssue(
                        level='warning',
                        section='paths',
                        field='raw_data_dir',
                        message=f"Raw data directory does not exist: {raw_dir}",
                        suggestion="Create the directory or update the path"
                    ))
        
        # Validate file extension format
        if 'file_extension' in paths:
            ext = paths['file_extension']
            if not ext.startswith('.'):
                self.issues.append(ValidationIssue(
                    level='error',
                    section='paths',
                    field='file_extension',
                    message=f"File extension must start with '.': {ext}",
                    auto_fix=f".{ext}" if ext else '.vhdr'
                ))
            
            # Check for common extensions
            common_exts = ['.vhdr', '.edf', '.fif', '.set', '.bdf']
            if ext not in common_exts:
                self.issues.append(ValidationIssue(
                    level='warning',
                    section='paths',
                    field='file_extension',
                    message=f"Uncommon file extension: {ext}",
                    suggestion=f"Common extensions: {', '.join(common_exts)}"
                ))
    
    def _validate_participants(self, participants: Union[str, List, Dict], config: Dict[str, Any]):
        """Validate participant configuration."""
        if participants == 'auto':
            # Auto mode is valid, but check if raw_data_dir exists
            paths = config.get('paths', {})
            raw_dir = paths.get('raw_data_dir')
            if raw_dir and not Path(raw_dir).exists():
                self.issues.append(ValidationIssue(
                    level='warning',
                    section='participants',
                    field='auto',
                    message="Auto participant discovery requires existing raw_data_dir",
                    suggestion="Create the raw data directory or specify participants manually"
                ))
        
        elif isinstance(participants, list):
            if not participants:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='participants',
                    field='list',
                    message="Participant list cannot be empty",
                    suggestion="Add participant files or use 'auto'"
                ))
            
            # Check file extensions match
            paths = config.get('paths', {})
            expected_ext = paths.get('file_extension', '.vhdr')
            
            for i, participant in enumerate(participants):
                if isinstance(participant, str):
                    if not participant.endswith(expected_ext):
                        self.issues.append(ValidationIssue(
                            level='warning',
                            section='participants',
                            field=f'participant_{i}',
                            message=f"Participant file '{participant}' doesn't match expected extension '{expected_ext}'",
                            suggestion=f"Use consistent file extensions or update file_extension setting"
                        ))
        
        elif isinstance(participants, dict):
            if not participants:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='participants',
                    field='dict',
                    message="Participant dictionary cannot be empty",
                    suggestion="Add participant entries or use 'auto'"
                ))
        
        else:
            self.issues.append(ValidationIssue(
                level='error',
                section='participants',
                field='type',
                message=f"Invalid participant type: {type(participants)}",
                suggestion="Use 'auto', a list of files, or a dictionary",
                auto_fix='auto'
            ))
    
    def _validate_stages(self, stages: List, config: Dict[str, Any]):
        """Validate processing stages."""
        if not stages:
            self.issues.append(ValidationIssue(
                level='error',
                section='stages',
                field='list',
                message="At least one processing stage is required",
                suggestion="Add processing stages like filter, epoch, etc."
            ))
            return
        
        # Valid stage names (from stage registry)
        valid_stages = {
            'crop', 'adjust_events', 'correct_triggers', 'filter', 'compute_eog',
            'detect_bad_channels', 'rereference', 'remove_artifacts', 'remove_blinks_emcp',
            'clean_rawdata_asr', 'segment_condition', 'epoch', 'time_frequency',
            'time_frequency_raw', 'time_frequency_average', 'view'
        }
        
        stage_names = []
        for i, stage in enumerate(stages):
            if isinstance(stage, str):
                stage_name = stage
                stage_names.append(stage_name)
            elif isinstance(stage, dict) and len(stage) == 1:
                stage_name = list(stage.keys())[0]
                stage_names.append(stage_name)
            else:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='stages',
                    field=f'stage_{i}',
                    message=f"Invalid stage format: {stage}",
                    suggestion="Use string or single-key dictionary format"
                ))
                continue
            
            if stage_name not in valid_stages:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='stages',
                    field=f'stage_{i}',
                    message=f"Unknown stage: {stage_name}",
                    suggestion=f"Valid stages: {', '.join(sorted(valid_stages))}"
                ))
        
        # Check stage order logic
        self._validate_stage_order(stage_names)
    
    def _validate_stage_order(self, stage_names: List[str]):
        """Validate logical order of processing stages."""
        # Define logical dependencies
        dependencies = {
            'epoch': ['filter'],  # Epoching should come after filtering
            'remove_artifacts': ['filter'],  # ICA needs filtered data
            'rereference': ['detect_bad_channels'],  # Re-ref after bad channel detection
            'time_frequency': ['epoch'],  # TF analysis needs epochs
        }
        
        for stage, required_before in dependencies.items():
            if stage in stage_names:
                stage_idx = stage_names.index(stage)
                for required_stage in required_before:
                    if required_stage in stage_names:
                        required_idx = stage_names.index(required_stage)
                        if required_idx > stage_idx:
                            self.issues.append(ValidationIssue(
                                level='warning',
                                section='stages',
                                field='order',
                                message=f"'{required_stage}' should come before '{stage}'",
                                suggestion=f"Consider reordering stages for optimal results"
                            ))
    
    def _validate_conditions(self, conditions: List[Dict], config: Dict[str, Any]):
        """Validate experimental conditions."""
        if not conditions:
            # Check if epoching is enabled
            stages = config.get('stages', [])
            has_epoching = any(
                'epoch' in str(stage) for stage in stages
            )
            
            if has_epoching:
                self.issues.append(ValidationIssue(
                    level='warning',
                    section='conditions',
                    field='list',
                    message="No conditions defined but epoching is enabled",
                    suggestion="Define experimental conditions or remove epoching stage"
                ))
            return
        
        for i, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                self.issues.append(ValidationIssue(
                    level='error',
                    section='conditions',
                    field=f'condition_{i}',
                    message=f"Condition must be a dictionary: {condition}",
                    suggestion="Use format: {name: '...', condition_markers: [...]}"
                ))
                continue
            
            # Check required fields
            if 'name' not in condition:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='conditions',
                    field=f'condition_{i}',
                    message="Condition missing 'name' field",
                    auto_fix=f"Condition{i+1}"
                ))
            
            if 'condition_markers' not in condition:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='conditions',
                    field=f'condition_{i}',
                    message="Condition missing 'condition_markers' field",
                    suggestion="Add list of event markers for this condition"
                ))
            else:
                markers = condition['condition_markers']
                if not isinstance(markers, list) or not markers:
                    self.issues.append(ValidationIssue(
                        level='error',
                        section='conditions',
                        field=f'condition_{i}',
                        message="condition_markers must be a non-empty list",
                        suggestion="Example: [1, 'S1', 11]"
                    ))
    
    def _validate_study(self, study: Dict[str, Any], config: Dict[str, Any]):
        """Validate study information."""
        recommended_fields = ['name', 'description']
        
        for field in recommended_fields:
            if field not in study or not study[field]:
                self.issues.append(ValidationIssue(
                    level='info',
                    section='study',
                    field=field,
                    message=f"Study {field} is empty",
                    suggestion=f"Consider adding study {field} for documentation"
                ))
    
    def _validate_output(self, output: Dict[str, Any], config: Dict[str, Any]):
        """Validate output settings."""
        if 'figure_format' in output:
            valid_formats = ['png', 'pdf', 'svg', 'jpg']
            if output['figure_format'] not in valid_formats:
                self.issues.append(ValidationIssue(
                    level='error',
                    section='output',
                    field='figure_format',
                    message=f"Invalid figure format: {output['figure_format']}",
                    suggestion=f"Valid formats: {', '.join(valid_formats)}",
                    auto_fix='png'
                ))
        
        if 'dpi' in output:
            dpi = output['dpi']
            if not isinstance(dpi, int) or dpi < 72 or dpi > 600:
                self.issues.append(ValidationIssue(
                    level='warning',
                    section='output',
                    field='dpi',
                    message=f"DPI should be between 72-600: {dpi}",
                    suggestion="Use 150 for screen viewing, 300 for print",
                    auto_fix=150
                ))
    
    def _validate_cross_dependencies(self, config: Dict[str, Any]):
        """Validate dependencies between different sections."""
        # Check if file extension matches participant files
        paths = config.get('paths', {})
        participants = config.get('participants', [])
        
        if isinstance(participants, list) and 'file_extension' in paths:
            expected_ext = paths['file_extension']
            mismatched_files = []
            
            for participant in participants:
                if isinstance(participant, str) and not participant.endswith(expected_ext):
                    mismatched_files.append(participant)
            
            if mismatched_files:
                self.issues.append(ValidationIssue(
                    level='warning',
                    section='cross_validation',
                    field='file_extension_mismatch',
                    message=f"Some participant files don't match file_extension '{expected_ext}'",
                    suggestion="Update file_extension or rename participant files"
                ))
    
    def _validate_stage_parameters(self, config: Dict[str, Any]):
        """Validate parameters for each processing stage."""
        stages = config.get('stages', [])
        
        for i, stage in enumerate(stages):
            if isinstance(stage, dict) and len(stage) == 1:
                stage_name = list(stage.keys())[0]
                stage_params = stage[stage_name]
                
                if stage_name in self.parameter_constraints and isinstance(stage_params, dict):
                    self._validate_parameters(stage_name, stage_params, i)
    
    def _validate_parameters(self, stage_name: str, params: Dict[str, Any], stage_index: int):
        """Validate parameters for a specific stage."""
        constraints = self.parameter_constraints[stage_name]
        
        for param_name, param_value in params.items():
            if param_name in constraints:
                constraint = constraints[param_name]
                
                # Type validation
                if 'type' in constraint:
                    expected_types = constraint['type']
                    if not isinstance(expected_types, tuple):
                        expected_types = (expected_types,)
                    
                    if not isinstance(param_value, expected_types):
                        type_names = [t.__name__ for t in expected_types]
                        self.issues.append(ValidationIssue(
                            level='error',
                            section='stages',
                            field=f'{stage_name}.{param_name}',
                            message=f"Parameter {param_name} should be {'/'.join(type_names)}, got {type(param_value).__name__}",
                            suggestion=f"Check parameter type for {stage_name}.{param_name}"
                        ))
                        continue
                
                # Range validation for numeric types
                if isinstance(param_value, (int, float)):
                    if 'min' in constraint and param_value < constraint['min']:
                        self.issues.append(ValidationIssue(
                            level='error',
                            section='stages',
                            field=f'{stage_name}.{param_name}',
                            message=f"Parameter {param_name} below minimum: {param_value} < {constraint['min']}",
                            auto_fix=constraint['min']
                        ))
                    
                    if 'max' in constraint and param_value > constraint['max']:
                        self.issues.append(ValidationIssue(
                            level='error',
                            section='stages',
                            field=f'{stage_name}.{param_name}',
                            message=f"Parameter {param_name} above maximum: {param_value} > {constraint['max']}",
                            auto_fix=constraint['max']
                        ))
                
                # Choice validation
                if 'choices' in constraint and param_value not in constraint['choices']:
                    self.issues.append(ValidationIssue(
                        level='error',
                        section='stages',
                        field=f'{stage_name}.{param_name}',
                        message=f"Invalid choice for {param_name}: {param_value}",
                        suggestion=f"Valid choices: {', '.join(constraint['choices'])}",
                        auto_fix=constraint['choices'][0]
                    ))
    
    def auto_fix_config(self, config: Dict[str, Any], 
                       issues: Optional[List[ValidationIssue]] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Automatically fix configuration issues where possible.
        
        Args:
            config: Configuration dictionary to fix
            issues: Optional list of issues (will validate if not provided)
            
        Returns:
            Tuple of (fixed_config, list_of_fixes_applied)
        """
        if issues is None:
            issues = self.validate_config(config)
        
        fixed_config = config.copy()
        fixes_applied = []
        
        for issue in issues:
            if issue.auto_fix is not None:
                # Apply auto-fix
                if issue.section in fixed_config:
                    section = fixed_config[issue.section]
                    
                    if '.' in issue.field:
                        # Nested field (e.g., 'filter.l_freq')
                        parts = issue.field.split('.')
                        target = section
                        for part in parts[:-1]:
                            if isinstance(target, list):
                                # Handle stage parameters
                                for stage in target:
                                    if isinstance(stage, dict) and part in stage:
                                        target = stage[part]
                                        break
                            elif isinstance(target, dict) and part in target:
                                target = target[part]
                        
                        if isinstance(target, dict):
                            target[parts[-1]] = issue.auto_fix
                            fixes_applied.append(f"Fixed {issue.section}.{issue.field}: {issue.auto_fix}")
                    
                    else:
                        # Direct field
                        if isinstance(section, dict):
                            section[issue.field] = issue.auto_fix
                            fixes_applied.append(f"Fixed {issue.section}.{issue.field}: {issue.auto_fix}")
                        elif isinstance(section, list) and issue.field.isdigit():
                            idx = int(issue.field)
                            if 0 <= idx < len(section):
                                section[idx] = issue.auto_fix
                                fixes_applied.append(f"Fixed {issue.section}[{idx}]: {issue.auto_fix}")
                
                else:
                    # Create missing section
                    fixed_config[issue.section] = {issue.field: issue.auto_fix}
                    fixes_applied.append(f"Added {issue.section}.{issue.field}: {issue.auto_fix}")
        
        return fixed_config, fixes_applied
    
    def generate_validation_report(self, issues: List[ValidationIssue]) -> str:
        """Generate a human-readable validation report."""
        if not issues:
            return "✅ Configuration is valid - no issues found!"
        
        # Group issues by level
        errors = [i for i in issues if i.level == 'error']
        warnings = [i for i in issues if i.level == 'warning']
        info = [i for i in issues if i.level == 'info']
        
        report = []
        
        if errors:
            report.append("❌ ERRORS (must be fixed):")
            for issue in errors:
                report.append(f"  • {issue.section}.{issue.field}: {issue.message}")
                if issue.suggestion:
                    report.append(f"    💡 {issue.suggestion}")
                if issue.auto_fix is not None:
                    report.append(f"    🔧 Auto-fix available: {issue.auto_fix}")
            report.append("")
        
        if warnings:
            report.append("⚠️  WARNINGS (recommended to fix):")
            for issue in warnings:
                report.append(f"  • {issue.section}.{issue.field}: {issue.message}")
                if issue.suggestion:
                    report.append(f"    💡 {issue.suggestion}")
            report.append("")
        
        if info:
            report.append("ℹ️  INFO (optional improvements):")
            for issue in info:
                report.append(f"  • {issue.section}.{issue.field}: {issue.message}")
                if issue.suggestion:
                    report.append(f"    💡 {issue.suggestion}")
        
        return "\n".join(report)


def validate_config_file(config_path: str, auto_fix: bool = False) -> Tuple[List[ValidationIssue], Optional[str]]:
    """Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        auto_fix: Whether to apply automatic fixes
        
    Returns:
        Tuple of (issues, fixed_config_content) where fixed_config_content is None if no fixes applied
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return [ValidationIssue(
            level='error',
            section='file',
            field='loading',
            message=f"Failed to load configuration file: {e}"
        )], None
    
    validator = ConfigValidator()
    issues = validator.validate_config(config, config_path)
    
    fixed_content = None
    if auto_fix:
        fixed_config, fixes = validator.auto_fix_config(config, issues)
        if fixes:
            fixed_content = yaml.dump(fixed_config, default_flow_style=False, indent=2)
    
    return issues, fixed_content