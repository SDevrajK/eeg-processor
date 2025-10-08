"""
Automatic documentation generator for EEG Processor stages.

This module creates comprehensive markdown documentation for all processing stages,
including parameters, examples, and usage guidelines.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import inspect
from datetime import datetime

from .stage_documentation import StageDocumentationExtractor, StageInfo


class MarkdownDocGenerator:
    """Generator for markdown documentation of EEG processing stages."""
    
    def __init__(self):
        self.extractor = StageDocumentationExtractor()
        
    def generate_complete_documentation(self, output_dir: str = "docs/stages") -> Dict[str, str]:
        """Generate complete markdown documentation for all stages.
        
        Args:
            output_dir: Directory to save documentation files
            
        Returns:
            Dictionary mapping filenames to generated content
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Generate main index file
        index_content = self._generate_index()
        index_file = output_path / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index_content)
        generated_files["README.md"] = index_content
        
        # Generate category overview files
        categorized_stages = self.extractor.get_all_stages()
        for category, stages in categorized_stages.items():
            category_content = self._generate_category_overview(category, stages)
            category_file = output_path / f"{category}.md"
            with open(category_file, 'w', encoding='utf-8') as f:
                f.write(category_content)
            generated_files[f"{category}.md"] = category_content
        
        # Generate individual stage documentation
        for category, stages in categorized_stages.items():
            category_dir = output_path / category
            category_dir.mkdir(exist_ok=True)
            
            for stage_name in stages:
                stage_info = self.extractor.get_stage_info(stage_name)
                if stage_info:
                    stage_content = self._generate_stage_documentation(stage_info)
                    stage_file = category_dir / f"{stage_name}.md"
                    with open(stage_file, 'w', encoding='utf-8') as f:
                        f.write(stage_content)
                    generated_files[f"{category}/{stage_name}.md"] = stage_content
        
        # Generate quick reference
        quick_ref_content = self._generate_quick_reference()
        quick_ref_file = output_path / "quick-reference.md"
        with open(quick_ref_file, 'w', encoding='utf-8') as f:
            f.write(quick_ref_content)
        generated_files["quick-reference.md"] = quick_ref_content
        
        # Generate troubleshooting guide
        troubleshooting_content = self._generate_troubleshooting_guide()
        troubleshooting_file = output_path / "troubleshooting.md"
        with open(troubleshooting_file, 'w', encoding='utf-8') as f:
            f.write(troubleshooting_content)
        generated_files["troubleshooting.md"] = troubleshooting_content
        
        return generated_files
    
    def _generate_index(self) -> str:
        """Generate the main index documentation."""
        content = []
        
        # Header
        content.append("# EEG Processor - Stage Documentation")
        content.append("")
        content.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        content.append("")
        content.append("This documentation provides comprehensive information about all available")
        content.append("processing stages in the EEG Processor pipeline.")
        content.append("")
        
        # Quick links
        content.append("## Quick Links")
        content.append("")
        content.append("- [📋 Quick Reference](quick-reference.md) - Overview of all stages")
        content.append("- [🔧 Troubleshooting](troubleshooting.md) - Common issues and solutions")
        content.append("")
        
        # Categories overview
        content.append("## Processing Categories")
        content.append("")
        content.append("EEG processing stages are organized into logical categories:")
        content.append("")
        
        categorized_stages = self.extractor.get_all_stages()
        for category, stages in categorized_stages.items():
            category_desc = self.extractor.category_descriptions.get(
                category, category.replace('_', ' ').title()
            )
            content.append(f"### [{category_desc}]({category}.md)")
            content.append("")
            content.append(f"*{len(stages)} stages available*")
            content.append("")
            
            # List stages in category
            for stage_name in sorted(stages):
                stage_info = self.extractor.get_stage_info(stage_name)
                if stage_info:
                    brief_desc = stage_info.description.split('.')[0]
                    content.append(f"- [`{stage_name}`]({category}/{stage_name}.md) - {brief_desc}")
            content.append("")
        
        # Usage instructions
        content.append("## Using This Documentation")
        content.append("")
        content.append("### Command Line Help")
        content.append("")
        content.append("```bash")
        content.append("# List all available stages")
        content.append("eeg-processor list-stages")
        content.append("")
        content.append("# Get help for a specific stage")
        content.append("eeg-processor help-stage filter")
        content.append("")
        content.append("# Show examples")
        content.append("eeg-processor help-stage filter --examples")
        content.append("```")
        content.append("")
        
        content.append("### Configuration Usage")
        content.append("")
        content.append("Each stage can be used in your YAML configuration file:")
        content.append("")
        content.append("```yaml")
        content.append("stages:")
        content.append("  # Simple stage (uses defaults)")
        content.append("  - filter")
        content.append("  ")
        content.append("  # Stage with parameters")
        content.append("  - filter:")
        content.append("      l_freq: 0.1")
        content.append("      h_freq: 40")
        content.append("      notch: 50")
        content.append("```")
        content.append("")
        
        # Footer
        content.append("---")
        content.append("")
        content.append("*This documentation is automatically generated from the source code.*")
        content.append("*For the most up-to-date information, use the CLI help commands.*")
        
        return "\n".join(content)
    
    def _generate_category_overview(self, category: str, stages: List[str]) -> str:
        """Generate overview documentation for a category."""
        content = []
        
        category_desc = self.extractor.category_descriptions.get(
            category, category.replace('_', ' ').title()
        )
        
        # Header
        content.append(f"# {category_desc}")
        content.append("")
        content.append(f"*{len(stages)} stages in this category*")
        content.append("")
        
        # Category description
        category_descriptions = {
            'data_handling': (
                "Data handling stages manage the loading, saving, and initial preparation "
                "of EEG data. These stages typically run at the beginning or end of the pipeline."
            ),
            'preprocessing': (
                "Preprocessing stages clean and prepare the raw EEG data for analysis. "
                "This includes filtering, artifact removal, and channel management."
            ),
            'condition_handling': (
                "Condition handling stages work with experimental conditions and events. "
                "They segment data around events and organize it for analysis."
            ),
            'post_epoching': (
                "Post-epoching stages operate on segmented data to extract meaningful "
                "information and perform advanced analyses."
            ),
            'visualization': (
                "Visualization stages create plots and interactive displays for "
                "data inspection and quality control."
            )
        }
        
        if category in category_descriptions:
            content.append("## Overview")
            content.append("")
            content.append(category_descriptions[category])
            content.append("")
        
        # Typical pipeline order
        if category in ['data_handling', 'preprocessing', 'condition_handling']:
            content.append("## Typical Pipeline Order")
            content.append("")
            
            pipeline_orders = {
                'data_handling': ["crop", "adjust_events", "correct_triggers"],
                'preprocessing': ["filter", "detect_bad_channels", "rereference", "remove_artifacts", "clean_rawdata_asr"],
                'condition_handling': ["segment_condition", "epoch"]
            }
            
            if category in pipeline_orders:
                for i, stage in enumerate(pipeline_orders[category], 1):
                    if stage in stages:
                        content.append(f"{i}. [`{stage}`]({category}/{stage}.md)")
                content.append("")
        
        # Stage listings
        content.append("## Available Stages")
        content.append("")
        
        for stage_name in sorted(stages):
            stage_info = self.extractor.get_stage_info(stage_name)
            if stage_info:
                content.append(f"### [`{stage_name}`]({stage_name}.md)")
                content.append("")
                content.append(stage_info.description)
                content.append("")
                
                # Show key parameters
                if stage_info.parameters:
                    key_params = list(stage_info.parameters.items())[:3]  # Show first 3
                    if key_params:
                        content.append("**Key Parameters:**")
                        for param_name, param_info in key_params:
                            param_type = param_info.get('type', 'any')
                            content.append(f"- `{param_name}` (*{param_type}*)")
                        content.append("")
                
                content.append("---")
                content.append("")
        
        # Back link
        content.append("[← Back to Overview](README.md)")
        
        return "\n".join(content)
    
    def _generate_stage_documentation(self, stage_info: StageInfo) -> str:
        """Generate detailed documentation for a single stage."""
        content = []
        
        # Header
        content.append(f"# {stage_info.name}")
        content.append("")
        content.append(stage_info.description)
        content.append("")
        
        # Function signature
        if hasattr(stage_info, 'signature') and stage_info.signature:
            content.append("## Function Signature")
            content.append("")
            content.append("```python")
            content.append(f"def {stage_info.name}{stage_info.signature}")
            content.append("```")
            content.append("")
        
        # Parameters section
        if stage_info.parameters:
            content.append("## Parameters")
            content.append("")
            
            for param_name, param_info in stage_info.parameters.items():
                param_type = param_info.get('type', 'any')
                default_val = param_info.get('default', 'None')
                required = param_info.get('required', False)
                
                content.append(f"### `{param_name}`")
                content.append("")
                content.append(f"- **Type:** `{param_type}`")
                content.append(f"- **Required:** {'Yes' if required else 'No'}")
                if not required:
                    content.append(f"- **Default:** `{default_val}`")
                
                if 'description' in param_info:
                    content.append(f"- **Description:** {param_info['description']}")
                
                content.append("")
        
        # Return values
        if stage_info.returns:
            content.append("## Returns")
            content.append("")
            if isinstance(stage_info.returns, dict):
                for return_name, return_info in stage_info.returns.items():
                    content.append(f"- **`{return_name}`:** {return_info}")
            else:
                content.append(f"- {stage_info.returns}")
            content.append("")
        
        # Usage examples
        content.append("## Usage Examples")
        content.append("")
        
        # Basic usage
        content.append("### Basic Usage")
        content.append("")
        content.append("```yaml")
        content.append("stages:")
        content.append(f"  - {stage_info.name}")
        content.append("```")
        content.append("")
        
        # Advanced usage with parameters
        if stage_info.parameters:
            content.append("### With Custom Parameters")
            content.append("")
            content.append("```yaml")
            content.append("stages:")
            content.append(f"  - {stage_info.name}:")
            
            # Show example parameters
            example_params = self._get_example_parameters(stage_info.name, stage_info.parameters)
            for param_name, example_val in example_params.items():
                if isinstance(example_val, str):
                    content.append(f"      {param_name}: \"{example_val}\"")
                else:
                    content.append(f"      {param_name}: {example_val}")
            content.append("```")
            content.append("")
        
        # CLI usage
        content.append("### Command Line Help")
        content.append("")
        content.append("```bash")
        content.append(f"eeg-processor help-stage {stage_info.name}")
        content.append("```")
        content.append("")
        
        # Notes and tips
        if stage_info.notes:
            content.append("## Notes")
            content.append("")
            for note in stage_info.notes:
                content.append(f"- {note}")
            content.append("")
        
        # Dependencies and order
        dependencies = self._get_stage_dependencies(stage_info.name)
        if dependencies:
            content.append("## Dependencies")
            content.append("")
            content.append("This stage should typically be run after:")
            content.append("")
            for dep in dependencies:
                content.append(f"- `{dep}`")
            content.append("")
        
        # Related stages
        related_stages = self._get_related_stages(stage_info.name)
        if related_stages:
            content.append("## Related Stages")
            content.append("")
            for related in related_stages:
                content.append(f"- [`{related}`](../{self._get_stage_category(related)}/{related}.md)")
            content.append("")
        
        # Common issues
        common_issues = self._get_common_issues(stage_info.name)
        if common_issues:
            content.append("## Common Issues")
            content.append("")
            for issue, solution in common_issues.items():
                content.append(f"**{issue}**")
                content.append("")
                content.append(solution)
                content.append("")
        
        # Footer with navigation
        content.append("---")
        content.append("")
        category = self._get_stage_category(stage_info.name)
        content.append(f"[← Back to {category.replace('_', ' ').title()}](../{category}.md) | ")
        content.append("[📋 Quick Reference](../quick-reference.md) | ")
        content.append("[🏠 Main Index](../README.md)")
        
        return "\n".join(content)
    
    def _get_example_parameters(self, stage_name: str, parameters: Dict) -> Dict[str, Any]:
        """Get example parameter values for a stage."""
        examples = {
            'filter': {
                'l_freq': 0.1,
                'h_freq': 40,
                'notch': 50
            },
            'detect_bad_channels': {
                'threshold': 1.5,
                'n_neighbors': 8
            },
            'rereference': {
                'method': 'average'
            },
            'epoch': {
                'tmin': -0.2,
                'tmax': 0.8,
                'baseline': [-0.2, 0]
            },
            'clean_rawdata_asr': {
                'cutoff': 20,
                'method': 'euclid'
            },
            'remove_blinks_emcp': {
                'method': 'eog_regression',
                'eog_channels': ['HEOG', 'VEOG']
            }
        }
        
        if stage_name in examples:
            return examples[stage_name]
        
        # Generate generic examples based on parameter types
        generic_examples = {}
        for param_name, param_info in list(parameters.items())[:3]:  # First 3 params
            param_type = param_info.get('type', 'any')
            if 'float' in param_type.lower():
                generic_examples[param_name] = 1.0
            elif 'int' in param_type.lower():
                generic_examples[param_name] = 10
            elif 'bool' in param_type.lower():
                generic_examples[param_name] = True
            elif 'str' in param_type.lower():
                generic_examples[param_name] = "example"
            elif 'list' in param_type.lower():
                generic_examples[param_name] = [1, 2, 3]
        
        return generic_examples
    
    def _get_stage_dependencies(self, stage_name: str) -> List[str]:
        """Get stages that should typically run before this stage."""
        dependencies = {
            'epoch': ['filter', 'detect_bad_channels'],
            'remove_artifacts': ['filter'],
            'rereference': ['detect_bad_channels'],
            'time_frequency': ['epoch'],
            'time_frequency_average': ['time_frequency']
        }
        return dependencies.get(stage_name, [])
    
    def _get_related_stages(self, stage_name: str) -> List[str]:
        """Get stages that are related or alternative to this stage."""
        related = {
            'filter': ['remove_artifacts', 'clean_rawdata_asr'],
            'remove_artifacts': ['clean_rawdata_asr', 'remove_blinks_emcp'],
            'clean_rawdata_asr': ['remove_artifacts', 'remove_blinks_emcp'],
            'epoch': ['segment_condition'],
            'time_frequency': ['time_frequency_raw', 'time_frequency_average']
        }
        return related.get(stage_name, [])
    
    def _get_stage_category(self, stage_name: str) -> str:
        """Get the category for a stage."""
        categorized_stages = self.extractor.get_all_stages()
        for category, stages in categorized_stages.items():
            if stage_name in stages:
                return category
        return 'misc'
    
    def _get_common_issues(self, stage_name: str) -> Dict[str, str]:
        """Get common issues and solutions for a stage."""
        issues = {
            'filter': {
                'Filtering removes too much data': (
                    "Check your frequency ranges. High-pass filters above 1 Hz may remove "
                    "important slow components. Low-pass filters below 30 Hz may remove "
                    "important frequency content for some analyses."
                ),
                'Edge artifacts': (
                    "Filtering can introduce artifacts at the beginning and end of recordings. "
                    "Consider cropping data or using longer recordings."
                )
            },
            'detect_bad_channels': {
                'Too many channels detected as bad': (
                    "Lower the threshold parameter or check if your data has systemic issues. "
                    "More than 20% bad channels often indicates recording problems."
                ),
                'Good channels marked as bad': (
                    "Increase the threshold parameter or check the n_neighbors setting. "
                    "Dense electrode arrays may need higher n_neighbors values."
                )
            },
            'epoch': {
                'Not enough epochs after rejection': (
                    "Check your event markers and epoch timing. Consider relaxing rejection "
                    "thresholds or improving preprocessing steps."
                ),
                'Baseline period issues': (
                    "Ensure baseline period is within the epoch window and doesn't overlap "
                    "with your events of interest."
                )
            },
            'clean_rawdata_asr': {
                'Over-correction of data': (
                    "Increase the cutoff parameter. Values too low (< 10) may remove valid data. "
                    "Start with cutoff=20 and adjust based on your data quality."
                ),
                'Insufficient artifact removal': (
                    "Decrease the cutoff parameter, but be careful not to go below 10. "
                    "Also ensure you have sufficient calibration data."
                )
            }
        }
        return issues.get(stage_name, {})
    
    def _generate_quick_reference(self) -> str:
        """Generate quick reference documentation."""
        content = []
        
        content.append("# EEG Processor - Quick Reference")
        content.append("")
        content.append("Quick overview of all available processing stages.")
        content.append("")
        
        categorized_stages = self.extractor.get_all_stages()
        
        for category, stages in categorized_stages.items():
            category_desc = self.extractor.category_descriptions.get(
                category, category.replace('_', ' ').title()
            )
            
            content.append(f"## {category_desc}")
            content.append("")
            content.append("| Stage | Description | Key Parameters |")
            content.append("|-------|-------------|----------------|")
            
            for stage_name in sorted(stages):
                stage_info = self.extractor.get_stage_info(stage_name)
                if stage_info:
                    brief_desc = stage_info.description.split('.')[0]
                    
                    # Get key parameters
                    key_params = []
                    if stage_info.parameters:
                        for param_name in list(stage_info.parameters.keys())[:2]:  # First 2
                            key_params.append(f"`{param_name}`")
                    
                    key_params_str = ", ".join(key_params) if key_params else "None"
                    
                    content.append(f"| `{stage_name}` | {brief_desc} | {key_params_str} |")
            
            content.append("")
        
        content.append("## Common Pipeline Examples")
        content.append("")
        
        content.append("### Basic ERP Pipeline")
        content.append("```yaml")
        content.append("stages:")
        content.append("  - filter:")
        content.append("      l_freq: 0.1")
        content.append("      h_freq: 40")
        content.append("  - detect_bad_channels")
        content.append("  - rereference:")
        content.append("      method: average")
        content.append("  - epoch:")
        content.append("      tmin: -0.2")
        content.append("      tmax: 0.8")
        content.append("```")
        content.append("")
        
        content.append("### Artifact Removal Pipeline")
        content.append("```yaml")
        content.append("stages:")
        content.append("  - filter:")
        content.append("      l_freq: 1.0")
        content.append("      h_freq: 40")
        content.append("  - clean_rawdata_asr:")
        content.append("      cutoff: 20")
        content.append("  - remove_blinks_emcp:")
        content.append("      method: eog_regression")
        content.append("      eog_channels: ['HEOG', 'VEOG']")
        content.append("  - remove_artifacts:")
        content.append("      method: ica")
        content.append("```")
        content.append("")
        
        content.append("[🏠 Back to Main Index](README.md)")
        
        return "\n".join(content)
    
    def _generate_troubleshooting_guide(self) -> str:
        """Generate troubleshooting guide."""
        content = []
        
        content.append("# EEG Processor - Troubleshooting Guide")
        content.append("")
        content.append("Common issues and solutions when using processing stages.")
        content.append("")
        
        content.append("## General Issues")
        content.append("")
        
        general_issues = {
            "Configuration file not loading": (
                "1. Check YAML syntax using `eeg-processor validate config.yml`\n"
                "2. Ensure all required fields are present\n"
                "3. Verify file paths exist and are accessible"
            ),
            "Stage not found errors": (
                "1. Use `eeg-processor list-stages` to see available stages\n"
                "2. Check spelling of stage names in your configuration\n"
                "3. Ensure you're using the correct stage names, not function names"
            ),
            "Memory errors during processing": (
                "1. Reduce the number of parallel jobs\n"
                "2. Process participants individually using `--participant`\n"
                "3. Enable intermediate file saving to avoid recomputation"
            ),
            "Slow processing": (
                "1. Use parallel processing with `-j` flag\n"
                "2. Consider using ASR for artifact removal instead of ICA\n"
                "3. Filter data early in the pipeline to reduce computational load"
            )
        }
        
        for issue, solution in general_issues.items():
            content.append(f"### {issue}")
            content.append("")
            content.append(solution)
            content.append("")
        
        content.append("## Stage-Specific Issues")
        content.append("")
        
        # Get common issues for key stages
        key_stages = ['filter', 'detect_bad_channels', 'epoch', 'clean_rawdata_asr', 'remove_artifacts']
        
        for stage_name in key_stages:
            stage_issues = self._get_common_issues(stage_name)
            if stage_issues:
                content.append(f"### {stage_name}")
                content.append("")
                for issue, solution in stage_issues.items():
                    content.append(f"**{issue}**")
                    content.append("")
                    content.append(solution)
                    content.append("")
        
        content.append("## Data Quality Issues")
        content.append("")
        
        quality_issues = {
            "Too many bad channels detected": (
                "- Check recording setup and electrode impedances\n"
                "- Verify reference electrode is properly connected\n"
                "- Consider if high bad channel count is due to participant factors\n"
                "- Adjust bad channel detection threshold if appropriate"
            ),
            "Excessive artifact rejection": (
                "- Review rejection thresholds - they may be too strict\n"
                "- Check if artifacts are due to systematic issues (line noise, movement)\n"
                "- Consider using ASR before epoching to clean continuous data\n"
                "- Verify event timing is correct"
            ),
            "Poor signal quality after preprocessing": (
                "- Check filter settings - avoid over-filtering\n"
                "- Verify reference choice is appropriate for your data\n"
                "- Consider artifact removal order (ASR → EMCP → ICA)\n"
                "- Review original data quality"
            )
        }
        
        for issue, solution in quality_issues.items():
            content.append(f"### {issue}")
            content.append("")
            content.append(solution)
            content.append("")
        
        content.append("## Getting Help")
        content.append("")
        content.append("If you're still experiencing issues:")
        content.append("")
        content.append("1. **Use CLI help commands:**")
        content.append("   ```bash")
        content.append("   eeg-processor help-stage <stage_name>")
        content.append("   eeg-processor validate <config_file>")
        content.append("   ```")
        content.append("")
        content.append("2. **Check configuration validation:**")
        content.append("   ```bash")
        content.append("   eeg-processor validate config.yml --detailed")
        content.append("   ```")
        content.append("")
        content.append("3. **Run with verbose output:**")
        content.append("   ```bash")
        content.append("   eeg-processor process config.yml --verbose")
        content.append("   ```")
        content.append("")
        content.append("4. **Test with minimal configuration:**")
        content.append("   ```bash")
        content.append("   eeg-processor create-config --minimal")
        content.append("   ```")
        content.append("")
        
        content.append("[🏠 Back to Main Index](README.md)")
        
        return "\n".join(content)


def generate_stage_documentation(output_dir: str = "docs/stages") -> Dict[str, str]:
    """Generate complete markdown documentation for all stages.
    
    Args:
        output_dir: Directory to save documentation files
        
    Returns:
        Dictionary mapping filenames to generated content
    """
    generator = MarkdownDocGenerator()
    return generator.generate_complete_documentation(output_dir)