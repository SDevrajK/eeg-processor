"""
Enhanced interactive configuration wizard for EEG Processor.

This module provides an improved configuration wizard that integrates with the
preset system and stage documentation for a better user experience.
"""

import click
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .preset_manager import PresetManager, PresetInfo
from .stage_documentation import StageDocumentationExtractor
from .smart_defaults import SmartDefaultsEngine
from .exceptions import ConfigurationError


class EnhancedConfigWizard:
    """Enhanced interactive wizard for creating EEG Processor configurations."""
    
    def __init__(self):
        self.config = {}
        self.preset_manager = PresetManager()
        self.stage_extractor = StageDocumentationExtractor()
        self.smart_defaults = SmartDefaultsEngine()
        self.current_preset = None
        self.advanced_mode = False
        self.user_level = 'intermediate'
        
    def run_wizard(self, output_path: str) -> Dict[str, Any]:
        """Run the enhanced interactive configuration wizard."""
        self._show_welcome()
        
        # Step 1: Choose configuration approach
        approach = self._choose_configuration_approach()
        
        if approach == 'preset':
            # Preset-based configuration
            self._configure_with_preset()
        elif approach == 'scratch':
            # Build from scratch
            self._configure_from_scratch()
        elif approach == 'template':
            # Start with template and customize
            self._configure_with_template()
        
        # Common customization steps
        self._customize_paths()
        self._customize_participants()
        self._customize_conditions()
        
        # Advanced options (if requested)
        if self.advanced_mode or click.confirm("Configure advanced options?", default=False):
            self._configure_advanced_options()
        
        # Review and save
        self._review_and_save(output_path)
        
        return self.config
    
    def _show_welcome(self):
        """Show welcome message and gather initial preferences."""
        click.echo()
        click.echo("🧠 " + click.style("EEG Processor Configuration Wizard", fg='blue', bold=True))
        click.echo("=" * 60)
        click.echo("This enhanced wizard will help you create a configuration file")
        click.echo("for your EEG data processing pipeline.")
        click.echo()
        
        # Ask about experience level
        experience_levels = {
            '1': ('beginner', 'New to EEG processing'),
            '2': ('intermediate', 'Some experience with EEG analysis'),
            '3': ('advanced', 'Experienced researcher, show all options')
        }
        
        click.echo("What's your experience level with EEG processing?")
        for key, (level, description) in experience_levels.items():
            click.echo(f"  {key}. {description}")
        
        level_choice = click.prompt("Select experience level", type=click.Choice(['1', '2', '3']))
        experience_level, _ = experience_levels[level_choice]
        
        self.advanced_mode = experience_level == 'advanced'
        self.user_level = experience_level
        self.smart_defaults.update_context('user_level', experience_level)
        
        if experience_level == 'beginner':
            click.echo("\n💡 " + click.style("Tip:", fg='yellow') + " We'll use simple presets and provide helpful guidance.")
        elif experience_level == 'intermediate':
            click.echo("\n💡 " + click.style("Tip:", fg='yellow') + " We'll show common options with explanations.")
        else:
            click.echo("\n💡 " + click.style("Tip:", fg='yellow') + " Full control mode - all options available.")
        
        click.echo()
    
    def _choose_configuration_approach(self) -> str:
        """Let user choose how to configure their pipeline."""
        click.echo("🎯 " + click.style("Step 1: Configuration Approach", fg='green', bold=True))
        click.echo("-" * 40)
        
        approaches = {
            '1': ('preset', 'Use a preset configuration (recommended)', True),
            '2': ('scratch', 'Build configuration from scratch', False),
            '3': ('template', 'Start with template and customize', False)
        }
        
        click.echo("How would you like to create your configuration?")
        for key, (approach_id, description, recommended) in approaches.items():
            marker = "⭐ " if recommended else "   "
            click.echo(f"  {key}. {marker}{description}")
        
        choice = click.prompt("Select approach", type=click.Choice(['1', '2', '3']), default='1')
        approach_id, _, _ = approaches[choice]
        
        click.echo(f"✓ Using {approach_id} approach")
        click.echo()
        
        return approach_id
    
    def _configure_with_preset(self):
        """Configure using a preset as foundation."""
        click.echo("📦 " + click.style("Preset Selection", fg='blue', bold=True))
        click.echo("-" * 30)
        
        # Get available presets
        presets_by_category = self.preset_manager.get_available_presets()
        
        if not presets_by_category:
            click.echo("❌ No presets available. Falling back to scratch configuration.")
            self._configure_from_scratch()
            return
        
        # Show presets by category
        preset_choices = {}
        choice_num = 1
        
        for category, presets in presets_by_category.items():
            click.echo(f"\n{category.upper()} PRESETS:")
            for preset_info in presets:
                preset_choices[str(choice_num)] = preset_info
                # Truncate long descriptions for display
                desc = preset_info.description
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                
                click.echo(f"  {choice_num}. {click.style(preset_info.name, fg='cyan')} - {desc}")
                
                # Show use cases for context
                if preset_info.use_cases:
                    use_cases = ", ".join(preset_info.use_cases[:2])
                    if len(preset_info.use_cases) > 2:
                        use_cases += f" (+{len(preset_info.use_cases)-2} more)"
                    click.echo(f"      💡 Good for: {use_cases}")
                
                choice_num += 1
        
        click.echo()
        
        # Let user choose preset
        preset_choice = click.prompt(
            "Select preset", 
            type=click.Choice(list(preset_choices.keys()))
        )
        
        selected_preset = preset_choices[preset_choice]
        self.current_preset = selected_preset
        
        # Load preset configuration
        try:
            preset_data = self.preset_manager.load_preset(selected_preset.name)
            self.config = copy.deepcopy(preset_data['config_template'])
            
            click.echo(f"✓ Loaded preset: {click.style(selected_preset.name, fg='green')}")
            click.echo(f"  Description: {selected_preset.description}")
            
            # Show what this preset includes
            if 'stages' in self.config:
                stage_names = []
                for stage in self.config['stages']:
                    if isinstance(stage, dict):
                        stage_names.append(list(stage.keys())[0])
                    else:
                        stage_names.append(str(stage))
                
                click.echo(f"  Includes {len(stage_names)} processing stages:")
                click.echo(f"    {' → '.join(stage_names)}")
            
            click.echo()
            
        except Exception as e:
            click.echo(f"❌ Error loading preset: {e}")
            click.echo("Falling back to scratch configuration.")
            self._configure_from_scratch()
    
    def _configure_from_scratch(self):
        """Build configuration from scratch with guided stage selection."""
        click.echo("🔧 " + click.style("Building from Scratch", fg='blue', bold=True))
        click.echo("-" * 35)
        
        # Initialize basic structure
        self.config = {
            'study': {
                'name': 'EEG Study',
                'description': '',
                'experimenter': ''
            },
            'paths': {},
            'participants': 'auto',
            'stages': [],
            'conditions': [],
            'output': {
                'save_intermediates': False,
                'figure_format': 'png',
                'dpi': 150
            }
        }
        
        # Get available stages by category
        categorized_stages = self.stage_extractor.get_all_stages()
        
        click.echo("Let's build your processing pipeline step by step.")
        click.echo("We'll go through each category of processing stages:")
        click.echo()
        
        selected_stages = []
        
        for category, stages in categorized_stages.items():
            category_desc = self.stage_extractor.category_descriptions.get(
                category, category.replace('_', ' ').title()
            )
            
            click.echo(f"📂 {click.style(category_desc.upper(), fg='yellow')}")
            click.echo("-" * len(category_desc))
            
            # Get progressive options for this category
            visible_options = self.smart_defaults.get_progressive_options(self.user_level, 'stages')
            
            # Show stages in this category
            for stage in sorted(stages):
                # Check if this stage should be shown based on user level
                if visible_options and stage in visible_options and not visible_options[stage]:
                    continue  # Skip this stage for this user level
                
                stage_info = self.stage_extractor.get_stage_info(stage)
                if stage_info:
                    desc = stage_info.description.split('.')[0]  # First sentence
                    
                    # Smart default recommendation
                    recommended = self._is_stage_recommended(stage, category)
                    default_choice = recommended if self.user_level == 'beginner' else None
                    
                    include = click.confirm(
                        f"Include {click.style(stage, fg='cyan')}? - {desc}",
                        default=default_choice
                    )
                    
                    if include:
                        selected_stages.append(stage)
                        
                        # Contextual help integration
                        if self.user_level == 'beginner' or click.confirm(f"  Show details for {stage}?", default=False):
                            self._show_stage_help(stage_info, contextual=True)
            
            click.echo()
        
        # Convert to stage config format
        self.config['stages'] = [stage for stage in selected_stages]
        
        click.echo(f"✓ Pipeline configured with {len(selected_stages)} stages")
        if selected_stages:
            click.echo(f"  Order: {' → '.join(selected_stages)}")
        click.echo()
    
    def _configure_with_template(self):
        """Start with a template and allow customization."""
        click.echo("📋 " + click.style("Template-based Configuration", fg='blue', bold=True))
        click.echo("-" * 40)
        
        # Use basic-erp as default template
        try:
            template_data = self.preset_manager.load_preset('basic-erp')
            self.config = copy.deepcopy(template_data['config_template'])
            
            click.echo("✓ Starting with Basic ERP template")
            click.echo("  You can customize any part of this configuration.")
            click.echo()
            
        except Exception as e:
            click.echo(f"❌ Error loading template: {e}")
            self._configure_from_scratch()
    
    def _customize_paths(self):
        """Customize data paths with smart defaults."""
        click.echo("📂 " + click.style("Step 2: Data Paths", fg='green', bold=True))
        click.echo("-" * 25)
        
        # Show current paths if any
        current_paths = self.config.get('paths', {})
        
        click.echo("Configure the paths where your data is located:")
        
        # Raw data directory
        default_raw = current_paths.get('raw_data_dir', 'data/raw')
        raw_data_dir = click.prompt(
            "📁 Raw EEG data directory",
            default=default_raw,
            type=click.Path()
        )
        
        # Results directory  
        default_results = current_paths.get('results_dir', 'results')
        results_dir = click.prompt(
            "📊 Results output directory",
            default=default_results,
            type=click.Path()
        )
        
        # File extension with smart detection
        if not self.config.get('paths', {}).get('file_extension'):
            click.echo("\n💡 " + click.style("Tip:", fg='yellow') + " We can detect your file format automatically.")
            
            # Get smart suggestions based on directory contents
            extension_suggestions = self.smart_defaults.get_file_extension_suggestions(raw_data_dir)
            
            common_extensions = {}
            choice_num = 1
            
            click.echo("What type of EEG files do you have?")
            for ext, description, is_recommended in extension_suggestions:
                marker = "⭐ " if is_recommended else "   "
                common_extensions[str(choice_num)] = (ext, description)
                click.echo(f"  {choice_num}. {marker}{description}")
                choice_num += 1
            
            # Add auto-detect option
            common_extensions[str(choice_num)] = ('auto', 'Auto-detect from files in directory')
            click.echo(f"  {choice_num}. Auto-detect from files in directory")
            
            ext_choice = click.prompt("Select file type", type=click.Choice(list(common_extensions.keys())))
            file_extension, _ = common_extensions[ext_choice]
            
            if file_extension == 'auto':
                file_extension = '.vhdr'  # Default fallback
                click.echo("  Using .vhdr as default (you can change this later)")
        else:
            file_extension = current_paths.get('file_extension', '.vhdr')
        
        # Update paths
        self.config['paths'] = {
            'raw_data_dir': raw_data_dir,
            'results_dir': results_dir,
            'file_extension': file_extension
        }
        
        # Optional dataset name
        if click.confirm("Add dataset name for result organization?", default=False):
            dataset_name = click.prompt("Dataset name")
            self.config['dataset_name'] = dataset_name
        
        click.echo(f"✓ Paths configured")
        click.echo()
    
    def _customize_participants(self):
        """Customize participant configuration."""
        click.echo("👥 " + click.style("Step 3: Participants", fg='green', bold=True))
        click.echo("-" * 30)
        
        participant_modes = {
            '1': ('auto', 'Automatically find all EEG files in data directory'),
            '2': ('manual', 'Manually specify participant files'),
            '3': ('list', 'Provide a simple list of filenames')
        }
        
        click.echo("How do you want to specify participants?")
        for key, (mode_id, description) in participant_modes.items():
            marker = "⭐ " if mode_id == 'auto' else "   "
            click.echo(f"  {key}. {marker}{description}")
        
        mode_choice = click.prompt("Select mode", type=click.Choice(['1', '2', '3']), default='1')
        mode_id, _ = participant_modes[mode_choice]
        
        if mode_id == 'auto':
            self.config['participants'] = 'auto'
            click.echo("✓ Auto-discovery enabled - all files will be processed")
            
        elif mode_id == 'list':
            participants = []
            click.echo("Enter participant filenames (press Enter with empty name to finish):")
            click.echo("💡 " + click.style("Tip:", fg='yellow') + " Just enter the filename, e.g., 'sub-01.vhdr'")
            
            while True:
                filename = click.prompt("Filename", default="", show_default=False)
                if not filename:
                    break
                participants.append(filename)
                click.echo(f"   Added: {filename}")
            
            self.config['participants'] = participants
            click.echo(f"✓ {len(participants)} participants specified")
            
        elif mode_id == 'manual':
            # More advanced participant specification
            click.echo("💡 " + click.style("Advanced mode:", fg='yellow') + " You can specify detailed participant info")
            # For now, fall back to simple list mode
            self._customize_participants()
            return
        
        click.echo()
    
    def _customize_conditions(self):
        """Customize experimental conditions with guidance."""
        click.echo("🎯 " + click.style("Step 4: Experimental Conditions", fg='green', bold=True))
        click.echo("-" * 45)
        
        # Check if epoching is in the pipeline
        stages = self.config.get('stages', [])
        has_epoching = False
        for stage in stages:
            if isinstance(stage, dict) and 'epoch' in stage:
                has_epoching = True
                break
            elif isinstance(stage, str) and stage == 'epoch':
                has_epoching = True
                break
        
        if not has_epoching:
            click.echo("ℹ️  Skipping conditions setup (no epoching in pipeline)")
            self.config['conditions'] = []
            click.echo()
            return
        
        click.echo("Define your experimental conditions and their trigger markers:")
        click.echo("💡 " + click.style("Tip:", fg='yellow') + " These are the event types you want to analyze")
        click.echo()
        
        conditions = []
        examples_shown = False
        
        while True:
            if not examples_shown and not conditions:
                click.echo("Examples of conditions:")
                click.echo("  • 'Standard' with markers [1, 'S1']")
                click.echo("  • 'Target' with markers [2, 'S2']")
                click.echo("  • 'Faces' with markers [10, 11, 12]")
                click.echo()
                examples_shown = True
            
            condition_name = click.prompt("Condition name", default="", show_default=False)
            if not condition_name:
                break
            
            # Get markers for this condition
            markers = []
            click.echo(f"Enter trigger markers for '{condition_name}' (press Enter when done):")
            
            while True:
                marker = click.prompt("  Trigger marker", default="", show_default=False)
                if not marker:
                    break
                
                # Try to convert to int, keep as string if not possible
                try:
                    marker = int(marker)
                except ValueError:
                    pass
                
                markers.append(marker)
                click.echo(f"    Added: {marker}")
            
            if markers:
                # Optional description
                description = click.prompt(f"Description for '{condition_name}' (optional)", default="")
                
                condition = {
                    'name': condition_name,
                    'condition_markers': markers
                }
                if description:
                    condition['description'] = description
                
                conditions.append(condition)
                click.echo(f"✓ Added condition '{condition_name}' with {len(markers)} markers")
                click.echo()
        
        self.config['conditions'] = conditions
        click.echo(f"✓ {len(conditions)} conditions configured")
        click.echo()
    
    def _configure_advanced_options(self):
        """Configure advanced processing options."""
        click.echo("⚙️ " + click.style("Advanced Options", fg='blue', bold=True))
        click.echo("-" * 25)
        
        # Output settings
        if click.confirm("Configure output settings?", default=False):
            output_config = self.config.get('output', {})
            
            save_intermediates = click.confirm(
                "Save intermediate processing files?", 
                default=output_config.get('save_intermediates', False)
            )
            
            figure_format = click.prompt(
                "Figure format",
                type=click.Choice(['png', 'pdf', 'svg']),
                default=output_config.get('figure_format', 'png')
            )
            
            dpi = click.prompt(
                "Figure DPI",
                type=int,
                default=output_config.get('dpi', 150)
            )
            
            self.config['output'] = {
                'save_intermediates': save_intermediates,
                'figure_format': figure_format,
                'dpi': dpi,
                'create_report': True
            }
            
            click.echo("✓ Output settings configured")
        
        # Stage parameter customization
        if click.confirm("Customize stage parameters?", default=False):
            self._customize_stage_parameters()
        
        click.echo()
    
    def _customize_stage_parameters(self):
        """Allow customization of stage parameters."""
        click.echo("🔧 Stage Parameter Customization")
        click.echo("-" * 35)
        
        stages = self.config.get('stages', [])
        updated_stages = []
        
        for i, stage in enumerate(stages):
            if isinstance(stage, dict):
                stage_name = list(stage.keys())[0]
                current_params = stage[stage_name]
            else:
                stage_name = stage
                current_params = {}
            
            click.echo(f"\n{i+1}. {click.style(stage_name, fg='cyan')}")
            
            # Show current parameters if any
            if current_params:
                click.echo(f"   Current parameters: {current_params}")
            
            # Get stage info for help
            stage_info = self.stage_extractor.get_stage_info(stage_name)
            
            if click.confirm(f"   Customize {stage_name} parameters?", default=False):
                if stage_info:
                    self._show_stage_help(stage_info, brief=True)
                
                # For now, keep existing parameters
                # TODO: Implement dynamic parameter editing
                click.echo(f"   Keeping current parameters for {stage_name}")
                
            updated_stages.append(stage)
        
        self.config['stages'] = updated_stages
    
    def _show_stage_help(self, stage_info, brief: bool = False, contextual: bool = False):
        """Show help information for a stage."""
        click.echo(f"\n📖 {click.style(stage_info.name + ' Help', fg='blue')}")
        click.echo("-" * 30)
        click.echo(f"Description: {stage_info.description}")
        
        # Show contextual information based on user level
        if contextual and self.user_level == 'beginner':
            # Show simplified explanation for beginners
            if stage_info.name == 'filter':
                click.echo("\n🎯 Why use filtering?")
                click.echo("   Removes noise and focuses on brain signals of interest")
            elif stage_info.name == 'detect_bad_channels':
                click.echo("\n🎯 Why detect bad channels?")
                click.echo("   Identifies and fixes broken or noisy electrodes")
            elif stage_info.name == 'rereference':
                click.echo("\n🎯 Why re-reference?")
                click.echo("   Standardizes the electrical reference for all channels")
            elif stage_info.name == 'epoch':
                click.echo("\n🎯 Why create epochs?")
                click.echo("   Extracts time-locked segments around your events of interest")
        
        if not brief and stage_info.parameters:
            param_count = len(stage_info.parameters)
            if self.user_level == 'beginner' and param_count > 2:
                # Show only most important parameters for beginners
                important_params = ['l_freq', 'h_freq', 'threshold', 'method', 'tmin', 'tmax']
                shown_params = []
                for param_name, param_info in stage_info.parameters.items():
                    if param_name in important_params:
                        shown_params.append((param_name, param_info))
                    if len(shown_params) >= 2:
                        break
                
                if shown_params:
                    click.echo("\n📊 Key parameters:")
                    for param_name, param_info in shown_params:
                        required = "required" if param_info['required'] else f"default: {param_info['default']}"
                        click.echo(f"  • {param_name} ({param_info['type']}) - {required}")
            else:
                # Show more parameters for intermediate/advanced users
                click.echo("\n📊 Parameters:")
                for param_name, param_info in list(stage_info.parameters.items())[:3]:
                    required = "required" if param_info['required'] else f"default: {param_info['default']}"
                    click.echo(f"  • {param_name} ({param_info['type']}) - {required}")
                
                if len(stage_info.parameters) > 3:
                    click.echo(f"  ... and {len(stage_info.parameters)-3} more parameters")
        
        if stage_info.notes and contextual:
            click.echo(f"\n💡 Tip: {stage_info.notes[0]}")
        
        click.echo()
    
    def _is_stage_recommended(self, stage_name: str, category: str) -> bool:
        """Determine if a stage is recommended for beginners."""
        # Core stages that are almost always needed
        core_stages = ['filter', 'detect_bad_channels', 'rereference', 'epoch']
        
        # Advanced stages that beginners might skip
        advanced_stages = ['clean_rawdata_asr', 'remove_artifacts', 'time_frequency']
        
        if stage_name in core_stages:
            return True
        elif stage_name in advanced_stages and self.user_level == 'beginner':
            return False
        else:
            return self.user_level != 'beginner'
    
    def _review_and_save(self, output_path: str):
        """Review configuration and save to file."""
        click.echo("📋 " + click.style("Step 5: Review & Save", fg='green', bold=True))
        click.echo("-" * 35)
        
        # Configuration summary
        click.echo("Configuration Summary:")
        click.echo("-" * 20)
        
        if self.current_preset:
            click.echo(f"🎯 Based on preset: {click.style(self.current_preset.name, fg='cyan')}")
        
        click.echo(f"📁 Raw data: {self.config['paths']['raw_data_dir']}")
        click.echo(f"📊 Results: {self.config['paths']['results_dir']}")
        click.echo(f"📄 File type: {self.config['paths']['file_extension']}")
        
        # Participants summary
        participants = self.config.get('participants', 'auto')
        if participants == 'auto':
            click.echo(f"👥 Participants: auto-discovery")
        elif isinstance(participants, list):
            click.echo(f"👥 Participants: {len(participants)} specified")
        
        # Pipeline summary
        stages = self.config.get('stages', [])
        if stages:
            stage_names = []
            for stage in stages:
                if isinstance(stage, dict):
                    stage_names.append(list(stage.keys())[0])
                else:
                    stage_names.append(str(stage))
            click.echo(f"⚙️  Pipeline: {' → '.join(stage_names)}")
        
        # Conditions summary
        conditions = self.config.get('conditions', [])
        if conditions:
            condition_names = [c['name'] for c in conditions]
            click.echo(f"🎯 Conditions: {', '.join(condition_names)}")
        
        click.echo()
        
        # Option to show full config
        if click.confirm("Show full configuration details?", default=False):
            click.echo("\n" + "="*50)
            click.echo(yaml.dump(self.config, default_flow_style=False, indent=2))
            click.echo("="*50)
        
        # Save configuration
        if click.confirm(f"\n💾 Save configuration to '{output_path}'?", default=True):
            try:
                # Use the same enhanced formatting as presets
                self._write_config_with_comments(output_path)
                
                click.echo(f"✅ Configuration saved to: {click.style(output_path, fg='green')}")
                
                # Quick validation
                try:
                    from .config_loader import load_config
                    load_config(output_path)
                    click.echo("✅ Configuration is valid!")
                except Exception as e:
                    click.echo(f"⚠️  Validation warning: {e}")
                    click.echo("   You may want to review the configuration.")
                
            except Exception as e:
                click.echo(f"❌ Error saving configuration: {e}")
                return
        
        # Next steps
        click.echo()
        click.echo("🎉 " + click.style("Configuration wizard completed!", fg='green', bold=True))
        click.echo()
        click.echo("Next steps:")
        click.echo(f"  1. 📝 Review: {output_path}")
        click.echo(f"  2. ✓ Validate: eeg-processor validate {output_path}")
        click.echo(f"  3. 🚀 Process: eeg-processor process {output_path}")
        
        if self.config.get('conditions'):
            click.echo(f"  4. 📊 Generate reports: eeg-processor quality-report results/")
        
        click.echo()
    
    def _write_config_with_comments(self, output_path: str):
        """Write configuration with helpful comments."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# ========================================\n")
            f.write("# EEG PROCESSOR CONFIGURATION\n")
            f.write("# ========================================\n")
            f.write("# Generated by Enhanced Configuration Wizard\n")
            
            if self.current_preset:
                f.write(f"# Based on preset: {self.current_preset.name}\n")
            
            f.write("#\n")
            f.write("# This configuration was created interactively.\n")
            f.write("# You can edit any section below as needed.\n")
            f.write("# ========================================\n\n")
            
            # Write each section with comments
            sections = [
                ('study', 'STUDY INFORMATION'),
                ('paths', 'DATA PATHS'),
                ('dataset_name', 'DATASET ORGANIZATION'),
                ('participants', 'PARTICIPANTS'),
                ('stages', 'PROCESSING PIPELINE'),
                ('conditions', 'EXPERIMENTAL CONDITIONS'),
                ('output', 'OUTPUT SETTINGS')
            ]
            
            for section_key, section_title in sections:
                if section_key in self.config:
                    f.write(f"# ========================================\n")
                    f.write(f"# {section_title}\n")
                    f.write(f"# ========================================\n")
                    f.write(yaml.dump({section_key: self.config[section_key]}, 
                                    default_flow_style=False, indent=2))
                    f.write("\n")


def run_enhanced_wizard(output_path: str) -> Dict[str, Any]:
    """Run the enhanced interactive configuration wizard."""
    wizard = EnhancedConfigWizard()
    return wizard.run_wizard(output_path)