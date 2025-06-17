"""Interactive configuration wizard for EEG Processor."""

import click
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class InteractiveConfigWizard:
    """Interactive wizard for creating EEG Processor configurations."""
    
    def __init__(self):
        self.config = {}
        
    def run_wizard(self, output_path: str) -> Dict[str, Any]:
        """Run the interactive configuration wizard."""
        click.echo("🧠 EEG Processor Configuration Wizard")
        click.echo("=" * 50)
        click.echo("This wizard will help you create a configuration file for your EEG data.")
        click.echo()
        
        # Step 1: Data format and paths
        self._configure_data_format()
        self._configure_paths()
        
        # Step 2: Participants
        self._configure_participants()
        
        # Step 3: Processing stages
        self._configure_stages()
        
        # Step 4: Processing parameters
        self._configure_processing_parameters()
        
        # Step 5: Conditions
        self._configure_conditions()
        
        # Step 6: Quality control
        self._configure_quality_control()
        
        # Step 7: Review and save
        self._review_and_save(output_path)
        
        return self.config
    
    def _configure_data_format(self):
        """Configure data format."""
        click.echo("📁 Step 1: Data Format")
        click.echo("-" * 20)
        
        formats = {
            '1': ('brainvision', 'BrainVision (.vhdr/.vmrk/.eeg)'),
            '2': ('edf', 'European Data Format (.edf)'),
            '3': ('fif', 'FIF - MNE-Python native (.fif)'),
            '4': ('eeglab', 'EEGLAB (.set)'),
            '5': ('auto', 'Auto-detect format')
        }
        
        click.echo("Available data formats:")
        for key, (format_id, description) in formats.items():
            click.echo(f"  {key}. {description}")
        
        choice = click.prompt("Select data format", type=click.Choice(list(formats.keys())))
        data_format, _ = formats[choice]
        
        if data_format != 'auto':
            self.config['data_format'] = data_format
        
        click.echo(f"✓ Data format: {data_format}")
        click.echo()
    
    def _configure_paths(self):
        """Configure file paths."""
        click.echo("📂 Step 2: File Paths")
        click.echo("-" * 20)
        
        # Raw data directory
        raw_data_dir = click.prompt(
            "Raw data directory",
            default="data/raw",
            type=click.Path()
        )
        
        # Results directory
        results_dir = click.prompt(
            "Results directory",
            default="data/processed",
            type=click.Path()
        )
        
        # File extension
        default_ext = ".vhdr" if self.config.get('data_format') == 'brainvision' else ".edf"
        file_extension = click.prompt(
            "File extension",
            default=default_ext
        )
        
        self.config['paths'] = {
            'raw_data_dir': raw_data_dir,
            'results_dir': results_dir,
            'file_extension': file_extension
        }
        
        # Optional paths
        if click.confirm("Configure additional paths (interim, figures)?", default=False):
            interim_dir = click.prompt("Interim directory", default="data/interim")
            figures_dir = click.prompt("Figures directory", default="data/figures")
            
            self.config['paths']['interim_dir'] = interim_dir
            self.config['paths']['figures_dir'] = figures_dir
        
        click.echo("✓ Paths configured")
        click.echo()
    
    def _configure_participants(self):
        """Configure participants."""
        click.echo("👥 Step 3: Participants")
        click.echo("-" * 20)
        
        participant_modes = {
            '1': 'auto',
            '2': 'manual',
            '3': 'file'
        }
        
        click.echo("Participant selection modes:")
        click.echo("  1. Auto-discover all files in raw data directory")
        click.echo("  2. Manually specify participant files")
        click.echo("  3. Load from participant list file")
        
        mode_choice = click.prompt("Select mode", type=click.Choice(['1', '2', '3']))
        mode = participant_modes[mode_choice]
        
        if mode == 'auto':
            self.config['participants'] = 'auto'
            click.echo("✓ Auto-discovery enabled")
            
        elif mode == 'manual':
            participants = {}
            click.echo("Enter participant information (press Enter with empty ID to finish):")
            
            while True:
                participant_id = click.prompt("Participant ID", default="", show_default=False)
                if not participant_id:
                    break
                    
                filename = click.prompt(f"Filename for {participant_id}")
                participants[participant_id] = filename
            
            self.config['participants'] = participants
            click.echo(f"✓ {len(participants)} participants configured")
            
        elif mode == 'file':
            participant_file = click.prompt("Participant list file", type=click.Path(exists=True))
            # TODO: Implement participant file loading
            click.echo("✓ Participant file specified (loading to be implemented)")
        
        click.echo()
    
    def _configure_stages(self):
        """Configure processing stages."""
        click.echo("⚙️ Step 4: Processing Stages")
        click.echo("-" * 20)
        
        available_stages = [
            ('load_data', 'Load raw EEG data', True),
            ('montage', 'Apply electrode montage', False),
            ('filter', 'Apply frequency filters', True),
            ('bad_channels', 'Detect and interpolate bad channels', False),
            ('reref', 'Re-reference data', False),
            ('epoching', 'Create epochs around events', True),
            ('artifact_rejection', 'Reject artifacts', True),
            ('ica', 'Independent Component Analysis', False),
            ('time_frequency', 'Time-frequency analysis', False),
            ('evoked', 'Compute evoked responses', True),
            ('save_results', 'Save processed data', True)
        ]
        
        click.echo("Available processing stages:")
        for i, (stage_id, description, default) in enumerate(available_stages):
            status = "✓" if default else " "
            click.echo(f"  {i+1:2d}. [{status}] {description}")
        
        click.echo()
        click.echo("Select stages to include (default selection shown with ✓):")
        
        selected_stages = []
        for stage_id, description, default in available_stages:
            include = click.confirm(f"Include {description}?", default=default)
            if include:
                selected_stages.append(stage_id)
        
        self.config['stages'] = selected_stages
        click.echo(f"✓ {len(selected_stages)} stages selected")
        click.echo()
    
    def _configure_processing_parameters(self):
        """Configure processing parameters."""
        click.echo("🔧 Step 5: Processing Parameters")
        click.echo("-" * 20)
        
        # Filtering parameters
        if 'filter' in self.config['stages']:
            click.echo("Filtering parameters:")
            lowpass = click.prompt("Lowpass frequency (Hz)", default=40, type=float)
            highpass = click.prompt("Highpass frequency (Hz)", default=0.1, type=float)
            
            notch_filter = click.confirm("Apply notch filter?", default=True)
            notch_freq = 50 if notch_filter else None
            if notch_filter:
                notch_freq = click.prompt("Notch frequency (Hz)", default=50, type=float)
            
            self.config['filtering'] = {
                'lowpass': lowpass,
                'highpass': highpass
            }
            if notch_freq:
                self.config['filtering']['notch'] = notch_freq
            
            click.echo()
        
        # Epoching parameters
        if 'epoching' in self.config['stages']:
            click.echo("Epoching parameters:")
            tmin = click.prompt("Epoch start time (s)", default=-0.2, type=float)
            tmax = click.prompt("Epoch end time (s)", default=0.8, type=float)
            
            baseline_correction = click.confirm("Apply baseline correction?", default=True)
            baseline = None
            if baseline_correction:
                baseline_start = click.prompt("Baseline start (s)", default=-0.2, type=float)
                baseline_end = click.prompt("Baseline end (s)", default=0.0, type=float)
                baseline = [baseline_start, baseline_end]
            
            self.config['epoching'] = {
                'tmin': tmin,
                'tmax': tmax
            }
            if baseline:
                self.config['epoching']['baseline'] = baseline
            
            click.echo()
        
        # Artifact rejection parameters
        if 'artifact_rejection' in self.config['stages']:
            click.echo("Artifact rejection parameters:")
            peak_to_peak = click.prompt("Peak-to-peak threshold (µV)", default=100, type=float)
            self.config['artifact_rejection'] = {
                'peak_to_peak': peak_to_peak * 1e-6  # Convert to V
            }
            
            click.echo()
        
        click.echo("✓ Processing parameters configured")
        click.echo()
    
    def _configure_conditions(self):
        """Configure experimental conditions."""
        click.echo("🎯 Step 6: Experimental Conditions")
        click.echo("-" * 20)
        
        if 'epoching' not in self.config['stages']:
            click.echo("Skipping conditions (epoching not enabled)")
            click.echo()
            return
        
        conditions = []
        click.echo("Enter experimental conditions (press Enter with empty name to finish):")
        
        while True:
            condition_name = click.prompt("Condition name", default="", show_default=False)
            if not condition_name:
                break
            
            description = click.prompt(f"Description for '{condition_name}'", default="")
            
            markers = []
            click.echo(f"Enter trigger markers for '{condition_name}' (press Enter with empty marker to finish):")
            while True:
                marker = click.prompt("Trigger marker", default="", show_default=False)
                if not marker:
                    break
                # Try to convert to int, keep as string if not possible
                try:
                    marker = int(marker)
                except ValueError:
                    pass
                markers.append(marker)
            
            if markers:
                condition = {
                    'name': condition_name,
                    'condition_markers': markers
                }
                if description:
                    condition['description'] = description
                conditions.append(condition)
                click.echo(f"✓ Added condition '{condition_name}' with {len(markers)} markers")
        
        self.config['conditions'] = conditions
        click.echo(f"✓ {len(conditions)} conditions configured")
        click.echo()
    
    def _configure_quality_control(self):
        """Configure quality control options."""
        click.echo("📊 Step 7: Quality Control")
        click.echo("-" * 20)
        
        enable_qc = click.confirm("Enable quality control?", default=True)
        
        if enable_qc:
            generate_plots = click.confirm("Generate quality plots?", default=True)
            
            self.config['quality_control'] = {
                'enabled': True,
                'generate_plots': generate_plots
            }
            
            # Advanced QC settings
            if click.confirm("Configure quality thresholds?", default=False):
                thresholds = {}
                
                bad_channels_max = click.prompt(
                    "Maximum bad channels (ratio)", default=0.2, type=float
                )
                artifact_rejection_max = click.prompt(
                    "Maximum artifact rejection (ratio)", default=0.3, type=float
                )
                
                thresholds['bad_channels_max'] = bad_channels_max
                thresholds['artifact_rejection_max'] = artifact_rejection_max
                
                self.config['quality_control']['thresholds'] = thresholds
            
            click.echo("✓ Quality control configured")
        else:
            self.config['quality_control'] = {'enabled': False}
            click.echo("✓ Quality control disabled")
        
        click.echo()
    
    def _review_and_save(self, output_path: str):
        """Review configuration and save to file."""
        click.echo("📋 Step 8: Review Configuration")
        click.echo("-" * 20)
        
        # Display configuration summary
        click.echo("Configuration Summary:")
        click.echo(f"  Data format: {self.config.get('data_format', 'auto-detect')}")
        click.echo(f"  Raw data: {self.config['paths']['raw_data_dir']}")
        click.echo(f"  Results: {self.config['paths']['results_dir']}")
        click.echo(f"  Participants: {self._get_participant_summary()}")
        click.echo(f"  Stages: {len(self.config['stages'])} selected")
        click.echo(f"  Conditions: {len(self.config.get('conditions', []))}")
        click.echo(f"  Quality control: {'enabled' if self.config.get('quality_control', {}).get('enabled') else 'disabled'}")
        
        click.echo()
        
        # Show full configuration option
        if click.confirm("Show full configuration?", default=False):
            click.echo("\nFull configuration:")
            click.echo(yaml.dump(self.config, default_flow_style=False, indent=2))
        
        # Save configuration
        if click.confirm(f"Save configuration to '{output_path}'?", default=True):
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            click.echo(f"✓ Configuration saved to: {output_path}")
            
            # Validation
            click.echo("\nValidating configuration...")
            try:
                from ..utils.config_loader import load_config
                load_config(output_path)
                click.echo("✅ Configuration is valid!")
            except Exception as e:
                click.echo(f"⚠️ Configuration validation failed: {e}")
                click.echo("You may need to edit the file manually.")
        
        click.echo()
        click.echo("🎉 Configuration wizard completed!")
        click.echo(f"Next steps:")
        click.echo(f"  1. Review the configuration file: {output_path}")
        click.echo(f"  2. Validate: eeg-processor validate {output_path}")
        click.echo(f"  3. Run processing: eeg-processor process {output_path}")
    
    def _get_participant_summary(self) -> str:
        """Get summary of participant configuration."""
        participants = self.config.get('participants', 'auto')
        if participants == 'auto':
            return "auto-discovery"
        elif isinstance(participants, dict):
            return f"{len(participants)} manually specified"
        else:
            return str(participants)


def run_interactive_wizard(output_path: str) -> Dict[str, Any]:
    """Run the interactive configuration wizard."""
    wizard = InteractiveConfigWizard()
    return wizard.run_wizard(output_path)