#!/usr/bin/env python3
"""Command line interface for EEG Processor."""

import sys
import click
import yaml
from pathlib import Path
from typing import Optional, List

from .pipeline import EEGPipeline
from .utils.config_loader import load_config
from .utils.exceptions import EEGProcessorError, ConfigurationError, ValidationError
from .quality_control import generate_quality_reports


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """EEG Processor - Comprehensive EEG data processing pipeline.
    
    A robust pipeline for processing EEG data with quality control,
    artifact rejection, and flexible configuration options.
    
    Examples:
        eeg-processor process config/params.yml
        eeg-processor validate config/params.yml
        eeg-processor quality-report results/
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    
    # Configure logging based on verbosity
    if quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--participant', '-p', help='Process only specific participant')
@click.option('--output', '-o', help='Override output directory')
@click.option('--stages', help='Comma-separated list of stages to run')
@click.option('--dry-run', is_flag=True, help='Validate config and show what would be processed')
@click.option('--parallel', '-j', type=int, help='Number of parallel jobs')
@click.pass_context
def process(ctx, config_path: str, participant: Optional[str], output: Optional[str], 
           stages: Optional[str], dry_run: bool, parallel: Optional[int]):
    """Process EEG data using configuration file.
    
    CONFIG_PATH: Path to YAML configuration file
    
    Examples:
        eeg-processor process config/params.yml
        eeg-processor process config/params.yml --participant sub-01
        eeg-processor process config/params.yml --stages "load_data,filter,epoching"
        eeg-processor process config/params.yml --dry-run
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Apply command line overrides
        overrides = {}
        if output:
            overrides['paths'] = {'results_dir': output}
        if stages:
            overrides['stages'] = [s.strip() for s in stages.split(',')]
        if parallel:
            overrides['parallel'] = {'n_jobs': parallel}
        
        if overrides:
            config = load_config(config_path, overrides)
        
        if dry_run:
            _show_processing_plan(config, participant)
            return
        
        # Initialize pipeline
        pipeline = EEGPipeline(config)
        
        if not ctx.obj['quiet']:
            click.echo(f"Processing with config: {config_path}")
            if participant:
                click.echo(f"Processing participant: {participant}")
            else:
                click.echo(f"Processing {len(config.participants)} participants")
        
        # Run processing
        if participant:
            results = pipeline.run_participant(participant)
            if not ctx.obj['quiet']:
                click.echo(f"✓ Completed processing for {participant}")
        else:
            results = pipeline.run_all()
            if not ctx.obj['quiet']:
                click.echo(f"✓ Completed processing for {len(results)} participants")
        
        if not ctx.obj['quiet']:
            click.echo(f"Results saved to: {config.results_dir}")
    
    except (ConfigurationError, ValidationError) as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except EEGProcessorError as e:
        click.echo(f"Processing error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--fix', is_flag=True, help='Attempt to fix common issues automatically')
@click.pass_context
def validate(ctx, config_path: str, fix: bool):
    """Validate configuration file.
    
    CONFIG_PATH: Path to YAML configuration file to validate
    
    Examples:
        eeg-processor validate config/params.yml
        eeg-processor validate config/params.yml --fix
    """
    try:
        config = load_config(config_path)
        
        if not ctx.obj['quiet']:
            click.echo("✓ Configuration is valid")
            click.echo(f"  Data format: {getattr(config, 'data_format', 'auto-detected')}")
            click.echo(f"  Participants: {len(config.participants)}")
            click.echo(f"  Stages: {len(config.stages)}")
            click.echo(f"  Conditions: {len(config.conditions)}")
            click.echo(f"  Results dir: {config.results_dir}")
    
    except (ConfigurationError, ValidationError) as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        
        if fix:
            click.echo("Attempting to fix issues...")
            # TODO: Implement auto-fix functionality
            click.echo("Auto-fix not yet implemented")
        
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Unexpected error during validation: {e}", err=True)
        sys.exit(1)


@cli.command('quality-report')
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory for reports')
@click.option('--format', 'report_format', type=click.Choice(['html', 'pdf', 'json']), 
              default='html', help='Report format')
@click.pass_context
def quality_report(ctx, results_dir: str, output: Optional[str], report_format: str):
    """Generate quality control reports.
    
    RESULTS_DIR: Directory containing processing results
    
    Examples:
        eeg-processor quality-report results/
        eeg-processor quality-report results/ --format pdf
        eeg-processor quality-report results/ --output reports/
    """
    try:
        results_path = Path(results_dir)
        
        if not ctx.obj['quiet']:
            click.echo(f"Generating quality reports for: {results_path}")
        
        reports = generate_quality_reports(results_path, format=report_format)
        
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            # TODO: Save reports to specified output directory
        
        if not ctx.obj['quiet']:
            click.echo(f"✓ Generated {len(reports)} quality reports")
            if output:
                click.echo(f"Reports saved to: {output}")
    
    except Exception as e:
        click.echo(f"Error generating reports: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('participant_id')
@click.option('--stage', help='Start at specific processing stage')
@click.pass_context
def explore(ctx, config_path: str, participant_id: str, stage: Optional[str]):
    """Launch interactive exploration for a participant.
    
    CONFIG_PATH: Path to YAML configuration file
    PARTICIPANT_ID: ID of participant to explore
    
    Examples:
        eeg-processor explore config/params.yml sub-01
        eeg-processor explore config/params.yml sub-01 --stage filtering
    """
    try:
        config = load_config(config_path)
        pipeline = EEGPipeline(config)
        
        if not ctx.obj['quiet']:
            click.echo(f"Loading participant {participant_id} for interactive exploration...")
        
        # Load participant data
        participant_data = pipeline.load_participant_data(participant_id)
        
        # Prepare interactive environment
        banner = f"""
EEG Processor Interactive Exploration
====================================
Participant: {participant_id}
Config: {config_path}
Data loaded as 'raw'
Pipeline available as 'pipeline'

Available methods:
- pipeline.apply_stage(raw, 'filter', lowpass=40)
- pipeline.apply_stage(raw, 'epoching', tmin=-0.2, tmax=0.8)
- raw.plot()
- raw.plot_psd()

Type 'help()' for more information.
        """
        
        # Start IPython shell
        try:
            import IPython
            IPython.embed(banner1=banner, user_ns={
                'raw': participant_data,
                'pipeline': pipeline,
                'config': config
            })
        except ImportError:
            click.echo("IPython not available. Starting basic Python shell...")
            import code
            code.interact(banner=banner, local={
                'raw': participant_data,
                'pipeline': pipeline,
                'config': config
            })
    
    except Exception as e:
        click.echo(f"Error launching exploration: {e}", err=True)
        sys.exit(1)


@cli.command('create-config')
@click.option('--format', 'data_format', type=click.Choice(['brainvision', 'edf', 'fif', 'eeglab']),
              help='Data format to configure for')
@click.option('--output', '-o', default='config.yml', help='Output configuration file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive configuration wizard')
@click.pass_context
def create_config(ctx, data_format: Optional[str], output: str, interactive: bool):
    """Create a new configuration file.
    
    Examples:
        eeg-processor create-config --format brainvision
        eeg-processor create-config --interactive
        eeg-processor create-config --output my_config.yml
    """
    if interactive:
        _interactive_config_wizard(output)
    else:
        _create_template_config(data_format, output)
    
    if not ctx.obj['quiet']:
        click.echo(f"✓ Configuration created: {output}")


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--participant', '-p', help='Show info for specific participant')
@click.pass_context
def info(ctx, results_dir: str, participant: Optional[str]):
    """Show information about processed results.
    
    RESULTS_DIR: Directory containing processing results
    
    Examples:
        eeg-processor info results/
        eeg-processor info results/ --participant sub-01
    """
    try:
        results_path = Path(results_dir)
        
        # TODO: Implement results inspection
        if not ctx.obj['quiet']:
            click.echo(f"Results directory: {results_path}")
            if participant:
                click.echo(f"Participant: {participant}")
                # Show participant-specific info
            else:
                # Show overview of all results
                pass
    
    except Exception as e:
        click.echo(f"Error reading results: {e}", err=True)
        sys.exit(1)


def _show_processing_plan(config, participant: Optional[str]):
    """Show what would be processed in dry-run mode."""
    click.echo("Processing Plan:")
    click.echo("=" * 50)
    click.echo(f"Configuration: Valid")
    click.echo(f"Data directory: {config.raw_data_dir}")
    click.echo(f"Results directory: {config.results_dir}")
    click.echo(f"File extension: {config.file_extension}")
    
    if participant:
        click.echo(f"Participants: {participant} (single)")
    else:
        click.echo(f"Participants: {len(config.participants)} total")
        for i, p in enumerate(config.participants[:5]):  # Show first 5
            click.echo(f"  - {p}")
        if len(config.participants) > 5:
            click.echo(f"  ... and {len(config.participants) - 5} more")
    
    click.echo(f"Processing stages ({len(config.stages)}):")
    for stage in config.stages:
        click.echo(f"  - {stage}")
    
    if config.conditions:
        click.echo(f"Conditions ({len(config.conditions)}):")
        for condition in config.conditions:
            click.echo(f"  - {condition['name']}")


def _create_template_config(data_format: Optional[str], output: str):
    """Create a template configuration file."""
    template = {
        "paths": {
            "raw_data_dir": "data/raw",
            "results_dir": "data/processed",
            "file_extension": ".vhdr" if data_format == "brainvision" else ".edf"
        },
        "participants": "auto",
        "stages": [
            "load_data",
            "filter",
            "epoching",
            "artifact_rejection",
            "save_results"
        ],
        "filtering": {
            "lowpass": 40,
            "highpass": 0.1,
            "notch": 50
        },
        "epoching": {
            "tmin": -0.2,
            "tmax": 0.8,
            "baseline": [-0.2, 0]
        },
        "conditions": [
            {
                "name": "condition1",
                "condition_markers": ["S1", "S2"]
            }
        ]
    }
    
    if data_format:
        template["data_format"] = data_format
    
    with open(output, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)


def _interactive_config_wizard(output: str):
    """Interactive configuration wizard."""
    from .utils.interactive_config import run_interactive_wizard
    run_interactive_wizard(output)


if __name__ == '__main__':
    cli()