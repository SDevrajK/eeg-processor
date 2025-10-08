#!/usr/bin/env python3
"""Command line interface for EEG Processor."""

import sys
import copy
import click
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

from .pipeline import EEGPipeline
from .utils.config_loader import load_config
from .utils.exceptions import EEGProcessorError, ConfigurationError, ValidationError
from .utils.stage_documentation import StageDocumentationExtractor, format_stage_help, format_stage_list
from .utils.preset_manager import PresetManager
from .utils.doc_generator import generate_stage_documentation
from .utils.schema_generator import generate_json_schemas
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
@click.option('--detailed', '-d', is_flag=True, help='Show detailed validation report')
@click.option('--output', '-o', help='Save fixed configuration to file (requires --fix)')
@click.pass_context
def validate(ctx, config_path: str, fix: bool, detailed: bool, output: Optional[str]):
    """Validate configuration file with smart error detection.
    
    CONFIG_PATH: Path to YAML configuration file to validate
    
    This command provides comprehensive validation with helpful suggestions
    and automatic fixes for common configuration issues.
    
    Examples:
        eeg-processor validate config/params.yml
        eeg-processor validate config/params.yml --detailed
        eeg-processor validate config/params.yml --fix
        eeg-processor validate config/params.yml --fix --output fixed_config.yml
    """
    try:
        from .utils.config_validator import validate_config_file, ConfigValidator
        
        # Run smart validation
        issues, fixed_content = validate_config_file(config_path, auto_fix=fix)
        
        if not issues:
            # No issues found - also try loading with original loader for completeness
            try:
                config = load_config(config_path)
                if not ctx.obj['quiet']:
                    click.echo("✅ " + click.style("Configuration is valid!", fg='green', bold=True))
                    click.echo(f"  📁 Data format: {getattr(config, 'data_format', 'auto-detected')}")
                    click.echo(f"  👥 Participants: {len(config.participants) if hasattr(config, 'participants') else 'auto'}")
                    click.echo(f"  ⚙️  Stages: {len(config.stages) if hasattr(config, 'stages') else 0}")
                    click.echo(f"  🎯 Conditions: {len(config.conditions) if hasattr(config, 'conditions') else 0}")
                    click.echo(f"  📊 Results dir: {getattr(config, 'results_dir', 'Not specified')}")
            except Exception as load_error:
                click.echo(f"⚠️  Configuration syntax is valid but loading failed: {load_error}")
                sys.exit(1)
        else:
            # Issues found - display report
            validator = ConfigValidator()
            report = validator.generate_validation_report(issues)
            
            click.echo(report)
            
            # Check if there are critical errors
            errors = [i for i in issues if i.level == 'error']
            warnings = [i for i in issues if i.level == 'warning']
            
            if errors:
                click.echo(f"\n❌ Found {len(errors)} error(s) that must be fixed")
                if fix and fixed_content:
                    click.echo("🔧 Automatic fixes have been applied")
                elif not fix:
                    click.echo("💡 Use --fix to attempt automatic repairs")
            
            if warnings:
                click.echo(f"⚠️  Found {len(warnings)} warning(s) - recommended to address")
            
            # Handle auto-fix output
            if fix and fixed_content:
                output_path = output or config_path
                
                if output_path != config_path or click.confirm(f"Overwrite {config_path} with fixes?"):
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    click.echo(f"💾 Fixed configuration saved to: {output_path}")
                    
                    # Re-validate to show improvement
                    new_issues, _ = validate_config_file(output_path)
                    remaining_errors = [i for i in new_issues if i.level == 'error']
                    
                    if not remaining_errors:
                        click.echo("✅ All critical errors have been fixed!")
                    else:
                        click.echo(f"⚠️  {len(remaining_errors)} error(s) still need manual attention")
            
            # Show detailed report if requested
            if detailed:
                click.echo("\n" + "="*60)
                click.echo("DETAILED VALIDATION REPORT")
                click.echo("="*60)
                
                for issue in issues:
                    click.echo(f"\n{issue.level.upper()}: {issue.section}.{issue.field}")
                    click.echo(f"  Message: {issue.message}")
                    if issue.suggestion:
                        click.echo(f"  Suggestion: {issue.suggestion}")
                    if issue.auto_fix is not None:
                        click.echo(f"  Auto-fix: {issue.auto_fix}")
            
            # Exit with error code if there are critical issues
            if errors and not (fix and fixed_content):
                sys.exit(1)
    
    except FileNotFoundError:
        click.echo(f"❌ Configuration file not found: {config_path}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error during validation: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
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
@click.option('--preset', '-p', help='Use a preset configuration (e.g., basic-erp, minimal, artifact-removal)')
@click.option('--minimal', is_flag=True, help='Create minimal configuration (equivalent to --preset minimal)')
@click.option('--template', is_flag=True, help='Create full template with all options and comments')
@click.pass_context
def create_config(ctx, data_format: Optional[str], output: str, interactive: bool, 
                 preset: Optional[str], minimal: bool, template: bool):
    """Create a new configuration file.
    
    Multiple modes available:
    - Preset mode: Use pre-configured processing pipelines
    - Minimal mode: Bare minimum configuration for testing
    - Template mode: Full template with all options and documentation
    - Interactive mode: Guided configuration wizard
    
    Examples:
        eeg-processor create-config --preset basic-erp
        eeg-processor create-config --preset artifact-removal --output noisy_data.yml
        eeg-processor create-config --minimal
        eeg-processor create-config --template --format brainvision
        eeg-processor create-config --interactive
    """
    try:
        # Validate conflicting options
        mode_count = sum([bool(preset), minimal, template, interactive])
        if mode_count > 1:
            click.echo("Error: Cannot combine --preset, --minimal, --template, and --interactive options", err=True)
            sys.exit(1)
        
        if preset or minimal:
            # Preset-based configuration
            preset_name = "minimal" if minimal else preset
            _create_preset_config(preset_name, data_format, output, ctx)
        elif template:
            # Full template configuration
            _create_template_config(data_format, output)
        elif interactive:
            # Interactive wizard
            _interactive_config_wizard(output)
        else:
            # Default: create basic template
            _create_template_config(data_format, output)
        
        if not ctx.obj['quiet']:
            click.echo(f"Configuration created: {output}")
            
            # Provide helpful next steps
            if preset or minimal:
                click.echo(f"Next steps:")
                click.echo(f"  1. Edit {output} to set your data paths")
                click.echo(f"  2. Customize participant lists and conditions")
                click.echo(f"  3. Run: eeg-processor validate {output}")
                click.echo(f"  4. Process: eeg-processor process {output}")
    
    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


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


@cli.command('list-stages')
@click.option('--category', '-c', help='Filter by category (data_handling, preprocessing, condition_handling, post_epoching, visualization)')
@click.pass_context
def list_stages(ctx, category: Optional[str]):
    """List all available processing stages.
    
    Shows processing stages organized by category with brief descriptions.
    
    Categories:
        data_handling     - Data preparation and event management
        preprocessing     - Signal filtering and artifact removal  
        condition_handling - Experimental condition processing and epoching
        post_epoching     - Analysis of epoched data
        visualization     - Data visualization and inspection
    
    Examples:
        eeg-processor list-stages
        eeg-processor list-stages --category preprocessing
        eeg-processor list-stages -c condition_handling
    """
    try:
        extractor = StageDocumentationExtractor()
        categorized_stages = extractor.get_all_stages()
        
        # Validate category filter
        if category and category not in categorized_stages:
            valid_categories = ", ".join(sorted(categorized_stages.keys()))
            click.echo(f"Invalid category '{category}'. Valid categories: {valid_categories}", err=True)
            sys.exit(1)
        
        formatted_output = format_stage_list(categorized_stages, category)
        click.echo(formatted_output)
        
    except Exception as e:
        click.echo(f"Error listing stages: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command('help-stage')
@click.argument('stage_name', required=False)
@click.option('--examples', '-e', is_flag=True, help='Include usage examples in output')
@click.option('--all', '-a', is_flag=True, help='Show help for all stages')
@click.pass_context
def help_stage(ctx, stage_name: Optional[str], examples: bool, all: bool):
    """Show detailed help for a processing stage.
    
    Displays comprehensive documentation including parameters, return values,
    notes, and optionally usage examples for processing stages.
    
    STAGE_NAME: Name of the stage to get help for (optional if using --all)
    
    Examples:
        eeg-processor help-stage filter
        eeg-processor help-stage detect_bad_channels --examples
        eeg-processor help-stage --all
    """
    try:
        extractor = StageDocumentationExtractor()
        
        if all:
            # Show help for all stages
            categorized_stages = extractor.get_all_stages()
            
            click.echo("Comprehensive Stage Documentation")
            click.echo("=" * 50)
            
            for category, stages in categorized_stages.items():
                category_desc = extractor.category_descriptions.get(category, category.replace('_', ' ').title())
                click.echo(f"\n{category_desc.upper()}")
                click.echo("=" * len(category_desc))
                
                for stage in sorted(stages):
                    stage_info = extractor.get_stage_info(stage)
                    if stage_info:
                        formatted_help = format_stage_help(stage_info, include_examples=examples)
                        click.echo(formatted_help)
                        click.echo()  # Blank line between stages
            
        elif stage_name:
            # Show help for specific stage
            stage_info = extractor.get_stage_info(stage_name)
            
            if not stage_info:
                click.echo(f"Stage '{stage_name}' not found.", err=True)
                click.echo("Use 'eeg-processor list-stages' to see available stages.", err=True)
                sys.exit(1)
            
            formatted_help = format_stage_help(stage_info, include_examples=examples)
            click.echo(formatted_help)
            
        else:
            # No stage specified and --all not used
            click.echo("Please specify a stage name or use --all to show help for all stages.", err=True)
            click.echo("Use 'eeg-processor list-stages' to see available stages.", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error getting stage help: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command('generate-docs')
@click.option('--output', '-o', default='docs/stages', help='Output directory for documentation')
@click.option('--format', 'doc_format', type=click.Choice(['markdown', 'html']), 
              default='markdown', help='Documentation format')
@click.pass_context
def generate_docs(ctx, output: str, doc_format: str):
    """Generate comprehensive documentation for all processing stages.
    
    Creates detailed markdown documentation including:
    - Individual stage documentation with parameters and examples
    - Category overviews and pipeline guidance
    - Quick reference and troubleshooting guides
    
    Examples:
        eeg-processor generate-docs
        eeg-processor generate-docs --output docs/my-stages
        eeg-processor generate-docs --format html
    """
    try:
        if not ctx.obj['quiet']:
            click.echo(f"Generating {doc_format} documentation...")
            click.echo(f"Output directory: {output}")
        
        if doc_format == 'markdown':
            generated_files = generate_stage_documentation(output)
            
            if not ctx.obj['quiet']:
                click.echo(f"Generated {len(generated_files)} documentation files:")
                
                # Show generated files by category
                categories = {}
                for filename in generated_files.keys():
                    if '/' in filename:
                        category = filename.split('/')[0]
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(filename)
                    else:
                        if 'root' not in categories:
                            categories['root'] = []
                        categories['root'].append(filename)
                
                # Display organized output
                if 'root' in categories:
                    click.echo(f"  Main files:")
                    for filename in sorted(categories['root']):
                        click.echo(f"    - {filename}")
                
                for category, files in categories.items():
                    if category != 'root':
                        click.echo(f"  {category.replace('_', ' ').title()}:")
                        for filename in sorted(files):
                            stage_name = filename.split('/')[-1].replace('.md', '')
                            click.echo(f"    - {stage_name}")
                
                click.echo(f"")
                click.echo(f"View documentation: {output}/README.md")
                click.echo(f"Quick reference: {output}/quick-reference.md")
                click.echo(f"Troubleshooting: {output}/troubleshooting.md")
        
        elif doc_format == 'html':
            # HTML generation not implemented yet
            click.echo(f"HTML documentation generation not yet implemented.", err=True)
            click.echo(f"Use --format markdown for now.", err=True)
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error generating documentation: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command('generate-schemas')
@click.option('--output', '-o', default='schemas', help='Output directory for JSON schemas')
@click.pass_context
def generate_schemas(ctx, output: str):
    """Generate JSON schemas for configuration validation and IDE support.
    
    Creates comprehensive JSON schemas for:
    - Main configuration file validation
    - Individual stage parameter schemas
    - Preset configuration schemas
    - Combined stages schema for validation
    
    These schemas enable:
    - IDE autocomplete and validation
    - Programmatic configuration validation
    - API documentation generation
    - External tool integration
    
    Examples:
        eeg-processor generate-schemas
        eeg-processor generate-schemas --output my-schemas
    """
    try:
        if not ctx.obj['quiet']:
            click.echo(f"Generating JSON schemas...")
            click.echo(f"Output directory: {output}")
        
        schemas = generate_json_schemas(output)
        
        if not ctx.obj['quiet']:
            click.echo(f"Generated {len(schemas)} JSON schema files:")
            
            # Organize output by type
            main_schemas = ['main_config', 'stages', 'preset']
            stage_schemas = [k for k in schemas.keys() if k not in main_schemas]
            
            click.echo(f"  Main schemas:")
            for schema_name in main_schemas:
                if schema_name in schemas:
                    filename = {
                        'main_config': 'eeg_processor_config.json',
                        'stages': 'stages_schema.json', 
                        'preset': 'preset_schema.json'
                    }.get(schema_name, f'{schema_name}.json')
                    click.echo(f"    - {filename}")
            
            if stage_schemas:
                click.echo(f"  Stage schemas ({len(stage_schemas)} stages):")
                # Show first few stage schemas
                for schema_name in sorted(stage_schemas)[:5]:
                    click.echo(f"    - stage_{schema_name}.json")
                if len(stage_schemas) > 5:
                    click.echo(f"    ... and {len(stage_schemas)-5} more stage schemas")
            
            click.echo(f"")
            click.echo(f"Usage:")
            click.echo(f"  - IDE validation: Point your editor to {output}/eeg_processor_config.json")
            click.echo(f"  - Programmatic validation: Load schemas for config validation")
            click.echo(f"  - API docs: Use schemas for automatic documentation generation")
    
    except Exception as e:
        click.echo(f"Error generating schemas: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command('list-presets')
@click.option('--category', '-c', help='Filter by category (basic, advanced, research)')
@click.option('--tag', '-t', help='Filter by tag (e.g., erp, artifact-removal)')
@click.pass_context
def list_presets(ctx, category: Optional[str], tag: Optional[str]):
    """List all available configuration presets.
    
    Shows available presets organized by category with descriptions and metadata.
    
    Categories:
        basic     - Simple, robust presets for common use cases
        advanced  - Complex presets with comprehensive processing
        research  - Specialized presets for specific research applications
    
    Examples:
        eeg-processor list-presets
        eeg-processor list-presets --category basic
        eeg-processor list-presets --tag artifact-removal
    """
    try:
        preset_manager = PresetManager()
        
        if tag:
            # Filter by tag
            matching_presets = preset_manager.list_presets_by_tag(tag)
            if not matching_presets:
                click.echo(f"No presets found with tag '{tag}'")
                return
            
            click.echo(f"Presets tagged with '{tag}':")
            click.echo("=" * 50)
            for preset_info in matching_presets:
                _display_preset_info(preset_info)
        else:
            # List by category
            presets_by_category = preset_manager.get_available_presets(category)
            
            if not presets_by_category:
                if category:
                    click.echo(f"No presets found in category '{category}'")
                else:
                    click.echo("No presets available")
                return
            
            click.echo("Available Configuration Presets")
            click.echo("=" * 50)
            
            for cat_name, presets in presets_by_category.items():
                click.echo(f"\n{cat_name.upper()}")
                click.echo("-" * len(cat_name))
                
                for preset_info in presets:
                    _display_preset_info(preset_info, brief=True)
            
            click.echo(f"\nUse 'eeg-processor create-config --preset <name>' to create configuration")
            click.echo(f"Use 'eeg-processor list-presets --category <category>' to filter by category")
    
    except Exception as e:
        click.echo(f"Error listing presets: {e}", err=True)
        if ctx.obj['verbose']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _display_preset_info(preset_info, brief: bool = False):
    """Display preset information in a formatted way."""
    if brief:
        # Brief format for list view
        desc = preset_info.description[:80] + "..." if len(preset_info.description) > 80 else preset_info.description
        click.echo(f"  {preset_info.name:<20} - {desc}")
        if preset_info.tags:
            tags_str = ", ".join(preset_info.tags[:3])  # Show first 3 tags
            if len(preset_info.tags) > 3:
                tags_str += f" (+{len(preset_info.tags)-3} more)"
            click.echo(f"    Tags: {tags_str}")
    else:
        # Detailed format
        click.echo(f"\n{preset_info.name} (v{preset_info.version})")
        click.echo(f"  Category: {preset_info.category}")
        click.echo(f"  Description: {preset_info.description}")
        if preset_info.use_cases:
            click.echo(f"  Use cases: {', '.join(preset_info.use_cases[:3])}")
        if preset_info.tags:
            click.echo(f"  Tags: {', '.join(preset_info.tags)}")
        if preset_info.recommended_channels:
            click.echo(f"  Recommended channels: {preset_info.recommended_channels}")


def _create_preset_config(preset_name: str, data_format: Optional[str], output: str, ctx):
    """Create configuration from a preset."""
    try:
        preset_manager = PresetManager()
        
        # Load preset
        preset_data = preset_manager.load_preset(preset_name)
        
        # Get config template from preset
        config_template = preset_data['config_template']
        
        # Apply data format if specified
        if data_format:
            # Update file extension based on format
            format_extensions = {
                'brainvision': '.vhdr',
                'edf': '.edf',
                'fif': '.fif',
                'eeglab': '.set'
            }
            if 'paths' not in config_template:
                config_template['paths'] = {}
            config_template['paths']['file_extension'] = format_extensions.get(data_format, '.vhdr')
        
        # Write configuration file with section dividers
        _write_preset_config_with_comments(config_template, preset_data['metadata'], output)
        
        if not ctx.obj['quiet']:
            # Show preset information
            click.echo(f"Created configuration from preset: {preset_data['metadata']['name']}")
            click.echo(f"Description: {preset_data['metadata']['description']}")
            
    except ConfigurationError as e:
        # Show available presets
        preset_manager = PresetManager()
        available_presets = []
        presets_by_category = preset_manager.get_available_presets()
        for presets in presets_by_category.values():
            available_presets.extend([p.name for p in presets])
        
        click.echo(f"Error: {e}", err=True)
        click.echo(f"Available presets: {', '.join(available_presets)}", err=True)
        click.echo("Use 'eeg-processor list-presets' to see all available presets", err=True)
        sys.exit(1)


def _write_preset_config_with_comments(config_template: Dict[str, Any], metadata: Dict[str, Any], output_path: str):
    """Write configuration file with proper section dividers and comments."""
    config = copy.deepcopy(config_template)
    
    # Prepare user-friendly values
    if 'paths' in config:
        config['paths']['raw_data_dir'] = config['paths'].get('raw_data_dir', 'data/raw')
        config['paths']['results_dir'] = config['paths'].get('results_dir', 'results')
    
    if config.get('participants') == "auto":
        config['participants'] = 'auto'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header comment
        f.write(f"# ========================================\n")
        f.write(f"# EEG PROCESSOR CONFIGURATION\n")
        f.write(f"# ========================================\n")
        f.write(f"# Generated from preset: {metadata['name']}\n")
        f.write(f"# Description: {metadata['description']}\n")
        f.write(f"# Version: {metadata['version']}\n")
        f.write(f"#\n")
        f.write(f"# REQUIRED CUSTOMIZATIONS:\n")
        f.write(f"#   1. Set your data paths below\n")
        f.write(f"#   2. List your participant files\n")
        f.write(f"#   3. Define your experimental conditions\n")
        f.write(f"#\n")
        f.write(f"# Generated by EEG Processor CLI\n")
        f.write(f"# ========================================\n\n")
        
        # 1. Study Information Section
        if 'study' in config:
            f.write("# ========================================\n")
            f.write("# STUDY INFORMATION\n")
            f.write("# ========================================\n")
            f.write(yaml.dump({'study': config['study']}, default_flow_style=False, indent=2))
            f.write("\n")
        
        # 2. Data Paths Section
        f.write("# ========================================\n")
        f.write("# DATA PATHS (REQUIRED - CUSTOMIZE THESE)\n")
        f.write("# ========================================\n")
        if 'paths' in config:
            # Add inline comments for required fields
            paths_section = "paths:\n"
            paths_section += f"  raw_data_dir: \"{config['paths']['raw_data_dir']}\"  # REQUIRED: Path to your raw EEG files\n"
            paths_section += f"  results_dir: \"{config['paths']['results_dir']}\"   # REQUIRED: Where to save processed data\n"
            
            # Add other path fields
            for key, value in config['paths'].items():
                if key not in ['raw_data_dir', 'results_dir']:
                    if isinstance(value, str):
                        paths_section += f"  {key}: \"{value}\"\n"
                    else:
                        paths_section += f"  {key}: {value}\n"
            
            f.write(paths_section)
            f.write("\n")
        
        # 3. Dataset Organization (if present)
        if 'dataset_name' in config:
            f.write("# ========================================\n")
            f.write("# DATASET ORGANIZATION (OPTIONAL)\n")
            f.write("# ========================================\n")
            f.write(yaml.dump({'dataset_name': config['dataset_name']}, default_flow_style=False, indent=2))
            f.write("\n")
        
        # 4. Participants Section
        f.write("# ========================================\n")
        f.write("# PARTICIPANTS (REQUIRED - CUSTOMIZE THIS)\n")
        f.write("# ========================================\n")
        f.write("# List your participant files or use 'auto' to process all files\n")
        f.write("# Examples:\n")
        f.write("#   - sub-01.vhdr\n")
        f.write("#   - sub-02.vhdr\n")
        f.write("# Or simply use: auto\n")
        f.write(yaml.dump({'participants': config.get('participants', 'auto')}, default_flow_style=False, indent=2))
        f.write("\n")
        
        # 5. Processing Pipeline Section
        f.write("# ========================================\n")
        f.write(f"# PROCESSING PIPELINE - {metadata['name'].upper()}\n")
        f.write("# ========================================\n")
        if 'stages' in config:
            f.write("# Processing stages in order - each stage processes the output of the previous\n")
            f.write(yaml.dump({'stages': config['stages']}, default_flow_style=False, indent=2))
            f.write("\n")
        
        # 6. Experimental Conditions Section
        f.write("# ========================================\n")
        f.write("# EXPERIMENTAL CONDITIONS (REQUIRED - CUSTOMIZE)\n")
        f.write("# ========================================\n")
        f.write("# Define your experimental conditions and their event markers\n")
        f.write("# Replace the example markers below with your actual experiment markers\n")
        if 'conditions' in config:
            f.write(yaml.dump({'conditions': config['conditions']}, default_flow_style=False, indent=2))
        else:
            example_conditions = [
                {
                    'name': 'Condition1',
                    'condition_markers': ['S1', 1],
                    'description': 'Replace with your actual condition'
                }
            ]
            f.write(yaml.dump({'conditions': example_conditions}, default_flow_style=False, indent=2))
        f.write("\n")
        
        # 7. Output Settings Section
        if 'output' in config:
            f.write("# ========================================\n")
            f.write("# OUTPUT SETTINGS (OPTIONAL)\n")
            f.write("# ========================================\n")
            f.write(yaml.dump({'output': config['output']}, default_flow_style=False, indent=2))


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
    from .utils.config_wizard import run_enhanced_wizard
    run_enhanced_wizard(output)


if __name__ == '__main__':
    cli()