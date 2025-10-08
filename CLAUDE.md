# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the EEG Processor codebase.

## Development Guidelines

All development work on this project must follow these core principles:

1. **Write minimal code** - Implement only what is needed to solve the immediate problem, avoiding over-engineering
2. **Prioritize conciseness** - Favor clean, compact solutions over backwards-compatibility when they conflict
3. **Remove legacy code** - Delete unused functions, deprecated methods, and dead code paths immediately
4. **Use descriptive naming** - Variables, functions, and classes should clearly indicate their purpose without requiring comments
5. **Follow standard conventions** - Adhere to PEP 8, type hints, and established Python scientific computing patterns

**Scientific Research Specific Guidelines:**

6. **Document scientific rationale** - Include brief comments explaining the scientific reasoning behind parameter choices, thresholds, and methodological decisions
7. **Validate inputs and outputs** - Always include sanity checks for data dimensions, value ranges, and expected data types to catch analysis errors early
8. **Prioritize reproducibility** - Use fixed random seeds, save processing parameters, and ensure identical inputs produce identical outputs across runs

These guidelines apply to all code contributions, workflow templates, and documentation.

## Project Overview

EEG Processor is a comprehensive EEG data processing pipeline for scientific research. It supports multiple EEG file formats (BrainVision, EDF, FIF, Curry, ANT, etc.) and provides a complete processing workflow from raw data to analysis results with built-in quality control and reporting.

## Critical Environment Requirements

### Platform: WSL2 with Miniconda
- **Environment**: Miniconda environment called `eeg-processor`
- **Python execution**: Use `python` directly when environment is activated
- **Environment activation**: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg-processor`

### Essential Commands

```bash
# ENVIRONMENT ACTIVATION
source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg-processor

# PYTHON EXECUTION (With activated environment)
python -m <module>
python script.py
python -c "import sys; print(sys.version)"

# TESTING
python -m pytest tests/ -v
python -m pytest tests/test_specific.py::test_function -v

# CODE QUALITY
python -m black src/ tests/
python -m flake8 src/
python -m mypy src/

# CLI USAGE
python -c "from src.eeg_processor.cli import cli; cli(['--help'])"
python -c "from src.eeg_processor.cli import cli; cli(['list-stages'])"
```

## Configuration System

### CLI Commands for Configuration Discovery

```bash
# List all processing stages by category
python -c "from src.eeg_processor.cli import cli; cli(['list-stages'])"

# Get detailed help for specific stages  
python -c "from src.eeg_processor.cli import cli; cli(['help-stage', 'filter'])"

# List available presets
python -c "from src.eeg_processor.cli import cli; cli(['list-presets'])"

# Create configuration from presets
python -c "from src.eeg_processor.cli import cli; cli(['create-config', '--preset', 'basic-erp', '--output', 'my_config.yml'])"

# Validate configuration
python -c "from src.eeg_processor.cli import cli; cli(['validate', 'config/my_config.yml'])"
```

### Available Processing Stages

**DATA PREPARATION AND EVENT MANAGEMENT**
- `adjust_events` - Adjust event times with optional in-place operation
- `correct_triggers` - Correct incorrectly coded triggers in EEG data  
- `crop` - Crop raw EEG data using either absolute times or event markers

**SIGNAL FILTERING AND ARTIFACT REMOVAL**
- `filter` - Apply filtering with optional in-place operation
- `compute_eog` - Compute HEOG/VEOG from electrode pairs
- `detect_bad_channels` - Detect and optionally interpolate bad channels using MNE's LOF method
- `rereference` - Apply rereferencing to raw EEG data with robust exclude handling
- `remove_artifacts` - Remove artifacts using Independent Component Analysis (ICA)
- `remove_blinks_emcp` - Remove blink artifacts using EOG regression methods
- `clean_rawdata_asr` - Clean raw EEG data using Artifact Subspace Reconstruction (ASR)

**EXPERIMENTAL CONDITION PROCESSING AND EPOCHING**
- `epoch` - Create epochs from Raw data - always returns new Epochs object
- `segment_condition` - Segment Raw data based on condition markers with optional in-place operation

**ANALYSIS OF EPOCHED DATA**
- `time_frequency` - Compute averaged time-frequency representation from epochs
- `time_frequency_average` - Convert RawTFR to AverageTFR by averaging across time dimension
- `time_frequency_raw` - Compute baseline power spectrum from continuous raw data

**DATA VISUALIZATION AND INSPECTION**
- `view` - Unified plotting interface

### Current Configuration Structure

All configurations use YAML format with comprehensive validation:

```yaml
# Study metadata (REQUIRED)
study:
  name: "My_EEG_Study"                    # Required
  dataset: "experiment1"                  # Optional: creates results/experiment1/
  description: "EEG data processing"      # Optional
  researcher: "Your Lab"                  # Optional

# Data paths (REQUIRED)
paths:
  raw_data: "data/raw/"                   # Directory containing raw EEG files
  results: "results/"                     # Where outputs are saved
  file_extension: ".vhdr"                 # File extension (.vhdr, .edf, .fif, .set)

# Participants (REQUIRED)
participants:
  # Simple format (most common)
  - "sub-01.vhdr"
  - "sub-02.vhdr"
  
  # Detailed format with metadata (alternative)
  # sub-01:
  #   file: "sub-01.vhdr"
  #   age: 25
  #   conditions: ["baseline", "task"]

# Processing pipeline (REQUIRED)
processing:
  - filter: {l_freq: 0.1, h_freq: 40}
  - detect_bad_channels: {interpolate: true}
  - rereference: {method: "average"}
  - remove_artifacts: {method: "ica"}
  - epoch: {tmin: -0.2, tmax: 0.8, baseline: [-0.2, 0]}

# Experimental conditions (REQUIRED for ERP analysis)
conditions:
  - name: "Standard"
    condition_markers: [1, 11, "S1", "S11"]
  - name: "Target"
    condition_markers: [2, 12, "S2", "S12"]

# Advanced: Time-windowed segmentation (for changing triggers)
# Use when trigger codes change during recording:
# conditions:
#   - name: "Early Phase"
#     markers: [100, 199]      # [start_marker, end_marker] for segmentation
#     t_min: 0                 # Search from beginning (seconds)
#     t_max: 300               # Search until 5 minutes
#   - name: "Late Phase"
#     markers: [200, 299]      # Different markers after time point
#     t_min: 300               # Search from 5 minutes onwards
#     t_max: 600               # Search until 10 minutes

# Output settings (optional)
output:
  save_intermediates: false
  figure_format: "png"
  dpi: 150
  create_report: true
```

### Legacy Configuration Support

The system maintains backward compatibility with the previous structure:
- `raw_data_dir` → `paths.raw_data`
- `results_dir` → `paths.results` 
- `stages` → `processing`
- `dataset_name` → `study.dataset`

Legacy configurations will continue to work but should be migrated to the new structure.

### Available Presets

**BASIC**
- `basic-erp` - Basic ERP processing (filter → bad channels → rereference → epoch)
- `minimal` - Minimal processing for testing (filter → epoch)

**ADVANCED** 
- `artifact-removal` - Comprehensive artifact removal (ASR + EMCP + ICA pipeline)

## Artifact Removal Methods

### ICA (Independent Component Analysis)
```yaml
stages:
  - remove_artifacts: {method: "ica"}
```
- **Best for**: Eye blinks, heartbeat, muscle artifacts
- **Features**: Automatic component classification with ICALabel

### ASR (Artifact Subspace Reconstruction)
```yaml
stages:
  - clean_rawdata_asr: {cutoff: 20, method: "euclid"}
```
- **Best for**: Brief high-amplitude artifacts, motion artifacts
- **Key Parameters**: `cutoff` (10-30 recommended), `method` ("euclid"/"riemann")

### EMCP (Eye Movement Correction Procedures)
```yaml
stages:
  - remove_blinks_emcp: {method: "eog_regression", eog_channels: ["HEOG", "VEOG"]}
  # OR
  - remove_blinks_emcp: {method: "gratton_coles", eog_channels: ["HEOG", "VEOG"]}
```
- **Methods**: `eog_regression` (standard) or `gratton_coles` (reference-agnostic)
- **Best for**: Blink artifact correction when EOG channels available

### Recommended Pipeline Order
1. **ASR** - Early continuous data artifact correction
2. **EMCP** - Blink correction (after bad channels, before rereferencing)  
3. **ICA** - Component-based artifacts (after EMCP)

## Architecture Overview

### Core Components
- **EEGPipeline** (`src/eeg_processor/pipeline.py`) - Main orchestrator
- **File I/O System** (`src/eeg_processor/file_io/`) - Format-specific loaders
- **Processing Modules** (`src/eeg_processor/processing/`) - Modular processing stages
- **Quality Control** (`src/eeg_processor/quality_control/`) - Comprehensive tracking and reporting
- **State Management** (`src/eeg_processor/state_management/`) - Data flow management
- **Configuration System** (`src/eeg_processor/utils/config_loader.py`) - YAML-based configuration

### Enhanced Configuration Tools
- **Stage Documentation** (`src/eeg_processor/utils/stage_documentation.py`) - Stage help system
- **Preset Manager** (`src/eeg_processor/utils/preset_manager.py`) - Preset management
- **Config Validator** (`src/eeg_processor/utils/config_validator.py`) - Smart validation
- **Schema System** (`schemas/`) - JSON schemas for all stages and configurations

### Key Design Patterns
- **Modular Architecture**: Independent, configurable processing stages
- **Format Agnostic**: Multi-format support with automatic detection
- **Memory-Efficient**: Careful memory management with garbage collection
- **Quality-First**: Built-in quality tracking throughout pipeline
- **Exception Hierarchy**: Structured error handling (ConfigurationError, ValidationError, etc.)

## Dependencies

**Core Scientific Stack**
- **MNE-Python** (≥1.7.0) - EEG processing library
- **NumPy** (≥1.21.0) - Numerical computing  
- **ASRpy** (≥0.0.4) - Artifact Subspace Reconstruction

**Development Tools**
- **Loguru** (≥0.7.0) - Advanced logging
- **PyYAML** - Configuration parsing
- **Click** - CLI framework
- **pytest** - Testing framework

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_config_loader.py -v
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest --cov=src --cov-report=html tests/
```

## Quality Control

The system includes comprehensive quality control:
- **Automatic tracking** - Processing metrics collected throughout pipeline
- **HTML report generation** - Interactive visualizations and statistics  
- **Quality flagging** - Configurable thresholds for automatic assessment
- **Multi-level reporting** - Individual participant and group-level summaries

### Quality Control Components
- **QualityTracker** - Central tracking system
- **QualityReporter** - Report generation and aggregation
- **HTMLGenerator** - Interactive HTML report creation
- **QualityFlagging** - Automated quality assessment