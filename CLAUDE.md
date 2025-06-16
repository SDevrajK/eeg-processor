# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG Processor is a comprehensive EEG data processing pipeline for scientific research. It supports multiple EEG file formats (BrainVision, EDF, FIF, Curry, ANT, etc.) and provides a complete processing workflow from raw data to analysis results with built-in quality control and reporting.

## Environment Setup

**Platform**: Windows 11 with WSL2
**Python**: Use `python3` command (not `python`) OR use venv's Python directly
**Virtual Environment**: **CRITICAL - Windows venv in WSL2 uses Scripts/ not bin/**

### Installation and Setup
```bash
# ACTIVATE VIRTUAL ENVIRONMENT (REQUIRED) - Windows venv in WSL2
source .venv/Scripts/activate

# Alternative: Use venv Python directly without activation
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Development installation
python3 -m pip install -e ".[dev]"

# Install pre-commit hooks (if .pre-commit-config.yaml exists)
pre-commit install
```

### Testing
```bash
# IMPORTANT: Always activate virtual environment first - Windows venv uses Scripts/
source .venv/Scripts/activate

# Alternative: Use venv Python directly
.venv/Scripts/python.exe -m pytest tests/ -v

# Run all tests
python3 -m pytest tests/ -v

# Run specific test modules
python3 -m pytest tests/test_config_loader.py -v
python3 -m pytest tests/test_pipeline.py -v

# Run a single test
python3 -m pytest tests/test_pipeline.py::test_specific_function -v
```

### Code Quality
```bash
# IMPORTANT: Always activate virtual environment first - Windows venv uses Scripts/
source .venv/Scripts/activate

# Alternative: Use venv Python directly
.venv/Scripts/python.exe -m black src/ tests/

# Format code
python3 -m black src/ tests/

# Lint code
python3 -m flake8 src/

# Type checking
python3 -m mypy src/

# Run all pre-commit checks
pre-commit run --all-files
```

### CLI Usage
```bash
# Process data with configuration
eeg-processor process config/params.yml

# Validate configuration
eeg-processor validate config/params.yml

# Generate quality reports
eeg-processor quality-report results/

# Interactive data exploration
eeg-processor explore config/params.yml sub-01
```

## Architecture Overview

### Core Components

**EEGPipeline** (`src/eeg_processor/pipeline.py`): Main orchestrator that coordinates the entire processing workflow. Manages configuration loading, participant processing, and result generation.

**File I/O System** (`src/eeg_processor/file_io/`): Format-specific loaders for different EEG file types with automatic format detection.

**Processing Modules** (`src/eeg_processor/processing/`): Modular processing stages including filtering, artifact removal (ICA, ASR), bad channel detection, epoching, and re-referencing.

**Quality Control** (`src/eeg_processor/quality_control/`): Comprehensive tracking and reporting system that monitors processing quality throughout the pipeline.

**State Management** (`src/eeg_processor/state_management/`): Handles data flow between processing stages and manages participant data.

**Configuration System** (`src/eeg_processor/utils/config_loader.py`): YAML-based configuration with validation and type-safe parameter management using PipelineConfig dataclass.

### Key Design Patterns

- **Modular Architecture**: Each processing stage is independent and configurable
- **Format Agnostic**: Support for multiple EEG formats with automatic detection
- **Memory-Efficient**: Careful memory management with garbage collection between participants
- **Quality-First**: Built-in quality tracking and reporting throughout the pipeline
- **Dual-Mode Operation**: Support for both batch processing and interactive exploration

## Configuration

The system uses YAML configuration files with comprehensive validation. Key configuration areas:

- **Data paths**: Input/output directory specification
- **Processing parameters**: Filtering, artifact removal (ICA, ASR), epoching settings
- **Quality control**: Thresholds and reporting options
- **Dataset organization**: Optional dataset names for multi-study projects

## Testing Strategy

Tests are organized in the `tests/` directory with:
- **Unit tests**: Individual module testing
- **Integration tests**: Full pipeline testing
- **Mock data**: Minimal test files for different EEG formats in `tests/test_data/`
- **Configuration testing**: Validation of config loading and parsing

## Dependencies

Core scientific dependencies:
- **MNE-Python** (≥1.7.0): EEG processing library
- **NumPy** (≥1.21.0): Numerical computing
- **Loguru** (≥0.7.0): Advanced logging
- **PyYAML**: Configuration parsing
- **Click**: CLI framework
- **ASRpy** (≥0.0.4): Artifact Subspace Reconstruction for artifact removal

## Artifact Removal Methods

The system supports multiple artifact removal approaches:

### ICA (Independent Component Analysis)
- **Method**: `remove_artifacts` with `method: "ica"`
- **Best for**: Eye blinks, heartbeat, muscle artifacts
- **Features**: Automatic component classification with ICALabel
- **Configuration**: Various thresholds for different artifact types

### ASR (Artifact Subspace Reconstruction)
- **Method**: `remove_artifacts_asr` or `remove_artifacts` with `method: "asr"`
- **Best for**: Brief high-amplitude artifacts, motion artifacts
- **Features**: Calibration-based subspace reconstruction
- **Key Parameters**:
  - `cutoff`: Standard deviation cutoff (5-100, recommend 10-30)
  - `method`: Distance metric ("euclid" or "riemann")
  - `show_plot`: Visualization of correction
- **Configuration Example**:
```yaml
processing:
  - remove_artifacts_asr:
      cutoff: 20                    # Conservative artifact detection
      method: "euclid"              # Euclidean distance metric
      calibration_duration: 60      # Use first 60s for calibration (optional)
      show_plot: false              # No visualization in batch mode
```

### Usage Recommendations
- **ASR**: Apply early in pipeline for continuous data artifact correction
- **ICA**: Apply after ASR for remaining component-based artifacts
- **Combined approach**: Use both for robust artifact removal in noisy environments
- **Calibration**: Include 1-2 minutes of eyes-closed resting data at recording start
- **Parameters**: Use consistent ASR parameters across subjects in a study for reproducibility

## Quality Control

The system includes comprehensive quality control with:
- Automatic tracking of processing metrics
- HTML report generation with visualizations
- Configurable quality thresholds
- Both individual and group-level reporting
- **ASR-specific metrics**: Correlation preservation, variance changes, channel-wise corrections