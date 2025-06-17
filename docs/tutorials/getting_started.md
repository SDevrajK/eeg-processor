# Getting Started with EEG Processor

## Introduction

EEG Processor is a comprehensive pipeline for processing EEG data with built-in quality control, artifact rejection, and flexible configuration options. This tutorial will guide you through your first EEG processing workflow.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for development installation)

### Install from PyPI (Recommended)

```bash
pip install eeg-processor
```

### Development Installation

```bash
git clone https://github.com/yourusername/eeg-processor.git
cd eeg-processor
pip install -e .
```

### Install Optional Dependencies

```bash
# For CLI functionality
pip install eeg-processor[cli]

# For development
pip install eeg-processor[dev]

# For all optional features
pip install eeg-processor[all]
```

## Quick Start

### 1. Prepare Your Data

Organize your EEG data in a directory structure:

```
data/
├── raw/
│   ├── sub-01_task-rest_eeg.vhdr
│   ├── sub-01_task-rest_eeg.vmrk
│   ├── sub-01_task-rest_eeg.eeg
│   ├── sub-02_task-rest_eeg.vhdr
│   └── ...
└── processed/  (will be created)
```

### 2. Create a Configuration File

Create a basic configuration file `config.yml`:

```yaml
paths:
  raw_data_dir: "data/raw"
  results_dir: "data/processed"
  file_extension: ".vhdr"

participants: "auto"  # Auto-discover all files

stages:
  - load_data
  - filter
  - epoching
  - artifact_rejection
  - save_results

filtering:
  lowpass: 40
  highpass: 0.1
  notch: 50

epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

conditions:
  - name: "target"
    condition_markers: ["S1", "S11"]
  - name: "standard"
    condition_markers: ["S2", "S12"]
```

### 3. Process Your Data

#### Using the Command Line

```bash
# Validate your configuration first
eeg-processor validate config.yml

# Run a dry-run to see what will be processed
eeg-processor process config.yml --dry-run

# Process all participants
eeg-processor process config.yml

# Process a single participant
eeg-processor process config.yml --participant sub-01
```

#### Using Python

```python
from eeg_processor import EEGPipeline

# Create and run pipeline
pipeline = EEGPipeline("config.yml")
results = pipeline.run_all()

print(f"Processed {len(results)} participants")
```

### 4. Generate Quality Reports

```bash
# Generate quality control reports
eeg-processor quality-report data/processed/

# Generate PDF reports
eeg-processor quality-report data/processed/ --format pdf
```

## Understanding the Output

After processing, your directory structure will look like:

```
data/
├── raw/
│   └── ... (original data)
├── processed/
│   ├── sub-01/
│   │   ├── epochs_target.fif
│   │   ├── epochs_standard.fif
│   │   ├── evoked_target.fif
│   │   ├── evoked_standard.fif
│   │   └── processing_log.txt
│   ├── sub-02/
│   │   └── ...
│   ├── quality/
│   │   ├── quality_metrics.json
│   │   ├── quality_report.html
│   │   └── plots/
│   └── grand_averages/
│       ├── grand_avg_target.fif
│       └── grand_avg_standard.fif
```

## Next Steps

1. **Explore Interactive Mode**: Use `eeg-processor explore config.yml sub-01` for interactive data exploration
2. **Customize Processing**: Modify your configuration for specific needs
3. **Quality Control**: Review the generated quality reports
4. **Analysis**: Use the analysis interface for further data analysis

## Common Use Cases

### ERP Analysis

```yaml
# Configuration for ERP analysis
epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

artifact_rejection:
  peak_to_peak: 100e-6
  flat_threshold: 1e-6

conditions:
  - name: "P300_target"
    condition_markers: ["S1"]
    description: "Target stimuli for P300"
  - name: "P300_standard"
    condition_markers: ["S2"]
    description: "Standard stimuli for P300"
```

### Resting State Analysis

```yaml
# Configuration for resting state
epoching:
  tmin: 0
  tmax: 2.0  # 2-second epochs
  baseline: null  # No baseline correction

filtering:
  lowpass: 40
  highpass: 1.0  # Higher highpass for resting state

stages:
  - load_data
  - filter
  - bad_channels
  - epoching
  - ica  # ICA for artifact removal
  - save_results
```

### Sleep EEG Analysis

```yaml
# Configuration for sleep EEG
data_format: "edf"

epoching:
  tmin: 0
  tmax: 30  # 30-second epochs
  baseline: null

filtering:
  lowpass: 30
  highpass: 0.5
  notch: null  # No notch filter

conditions:
  - name: "wake"
    condition_markers: ["Wake"]
  - name: "nrem"
    condition_markers: ["N1", "N2", "N3"]
  - name: "rem"
    condition_markers: ["REM"]
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   ```bash
   eeg-processor validate config.yml
   ```

2. **File Not Found**
   - Check file paths in configuration
   - Ensure data files exist and are readable

3. **Memory Issues**
   - Process participants individually: `--participant sub-01`
   - Reduce epoch length or increase decimation

4. **Processing Errors**
   - Use verbose mode: `eeg-processor process config.yml --verbose`
   - Check processing logs in results directory

### Getting Help

- Use `eeg-processor --help` for CLI help
- Check the API documentation
- Review example configurations
- Submit issues on GitHub

## Configuration Templates

Generate configuration templates for different data formats:

```bash
# BrainVision data
eeg-processor create-config --format brainvision --output brainvision_config.yml

# EDF data
eeg-processor create-config --format edf --output edf_config.yml

# Interactive wizard (coming soon)
eeg-processor create-config --interactive
```

## Best Practices

1. **Always validate configuration first**
2. **Use dry-run mode to preview processing**
3. **Start with a single participant to test settings**
4. **Review quality reports after processing**
5. **Keep original data separate from processed results**
6. **Use version control for configuration files**
7. **Document any custom processing steps**

## Advanced Features

- **Custom Processing Stages**: Define your own processing functions
- **Parallel Processing**: Use `--parallel` for faster processing
- **Interactive Exploration**: Explore data step-by-step
- **Quality Thresholds**: Set automatic quality control criteria
- **Batch Processing**: Process multiple datasets efficiently

Continue to the [Advanced Tutorial](advanced_tutorial.md) for more complex workflows and customization options.