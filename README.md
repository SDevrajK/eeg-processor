# EEG Processor

A comprehensive EEG processing pipeline with quality control and reporting capabilities.

## Features

- **Multi-format support**: BrainVision (.vhdr), EDF, FIF, Curry, ANT, Neuroscan, EEGLAB
- **Flexible processing pipeline**: Configurable stages for filtering, artifact removal, epoching, and more
- **Quality control**: Comprehensive tracking and reporting of processing quality metrics
- **Dataset organization**: Automatic separation of results by dataset name
- **Interactive exploration**: CLI and programmatic interfaces
- **Comprehensive reporting**: HTML reports with visualizations and statistics

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg-processor.git
cd eeg-processor

# Install in development mode
pip install -e .

# Or install with CLI support
pip install -e ".[cli]"

# Or install with development dependencies
pip install -e ".[dev]"
```

### From Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```python
from eeg_processor import EEGPipeline

# Load configuration and run pipeline
pipeline = EEGPipeline("config/processing_params.yml")
pipeline.run()

# Generate quality reports
pipeline.generate_quality_reports()
```

### 2. Configuration Example

Create a YAML configuration file:

```yaml
# Dataset name for result organization
dataset_name: "my_experiment"

# Paths
paths:
  raw_data: "data/raw/"
  results_dir: "results/"
  participants: ["sub01.vhdr", "sub02.vhdr"]

# Conditions
conditions:
  - name: "Baseline"
    condition_markers: [10, 20]
  - name: "Task"
    condition_markers: [30, 40]

# Processing stages
stages:
  - filter:
      l_freq: 1
      h_freq: 100
  - detect_bad_channels:
      interpolate: true
  - rereference:
      method: 'average'
  - epoch:
      tmin: -0.2
      tmax: 0.8
      baseline: [-0.2, 0]
```

### 3. CLI Usage

```bash
# Run batch processing
eeg-processor batch config/processing_params.yml

# Interactive exploration
eeg-processor explore config/processing_params.yml participant_01
```

## Configuration

### Dataset Organization

Use the `dataset_name` field to organize results by group:

```yaml
dataset_name: "control_group"  # Results go to results_dir/control_group/
```

### Supported Processing Stages

- `filter`: Frequency filtering
- `detect_bad_channels`: Bad channel detection and interpolation
- `rereference`: Re-referencing (average, REST, etc.)
- `blink_artifact`: Artifact removal (ICA, etc.)
- `epoch`: Event-related epoching
- `time_frequency`: Time-frequency analysis

### File Formats

Supported input formats:
- BrainVision (`.vhdr`, `.eeg`, `.vmrk`)
- EDF (`.edf`)
- FIF (`.fif`)
- Curry (`.dat`, `.dap`, `.rs3`)
- ANT (`.cnt`)
- Neuroscan (`.eeg`)
- EEGLAB (`.set`)

## Output Structure

```
results_dir/
├── dataset_name/           # If dataset_name is specified
│   ├── interim/           # Intermediate processing results
│   ├── processed/         # Final processed data
│   ├── figures/          # Generated plots
│   └── quality/          # Quality control reports
│       ├── quality_summary_report.html
│       ├── individual_reports/
│       └── quality_metrics.json
```

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/
flake8 src/
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{eeg_processor,
  title={EEG Processor: A Comprehensive EEG Processing Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/eeg-processor}
}
```

## Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues](https://github.com/yourusername/eeg-processor/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/eeg-processor/discussions)

## Changelog

### v0.1.0 (2025-01-XX)
- Initial release
- Multi-format EEG support
- Quality control and reporting
- Dataset organization features