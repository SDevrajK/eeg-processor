# EEG Processor API Documentation

## Overview

The EEG Processor provides a comprehensive API for processing EEG data with built-in quality control, artifact rejection, and flexible configuration options.

## Core Classes

### EEGPipeline

The main class for EEG data processing.

```python
from eeg_processor import EEGPipeline

# Initialize with config file
pipeline = EEGPipeline("config/processing_params.yml")

# Initialize with config dictionary
config_dict = {
    "paths": {"raw_data_dir": "data/raw", "results_dir": "data/processed"},
    "filtering": {"lowpass": 40, "highpass": 0.1}
}
pipeline = EEGPipeline(config_dict)
```

#### Methods

##### `__init__(config_path=None)`
Initialize the pipeline.

**Parameters:**
- `config_path` (str or dict, optional): Path to YAML config file or config dictionary

##### `load_config(config_path)`
Load configuration from file or dictionary.

**Parameters:**
- `config_path` (str or dict): Configuration source

**Returns:**
- `self`: Pipeline instance for method chaining

##### `run_all()`
Process all participants defined in configuration.

**Returns:**
- `dict`: Results dictionary with participant IDs as keys

##### `run_participant(participant_id)`
Process a single participant.

**Parameters:**
- `participant_id` (str): Participant identifier

**Returns:**
- `dict`: Processing results for the participant

##### `apply_stage(data, stage_name, condition=None, **params)`
Apply individual processing stage interactively.

**Parameters:**
- `data`: MNE data object (Raw, Epochs, or Evoked)
- `stage_name` (str): Name of processing stage
- `condition` (dict, optional): Condition dictionary
- `**params`: Stage-specific parameters

**Returns:**
- Processed MNE data object

##### `get_analysis_interface()`
Get interface for loading and analyzing processed data.

**Returns:**
- `AnalysisInterface`: Analysis interface object

### Configuration Loading

#### `load_config(config_path, override_params=None)`
Load and validate configuration.

```python
from eeg_processor import load_config

# Load from file
config = load_config("config/params.yml")

# Load with overrides
config = load_config("config/params.yml", {
    "filtering": {"lowpass": 30}
})

# Load from dictionary
config = load_config({
    "paths": {"raw_data_dir": "data"},
    "filtering": {"lowpass": 40}
})
```

**Parameters:**
- `config_path` (str or dict): Path to config file or config dictionary
- `override_params` (dict, optional): Parameters to override

**Returns:**
- `PipelineConfig`: Validated configuration object

**Raises:**
- `ConfigurationError`: If config file cannot be loaded
- `ValidationError`: If config validation fails

## Quality Control

### QualityTracker

Tracks quality metrics during processing.

```python
from eeg_processor.quality_control import QualityTracker

tracker = QualityTracker("/path/to/results")
tracker.track_participant_start("sub-01")
tracker.track_channel_quality(raw_data)
tracker.save_metrics()
```

#### Methods

##### `__init__(results_dir)`
Initialize quality tracker.

**Parameters:**
- `results_dir` (str or Path): Directory for saving quality metrics

##### `track_participant_start(participant_id)`
Initialize tracking for a participant.

**Parameters:**
- `participant_id` (str): Participant identifier

##### `track_channel_quality(raw)`
Track channel quality metrics.

**Parameters:**
- `raw`: MNE Raw object

##### `track_artifact_rejection(epochs)`
Track artifact rejection metrics.

**Parameters:**
- `epochs`: MNE Epochs object

##### `save_metrics()`
Save quality metrics to file.

### QualityReporter

Generate quality reports and visualizations.

```python
from eeg_processor.quality_control import QualityReporter

reporter = QualityReporter(quality_tracker)
report = reporter.generate_summary_report()
```

#### Methods

##### `__init__(quality_tracker)`
Initialize quality reporter.

**Parameters:**
- `quality_tracker`: QualityTracker instance

##### `generate_summary_report()`
Generate comprehensive quality report.

**Returns:**
- `dict`: Quality report dictionary

##### `assess_overall_quality()`
Assess overall data quality.

**Returns:**
- `dict`: Quality assessment with scores and flags

### Quality Control Functions

#### `generate_quality_reports(results_dir)`
Generate quality reports for all participants.

```python
from eeg_processor.quality_control import generate_quality_reports

reports = generate_quality_reports("/path/to/results")
```

**Parameters:**
- `results_dir` (str or Path): Results directory containing quality data

**Returns:**
- `dict`: Reports for all participants

## File I/O

### Supported Formats

The EEG Processor supports multiple file formats through format-specific loaders:

- **BrainVision** (.vhdr, .vmrk, .eeg)
- **EDF/EDF+** (.edf)
- **FIF** (.fif) - MNE-Python native format
- **EEGLAB** (.set)
- **Neuroscan** (.cnt, .eeg)
- **Curry** (.cdt)

### File Loading

```python
from eeg_processor.file_io import load_raw

# Automatic format detection
raw = load_raw("/path/to/data.vhdr")

# Explicit format specification
raw = load_raw("/path/to/data.edf", format="edf")
```

#### `load_raw(file_path, format=None, **kwargs)`
Load raw EEG data.

**Parameters:**
- `file_path` (str): Path to data file
- `format` (str, optional): File format ('brainvision', 'edf', 'fif', etc.)
- `**kwargs`: Format-specific loading parameters

**Returns:**
- `mne.io.Raw`: Loaded raw data object

## Processing Stages

### Available Stages

1. **load_data**: Load raw EEG data
2. **montage**: Apply electrode montage
3. **filter**: Apply frequency filters
4. **bad_channels**: Detect and interpolate bad channels
5. **reref**: Re-reference data
6. **epoching**: Create epochs around events
7. **artifact_rejection**: Reject artifacts
8. **ica**: Independent Component Analysis
9. **time_frequency**: Time-frequency analysis
10. **evoked**: Compute evoked responses
11. **save_results**: Save processed data

### Stage Parameters

Each stage accepts specific parameters through the configuration:

```yaml
filtering:
  lowpass: 40
  highpass: 0.1
  notch: 50

epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

artifact_rejection:
  peak_to_peak: 100e-6
  flat_threshold: 1e-6
```

## Analysis Interface

### AnalysisInterface

Interface for loading and analyzing processed data.

```python
# Get analysis interface
analysis = pipeline.get_analysis_interface()

# Load processed data
epochs = analysis.load_epochs("sub-01", "target")
evoked = analysis.load_evoked("sub-01", "target")

# Compute grand averages
grand_avg = analysis.compute_grand_average("target", 
                                         participants=["sub-01", "sub-02"])

# Generate plots
analysis.plot_condition_comparison(["target", "standard"])
```

#### Methods

##### `load_epochs(participant_id, condition)`
Load epochs for participant and condition.

**Parameters:**
- `participant_id` (str): Participant identifier
- `condition` (str): Condition name

**Returns:**
- `mne.Epochs`: Loaded epochs

##### `load_evoked(participant_id, condition)`
Load evoked response.

**Parameters:**
- `participant_id` (str): Participant identifier
- `condition` (str): Condition name

**Returns:**
- `mne.Evoked`: Loaded evoked response

##### `compute_grand_average(condition, participants=None)`
Compute grand average across participants.

**Parameters:**
- `condition` (str): Condition name
- `participants` (list, optional): Participant IDs to include

**Returns:**
- `mne.Evoked`: Grand average evoked response

##### `plot_condition_comparison(conditions, save_path=None)`
Generate condition comparison plots.

**Parameters:**
- `conditions` (list): List of condition names
- `save_path` (str, optional): Path to save plot

## Error Handling

### Custom Exceptions

```python
from eeg_processor.utils.exceptions import (
    EEGProcessorError,
    ConfigurationError,
    DataLoadError,
    ProcessingError,
    QualityControlError,
    FileFormatError,
    ValidationError
)
```

#### Exception Hierarchy

- `EEGProcessorError`: Base exception
  - `ConfigurationError`: Configuration issues
  - `DataLoadError`: Data loading problems
    - `FileFormatError`: File format issues
  - `ProcessingError`: Processing failures
  - `QualityControlError`: Quality control issues
  - `ValidationError`: Validation failures

### Error Handling Examples

```python
try:
    pipeline = EEGPipeline("config.yml")
    results = pipeline.run_all()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DataLoadError as e:
    print(f"Data loading error: {e}")
except ProcessingError as e:
    print(f"Processing error: {e}")
except EEGProcessorError as e:
    print(f"General EEG processor error: {e}")
```

## Configuration Schema

### Required Fields

```yaml
paths:
  raw_data_dir: "path/to/raw/data"    # Required
  results_dir: "path/to/results"      # Optional, defaults to "results"
```

### Optional Fields

```yaml
paths:
  interim_dir: "path/to/interim"      # Intermediate files
  figures_dir: "path/to/figures"      # Generated figures
  file_extension: ".vhdr"             # File extension for auto-discovery

participants:                         # Can be dict or "auto"
  sub-01: "sub-01_task-rest_eeg.vhdr"
  sub-02: "sub-02_task-rest_eeg.vhdr"

dataset_name: "MyExperiment"          # Dataset identifier

stages:                               # Processing stages to run
  - load_data
  - filter
  - epoching
  - save_results

conditions:                           # Experimental conditions
  - name: "target"
    condition_markers: ["S1", "S2"]
  - name: "standard"
    condition_markers: ["S3", "S4"]

quality_control:                      # Quality control settings
  enabled: true
  generate_plots: true
  thresholds:
    bad_channels_max: 0.2
    artifact_rejection_max: 0.3
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.py`: Common use cases
- `configuration_examples.py`: Configuration examples for different data types
- `advanced_processing.py`: Advanced processing techniques
- `quality_control_examples.py`: Quality control workflows

## Command Line Interface

```bash
# Process with config file
eeg-processor process config/params.yml

# Process single participant
eeg-processor process config/params.yml --participant sub-01

# Generate quality reports
eeg-processor quality-report results/

# Validate configuration
eeg-processor validate-config config/params.yml
```

See `eeg-processor --help` for full CLI documentation.