# EEG Processor - Stage Documentation

*Generated on 2025-06-18 16:57:22*

This documentation provides comprehensive information about all available
processing stages in the EEG Processor pipeline.

## Quick Links

- [📋 Quick Reference](quick-reference.md) - Overview of all stages
- [🔧 Troubleshooting](troubleshooting.md) - Common issues and solutions

## Processing Categories

EEG processing stages are organized into logical categories:

### [Data preparation and event management](data_handling.md)

*3 stages available*

- [`adjust_events`](data_handling/adjust_events.md) - Adjust event times with optional in-place operation
- [`correct_triggers`](data_handling/correct_triggers.md) - Correct incorrectly coded triggers in EEG data
- [`crop`](data_handling/crop.md) - Crop raw EEG data using either absolute times or event markers

### [Signal filtering and artifact removal](preprocessing.md)

*8 stages available*

- [`clean_rawdata_asr`](preprocessing/clean_rawdata_asr.md) - Clean raw EEG data using Artifact Subspace Reconstruction (ASR)
- [`compute_eog`](preprocessing/compute_eog.md) - Compute HEOG/VEOG from electrode pairs
- [`detect_bad_channels`](preprocessing/detect_bad_channels.md) - Detect and optionally interpolate bad channels using MNE's LOF method
- [`filter`](preprocessing/filter.md) - Apply filtering with optional in-place operation
- [`remove_artifacts`](preprocessing/remove_artifacts.md) - Remove artifacts using Independent Component Analysis (ICA)
- [`remove_blinks_emcp`](preprocessing/remove_blinks_emcp.md) - Remove blink artifacts using MNE's EOGRegression method
- [`rereference`](preprocessing/rereference.md) - Apply rereferencing to raw EEG data with robust exclude handling

### [Experimental condition processing and epoching](condition_handling.md)

*2 stages available*

- [`epoch`](condition_handling/epoch.md) - Create epochs from Raw data - always returns new Epochs object
- [`segment_condition`](condition_handling/segment_condition.md) - Segment Raw data based on condition markers with optional in-place operation

### [Analysis of epoched data](post_epoching.md)

*3 stages available*

- [`time_frequency`](post_epoching/time_frequency.md) - Compute averaged time-frequency representation from epochs
- [`time_frequency_average`](post_epoching/time_frequency_average.md) - Convert RawTFR to AverageTFR by averaging across time dimension
- [`time_frequency_raw`](post_epoching/time_frequency_raw.md) - Compute baseline power spectrum from continuous raw data

### [Data visualization and inspection](visualization.md)

*1 stages available*

- [`view`](visualization/view.md) - Unified plotting interface

## Using This Documentation

### Command Line Help

```bash
# List all available stages
eeg-processor list-stages

# Get help for a specific stage
eeg-processor help-stage filter

# Show examples
eeg-processor help-stage filter --examples
```

### Configuration Usage

Each stage can be used in your YAML configuration file:

```yaml
stages:
  # Simple stage (uses defaults)
  - filter
  
  # Stage with parameters
  - filter:
      l_freq: 0.1
      h_freq: 40
      notch: 50
```

---

*This documentation is automatically generated from the source code.*
*For the most up-to-date information, use the CLI help commands.*