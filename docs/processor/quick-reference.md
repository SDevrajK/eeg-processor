# EEG Processor - Quick Reference

Quick overview of all available processing stages.

## Data preparation and event management

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| `adjust_events` | Adjust event times with optional in-place operation | `raw`, `shift_ms` |
| `correct_triggers` | Correct incorrectly coded triggers in EEG data | `raw`, `condition` |
| `crop` | Crop raw EEG data using either absolute times or event markers | `raw`, `t_min` |

## Signal filtering and artifact removal

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| `clean_rawdata_asr` | Clean raw EEG data using Artifact Subspace Reconstruction (ASR) | `raw`, `cutoff` |
| `compute_eog` | Compute HEOG/VEOG from electrode pairs | `raw`, `heog_pair` |
| `detect_bad_channels` | Detect and optionally interpolate bad channels using MNE's LOF method | `raw`, `n_neighbors` |
| `filter` | Apply filtering with optional in-place operation | `raw`, `l_freq` |
| `remove_artifacts` | Remove artifacts using Independent Component Analysis (ICA) | `raw`, `n_components` |
| `remove_blinks_emcp` | Remove blink artifacts using MNE's EOGRegression method | `raw`, `eog_channels` |
| `rereference` | Apply rereferencing to raw EEG data with robust exclude handling | `raw`, `method` |

## Experimental condition processing and epoching

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| `epoch` | Create epochs from Raw data - always returns new Epochs object | `raw`, `condition` |
| `segment_condition` | Segment Raw data based on condition markers with optional in-place operation | `raw`, `condition` |

## Analysis of epoched data

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| `time_frequency` | Compute averaged time-frequency representation from epochs | `epochs`, `freq_range` |
| `time_frequency_average` | Convert RawTFR to AverageTFR by averaging across time dimension | `raw_tfr`, `method` |
| `time_frequency_raw` | Compute baseline power spectrum from continuous raw data | `raw`, `freq_range` |

## Data visualization and inspection

| Stage | Description | Key Parameters |
|-------|-------------|----------------|
| `view` | Unified plotting interface | `stage_name`, `kwargs` |

## Common Pipeline Examples

### Basic ERP Pipeline
```yaml
stages:
  - filter:
      l_freq: 0.1
      h_freq: 40
  - detect_bad_channels
  - rereference:
      method: average
  - epoch:
      tmin: -0.2
      tmax: 0.8
```

### Artifact Removal Pipeline
```yaml
stages:
  - filter:
      l_freq: 1.0
      h_freq: 40
  - clean_rawdata_asr:
      cutoff: 20
  - remove_blinks_emcp:
      method: eog_regression
      eog_channels: ['HEOG', 'VEOG']
  - remove_artifacts:
      method: ica
```

[🏠 Back to Main Index](README.md)