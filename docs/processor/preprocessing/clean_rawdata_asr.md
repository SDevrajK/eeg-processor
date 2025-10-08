# clean_rawdata_asr

Clean raw EEG data using Artifact Subspace Reconstruction (ASR). ASR is an intermediate data cleaning step designed to be applied after bad channel detection/interpolation but before ICA. It identifies and corrects brief high-amplitude artifacts by reconstructing corrupted signal subspaces using a calibration-based approach. **Recommended Pipeline Position:** 1. Bad channel detection and interpolation 2. ASR data cleaning (this function) ← Intermediate step 3. ICA artifact removal (for remaining component-based artifacts)

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `cutoff`

- **Type:** `int | float`
- **Required:** No
- **Default:** `20`

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"euclid"`

### `blocksize`

- **Type:** `int | None`
- **Required:** No
- **Default:** `None`

### `window_length`

- **Type:** `float`
- **Required:** No
- **Default:** `0.5`

### `window_overlap`

- **Type:** `float`
- **Required:** No
- **Default:** `0.66`

### `max_dropout_fraction`

- **Type:** `float`
- **Required:** No
- **Default:** `0.1`

### `min_clean_fraction`

- **Type:** `float`
- **Required:** No
- **Default:** `0.25`

### `calibration_duration`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `show_plot`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `plot_duration`

- **Type:** `float`
- **Required:** No
- **Default:** `10.0`

### `plot_start`

- **Type:** `float`
- **Required:** No
- **Default:** `5.0`

### `verbose`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

## Returns

- Raw object with ASR artifact correction applied

## Usage Examples

### Basic Usage

```yaml
stages:
  - clean_rawdata_asr
```

### With Custom Parameters

```yaml
stages:
  - clean_rawdata_asr:
      cutoff: 20
      method: "euclid"
```

### Command Line Help

```bash
eeg-processor help-stage clean_rawdata_asr
```

## Notes

- *Why ASR as Intermediate Step:**
- ASR corrects transient high-amplitude artifacts (motion, electrode pops, etc.)
- These artifacts can interfere with ICA decomposition quality
- ASR preserves underlying neural signals better than simple rejection
- Cleaner data leads to better ICA component separation
- *Calibration Approach:**
- **Automatic inline**: Uses beginning of provided data for calibration
- **Custom duration**: Specify calibration_duration for explicit calibration length
- ASRpy automatically identifies clean segments within the specified duration
- *Calibration Requirements:**
- Minimum 30 seconds of relatively clean data
- Recommended 60+ seconds for optimal performance
- Data should be high-pass filtered (≥0.5 Hz) before ASR
- At least min_clean_fraction (25%) must be artifact-free
- *Quality Tracking:**
- Stores detailed metrics in raw._asr_metrics for research QC
- Includes calibration quality, correlation preservation, variance changes
- ASR preserves brain signals while removing transient artifacts
- *Research Recommendations:**
- Apply after bad channel detection/interpolation but before ICA
- Use consistent parameters across subjects in a study
- Include 1-2 minutes of eyes-closed resting data at recording start
- Validate results with correlation analysis (should be >0.8)
- References:
- Mullen et al. (2015). Real-time neuroimaging and cognitive monitoring
- using wearable dry EEG. IEEE Trans Biomed Eng, 62(11), 2553-2567.
- Kothe & Jung (2016). Artifact removal techniques for EEG recordings.
- Chang et al. (2020). Evaluation of artifact subspace reconstruction for
- automatic artifact removal in single-trial analysis of ERPs. NeuroImage.

## Related Stages

- [`remove_artifacts`](../preprocessing/remove_artifacts.md)
- [`remove_blinks_emcp`](../preprocessing/remove_blinks_emcp.md)

## Common Issues

**Over-correction of data**

Increase the cutoff parameter. Values too low (< 10) may remove valid data. Start with cutoff=20 and adjust based on your data quality.

**Insufficient artifact removal**

Decrease the cutoff parameter, but be careful not to go below 10. Also ensure you have sufficient calibration data.

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)