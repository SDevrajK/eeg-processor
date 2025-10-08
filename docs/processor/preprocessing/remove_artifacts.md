# remove_artifacts

Remove artifacts using Independent Component Analysis (ICA). Combines automatic classification (ICALabel) with correlation-based detection to identify and remove eye blinks, muscle artifacts, cardiac artifacts, and line noise components.

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `n_components`

- **Type:** `float | int`
- **Required:** No
- **Default:** `15`

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"infomax"`

### `eog_channels`

- **Type:** `list | None`
- **Required:** No
- **Default:** `['VEOG', 'HEOG']`

### `ecg_channels`

- **Type:** `list | None`
- **Required:** No
- **Default:** `None`

### `auto_classify`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `use_arci`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `muscle_threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `0.8`

### `eye_threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `0.8`

### `heart_threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `0.8`

### `line_noise_threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `0.8`

### `arci_cardiac_freq_range`

- **Type:** `float`
- **Required:** No
- **Default:** `(0.6, 1.7)`

### `arci_regularity_threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `0.4`

### `plot_components`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `enable_manual`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `decim`

- **Type:** `int | None`
- **Required:** No
- **Default:** `None`

### `random_state`

- **Type:** `int`
- **Required:** No
- **Default:** `42`

### `verbose`

- **Type:** `bool | str | None`
- **Required:** No
- **Default:** `None`

## Returns

- Raw object with artifacts removed

## Usage Examples

### Basic Usage

```yaml
stages:
  - remove_artifacts
```

### With Custom Parameters

```yaml
stages:
  - remove_artifacts:
      n_components: 1.0
      method: "example"
```

### Command Line Help

```bash
eeg-processor help-stage remove_artifacts
```

## Notes

- Stores detailed metrics in raw._ica_metrics for quality tracking
- Combines multiple detection methods for robust artifact identification
- Supports both automatic and manual component selection
- ARCI method requires proper filtering (notch + 0.1-30 Hz recommended)
- Prints detailed ICLabel probability table for all components (if auto_classify=True)

## Dependencies

This stage should typically be run after:

- `filter`

## Related Stages

- [`clean_rawdata_asr`](../preprocessing/clean_rawdata_asr.md)
- [`remove_blinks_emcp`](../preprocessing/remove_blinks_emcp.md)

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)