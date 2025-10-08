# remove_blinks_emcp

Remove blink artifacts using MNE's EOGRegression method. This method uses MNE's EOGRegression class to identify and remove blink artifacts from EEG data. **IMPORTANT: For EEG data, apply the desired reference (typically average reference) before using this method, as recommended by MNE.**

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `eog_channels`

- **Type:** `list`
- **Required:** No
- **Default:** `['HEOG', 'VEOG']`

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

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Raw object with blink artifacts removed

## Usage Examples

### Basic Usage

```yaml
stages:
  - remove_blinks_emcp
```

### With Custom Parameters

```yaml
stages:
  - remove_blinks_emcp:
      method: "eog_regression"
      eog_channels: ['HEOG', 'VEOG']
```

### Command Line Help

```bash
eeg-processor help-stage remove_blinks_emcp
```

## Notes

- **Requires proper EEG reference (typically average) before application**
- Generates _emcp_metrics for quality tracking
- Should be used after re-referencing stage in processing pipeline
- Supports both Raw and Epochs data types

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)