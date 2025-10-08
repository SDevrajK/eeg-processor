# detect_bad_channels

Detect and optionally interpolate bad channels using MNE's LOF method. Uses Local Outlier Factor to identify channels that are outliers compared to their spatial neighbors, avoiding false positives from physiological artifacts like eyeblinks.

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `n_neighbors`

- **Type:** `int`
- **Required:** No
- **Default:** `10`

### `threshold`

- **Type:** `float`
- **Required:** No
- **Default:** `1.5`

### `interpolate`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

### `verbose`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `show_plot`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `plot_duration`

- **Type:** `float`
- **Required:** No
- **Default:** `2.0`

### `plot_start`

- **Type:** `float`
- **Required:** No
- **Default:** `5.0`

## Returns

- Raw object with bad channels detected and optionally interpolated

## Usage Examples

### Basic Usage

```yaml
stages:
  - detect_bad_channels
```

### With Custom Parameters

```yaml
stages:
  - detect_bad_channels:
      threshold: 1.5
      n_neighbors: 8
```

### Command Line Help

```bash
eeg-processor help-stage detect_bad_channels
```

## Notes

- LOF method is robust against physiological artifacts like eyeblinks
- Automatically excludes non-EEG channels from detection
- Stores detailed metrics in raw._bad_channel_metrics for quality tracking
- Uses LOF re-detection to validate interpolation success

## Common Issues

**Too many channels detected as bad**

Lower the threshold parameter or check if your data has systemic issues. More than 20% bad channels often indicates recording problems.

**Good channels marked as bad**

Increase the threshold parameter or check the n_neighbors setting. Dense electrode arrays may need higher n_neighbors values.

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)