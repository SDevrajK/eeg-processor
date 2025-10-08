# crop

Crop raw EEG data using either absolute times or event markers. Parameters ---------- crop_before : str | int | None Event code (e.g., 51, 'Stimulus/S 51', 'Stimulus/S  1') crop_after : str | int | None Event code to crop after last occurrence

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `t_min`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `t_max`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `crop_before`

- **Type:** `str | int | None`
- **Required:** No
- **Default:** `None`

### `crop_after`

- **Type:** `str | int | None`
- **Required:** No
- **Default:** `None`

### `segment_start`

- **Type:** `str | int | None`
- **Required:** No
- **Default:** `None`

### `segment_end`

- **Type:** `str | int | None`
- **Required:** No
- **Default:** `None`

### `show`

- **Type:** `bool`
- **Required:** No
- **Default:** `None`

### `padded`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

## Usage Examples

### Basic Usage

```yaml
stages:
  - crop
```

### With Custom Parameters

```yaml
stages:
  - crop:
      t_min: 1.0
      t_max: 1.0
```

### Command Line Help

```bash
eeg-processor help-stage crop
```

---

[← Back to Data Handling](../data_handling.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)