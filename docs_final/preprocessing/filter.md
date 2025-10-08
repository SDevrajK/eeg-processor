# filter

Apply filtering with optional in-place operation

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `l_freq`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `h_freq`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `notch`

- **Type:** `bool | float | list | None`
- **Required:** No
- **Default:** `None`

### `filter_kwargs`

- **Type:** `Any`
- **Required:** Yes

## Usage Examples

### Basic Usage

```yaml
stages:
  - filter
```

### With Custom Parameters

```yaml
stages:
  - filter:
      l_freq: 0.1
      h_freq: 40
      notch: 50
```

### Command Line Help

```bash
eeg-processor help-stage filter
```

## Related Stages

- [`remove_artifacts`](../preprocessing/remove_artifacts.md)
- [`clean_rawdata_asr`](../preprocessing/clean_rawdata_asr.md)

## Common Issues

**Filtering removes too much data**

Check your frequency ranges. High-pass filters above 1 Hz may remove important slow components. Low-pass filters below 30 Hz may remove important frequency content for some analyses.

**Edge artifacts**

Filtering can introduce artifacts at the beginning and end of recordings. Consider cropping data or using longer recordings.

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)