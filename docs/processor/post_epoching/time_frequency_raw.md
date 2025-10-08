# time_frequency_raw

Compute baseline power spectrum from continuous raw data

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `freq_range`

- **Type:** `list`
- **Required:** No
- **Default:** `[1, 50]`

### `n_freqs`

- **Type:** `int`
- **Required:** No
- **Default:** `20`

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"welch"`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Spectrum object (time-averaged power)

## Usage Examples

### Basic Usage

```yaml
stages:
  - time_frequency_raw
```

### With Custom Parameters

```yaml
stages:
  - time_frequency_raw:
      freq_range: [1, 2, 3]
      n_freqs: 10
```

### Command Line Help

```bash
eeg-processor help-stage time_frequency_raw
```

---

[← Back to Post Epoching](../post_epoching.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)