# time_frequency

Compute averaged time-frequency representation from epochs

## Parameters

### `epochs`

- **Type:** `Epochs`
- **Required:** Yes

### `freq_range`

- **Type:** `list`
- **Required:** No
- **Default:** `[1, 50]`

### `n_freqs`

- **Type:** `int`
- **Required:** No
- **Default:** `100`

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"morlet"`

### `n_cycles`

- **Type:** `float | list`
- **Required:** No
- **Default:** `None`

### `compute_itc`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

### `baseline`

- **Type:** `float | None`
- **Required:** No
- **Default:** `None`

### `baseline_mode`

- **Type:** `str`
- **Required:** No
- **Default:** `"percent"`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- AverageTFR object containing power and optionally ITC

## Usage Examples

### Basic Usage

```yaml
stages:
  - time_frequency
```

### With Custom Parameters

```yaml
stages:
  - time_frequency:
      freq_range: [1, 2, 3]
      n_freqs: 10
```

### Command Line Help

```bash
eeg-processor help-stage time_frequency
```

## Dependencies

This stage should typically be run after:

- `epoch`

## Related Stages

- [`time_frequency_raw`](../post_epoching/time_frequency_raw.md)
- [`time_frequency_average`](../post_epoching/time_frequency_average.md)

---

[← Back to Post Epoching](../post_epoching.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)