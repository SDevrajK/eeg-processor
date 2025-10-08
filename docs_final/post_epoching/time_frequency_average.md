# time_frequency_average

Convert RawTFR to AverageTFR by averaging across time dimension

## Parameters

### `raw_tfr`

- **Type:** `Any`
- **Required:** Yes

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"mean"`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- AverageTFR object with time dimension collapsed

## Usage Examples

### Basic Usage

```yaml
stages:
  - time_frequency_average
```

### With Custom Parameters

```yaml
stages:
  - time_frequency_average:
      method: "example"
```

### Command Line Help

```bash
eeg-processor help-stage time_frequency_average
```

## Dependencies

This stage should typically be run after:

- `time_frequency`

---

[← Back to Post Epoching](../post_epoching.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)