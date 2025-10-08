# epoch

Create epochs from Raw data - always returns new Epochs object

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `condition`

- **Type:** `dict`
- **Required:** Yes

### `tmin`

- **Type:** `float`
- **Required:** Yes

### `tmax`

- **Type:** `float`
- **Required:** Yes

### `baseline`

- **Type:** `tuple`
- **Required:** Yes

### `reject_bad`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

### `reject`

- **Type:** `dict | None`
- **Required:** No
- **Default:** `None`

### `flat`

- **Type:** `dict | None`
- **Required:** No
- **Default:** `None`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Epochs object with artifact rejection applied

## Usage Examples

### Basic Usage

```yaml
stages:
  - epoch
```

### With Custom Parameters

```yaml
stages:
  - epoch:
      tmin: -0.2
      tmax: 0.8
      baseline: [-0.2, 0]
```

### Command Line Help

```bash
eeg-processor help-stage epoch
```

## Notes

- since epoching necessarily creates a new Epochs object from Raw data.

## Dependencies

This stage should typically be run after:

- `filter`
- `detect_bad_channels`

## Related Stages

- [`segment_condition`](../condition_handling/segment_condition.md)

## Common Issues

**Not enough epochs after rejection**

Check your event markers and epoch timing. Consider relaxing rejection thresholds or improving preprocessing steps.

**Baseline period issues**

Ensure baseline period is within the epoch window and doesn't overlap with your events of interest.

---

[← Back to Condition Handling](../condition_handling.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)