# segment_condition

Segment Raw data based on condition markers with optional in-place operation

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `condition`

- **Type:** `dict`
- **Required:** Yes

### `padding`

- **Type:** `float`
- **Required:** No
- **Default:** `5.0`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Raw object containing only the segmented data

## Usage Examples

### Basic Usage

```yaml
stages:
  - segment_condition
```

### With Custom Parameters

```yaml
stages:
  - segment_condition:
      padding: 1.0
```

### Command Line Help

```bash
eeg-processor help-stage segment_condition
```

## Notes

- in-place operation replaces the original object's content entirely.

---

[← Back to Condition Handling](../condition_handling.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)