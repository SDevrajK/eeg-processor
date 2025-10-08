# adjust_events

Adjust event times with optional in-place operation

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `shift_ms`

- **Type:** `float`
- **Required:** Yes

### `target_events`

- **Type:** `list | None`
- **Required:** No
- **Default:** `None`

### `protect_events`

- **Type:** `list | None`
- **Required:** No
- **Default:** `None`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Raw object with adjusted event times

## Usage Examples

### Basic Usage

```yaml
stages:
  - adjust_events
```

### With Custom Parameters

```yaml
stages:
  - adjust_events:
      shift_ms: 1.0
      target_events: [1, 2, 3]
```

### Command Line Help

```bash
eeg-processor help-stage adjust_events
```

---

[← Back to Data Handling](../data_handling.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)