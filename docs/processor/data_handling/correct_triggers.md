# correct_triggers

Correct incorrectly coded triggers in EEG data.

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `condition`

- **Type:** `dict`
- **Required:** Yes

### `method`

- **Type:** `str`
- **Required:** No
- **Default:** `"alternating"`

### `corrupted_codes`

- **Type:** `list | None`
- **Required:** No
- **Default:** `None`

### `auto_detect_corrupted`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Returns

- Raw object with corrected triggers

## Usage Examples

### Basic Usage

```yaml
stages:
  - correct_triggers
```

### With Custom Parameters

```yaml
stages:
  - correct_triggers:
      method: "example"
```

### Command Line Help

```bash
eeg-processor help-stage correct_triggers
```

---

[← Back to Data Handling](../data_handling.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)