# rereference

Apply rereferencing to raw EEG data with robust exclude handling. Preserves existing projections (e.g., for blink correction) while adding new reference.

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `method`

- **Type:** `str | list`
- **Required:** No
- **Default:** `"average"`

### `exclude`

- **Type:** `list | None`
- **Required:** No
- **Default:** `None`

### `interpolate_bads`

- **Type:** `bool`
- **Required:** No
- **Default:** `True`

### `projection`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

### `verbose`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

## Returns

- Rereferenced Raw object.

## Usage Examples

### Basic Usage

```yaml
stages:
  - rereference
```

### With Custom Parameters

```yaml
stages:
  - rereference:
      method: "average"
```

### Command Line Help

```bash
eeg-processor help-stage rereference
```

## Dependencies

This stage should typically be run after:

- `detect_bad_channels`

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)