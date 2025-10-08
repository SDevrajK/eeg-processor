# compute_eog

Compute HEOG/VEOG from electrode pairs.

## Parameters

### `raw`

- **Type:** `BaseRaw`
- **Required:** Yes

### `heog_pair`

- **Type:** `tuple | None`
- **Required:** No
- **Default:** `None`

### `veog_pair`

- **Type:** `tuple | None`
- **Required:** No
- **Default:** `None`

### `ch_names`

- **Type:** `dict`
- **Required:** No
- **Default:** `{'heog': 'HEOG', 'veog': 'VEOG'}`

### `overwrite`

- **Type:** `bool`
- **Required:** No
- **Default:** `False`

## Returns

- Raw object with added HEOG/VEOG channels

## Usage Examples

### Basic Usage

```yaml
stages:
  - compute_eog
```

### With Custom Parameters

```yaml
stages:
  - compute_eog:
```

### Command Line Help

```bash
eeg-processor help-stage compute_eog
```

---

[← Back to Preprocessing](../preprocessing.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)