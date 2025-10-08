# view

Unified plotting interface

## Parameters

### `stage_name`

- **Type:** `str`
- **Required:** Yes

### `kwargs`

- **Type:** `Any`
- **Required:** Yes

## Usage Examples

### Basic Usage

```yaml
stages:
  - view
```

### With Custom Parameters

```yaml
stages:
  - view:
      stage_name: "example"
```

### Command Line Help

```bash
eeg-processor help-stage view
```

---

[← Back to Visualization](../visualization.md) | 
[📋 Quick Reference](../quick-reference.md) | 
[🏠 Main Index](../README.md)