# Fix: adjust_events Stage - Event Code Normalization

## Problem Identified

The `adjust_events` stage was causing **inconsistent trigger shifting between subjects** due to a critical bug in event matching.

### Root Cause

The function used **exact string matching** (`desc in target_events`) without normalizing event codes to match the file format. This caused problems with BrainVision files, which use different spacing patterns:

- Events 1-9: `"Stimulus/S  1"` (two spaces)
- Events 10-99: `"Stimulus/S 10"` (one space)
- Events 100+: `"Stimulus/S100"` (no space)

### Impact

If a user specified `target_events: ["Stimulus/S 1"]` (one space):
- ❌ **Did NOT match** events 1-9 in actual data (which have two spaces)
- ✅ **Would match** events 10-99 (which have one space)

This meant:
- **Subject A** with events 1-9 → events didn't shift (no match)
- **Subject B** with events 10-99 → events shifted correctly
- **Subject C** with mixed events → only some events shifted

## Solution Implemented

### 1. Added Format-Aware Normalization

The function now:
1. Inspects the actual annotation format in the data
2. Normalizes target/protect event codes to match that format
3. Handles both integer inputs (`[1, 2, 10]`) and string inputs (`["S1", "S10"]`)

### 2. Key Changes

**File**: `src/eeg_processor/utils/raw_data_tools.py`

```python
# Added internal normalization function
def _normalize_event_for_matching(event_input: Union[str, int]) -> str:
    """
    Normalize event code to match the format used in annotations.
    Inspects actual annotation format rather than relying on Raw object type.
    """
    # Detects BrainVision, Response, or plain numeric formats
    # Returns properly formatted string with correct spacing
```

**Key improvements**:
- Uses actual annotation format inspection instead of Raw object type
- Handles BrainVision spacing rules correctly
- Supports Response codes and plain numeric formats (Curry, Neuroscan)
- Works with both `RawArray` test objects and real EEG files

### 3. Enhanced Logging

Added debug and info logging:
- Shows normalized event codes
- Reports how many events will be shifted
- Helps diagnose issues in user projects

## Usage Examples

### Before Fix (Problematic)

```yaml
processing:
  - adjust_events:
      shift_ms: 50
      target_events: ["Stimulus/S 1"]  # Would fail to match single-digit events!
```

### After Fix (Works Correctly)

```yaml
processing:
  - adjust_events:
      shift_ms: 50
      target_events: [1, 2, 10]  # Automatically normalized to correct format
      # Works consistently across all subjects!
```

Both integer and string inputs now work:
```yaml
# All equivalent and work correctly:
target_events: [1, 10]
target_events: ["S1", "S10"]
target_events: ["1", "10"]
```

## Testing

Created comprehensive test suite (`tests/test_adjust_events.py`):
- ✅ 13 tests covering all scenarios
- ✅ Tests BrainVision and Curry formats
- ✅ Tests integer and string inputs
- ✅ Tests target/protect event interactions
- ✅ Tests edge cases (empty lists, negative shifts, etc.)
- ✅ **Critical test**: Verifies consistent behavior across single-digit vs double-digit events

## Migration Guide

### For Analysis Projects

**No changes required!** The fix is backward compatible and improves existing configs.

**Recommended update** for clarity:
```yaml
# Old (still works but verbose)
processing:
  - adjust_events:
      shift_ms: 50
      target_events: ["Stimulus/S  1", "Stimulus/S  2", "Stimulus/S 10"]

# New (cleaner and more reliable)
processing:
  - adjust_events:
      shift_ms: 50
      target_events: [1, 2, 10]
```

### Updated Schema

The schema now documents:
- Accepts both `number` and `string` types
- Describes automatic normalization
- Clarifies precedence (protect_events > target_events)

## Verification

To verify the fix in your analysis project:

1. Check the logs for normalization messages:
   ```
   DEBUG | Target events normalized: [1, 10] -> ['Stimulus/S  1', 'Stimulus/S 10']
   INFO  | Shifting 2 of 5 events by 50 ms
   ```

2. Verify consistent shifting across all subjects regardless of event codes

## Files Changed

- `src/eeg_processor/utils/raw_data_tools.py` - Core implementation fix
- `schemas/stage_adjust_events.json` - Updated schema with better descriptions
- `config/template_config.yml` - Updated examples with integers
- `tests/test_adjust_events.py` - New comprehensive test suite (13 tests)
- `docs/fix_adjust_events_normalization.md` - This documentation

## Technical Details

### Normalization Logic

```python
# BrainVision: Spacing depends on number of digits
if num < 10:
    return f"Stimulus/S  {num}"  # Two spaces
elif num <= 99:
    return f"Stimulus/S {num}"   # One space
else:
    return f"Stimulus/S{num}"    # No space

# Curry/Neuroscan: Plain numbers
return str(num)
```

### Format Detection

Instead of checking Raw object type (which fails for `RawArray`), we inspect the first annotation:
```python
sample = annotations.description[0]
if sample.startswith("Stimulus/S"):
    # Apply BrainVision formatting
elif sample.startswith("Response/R"):
    # Apply Response formatting
else:
    # Plain numeric format
```

This makes the function robust across all file types and test scenarios.
