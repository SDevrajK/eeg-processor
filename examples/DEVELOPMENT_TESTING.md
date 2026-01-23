# Development Testing Guide

Quick testing workflows for rapid development iteration without needing to reinstall the package.

## Quick Start

### Fastest: Minimal Function Test (No File I/O)

Test changes to `custom_wavelet.py` in **~2 seconds**:

```bash
# From project root
python examples/test_custom_cwt_minimal.py
```

**What it does**:
- Generates synthetic epochs in memory
- Tests `compute_custom_cwt_tfr()` directly
- No file I/O or pipeline overhead
- **Perfect for development iteration**

### Fast: Full Pipeline Test (With Synthetic Data)

Test complete pipeline with synthetic data in **~10 seconds**:

```bash
# From project root
python examples/test_dev.py
```

**What it does**:
- Generates synthetic BrainVision data
- Runs full pipeline with `custom_cwt` stage
- Saves results to temporary directory
- Auto-cleanup (or `--no-cleanup` to inspect results)

**With custom config**:
```bash
python examples/test_dev.py --config examples/configs/test_custom_cwt.yml
```

### Slow: Test With Your Real Data

Use example config with your data:

```bash
# Edit the config first
vim examples/configs/test_custom_cwt.yml

# Update paths.raw_data and participants
# Then run your normal workflow
python -m eeg_processor.cli process examples/configs/test_custom_cwt.yml
```

## Development Workflow

### Typical Iteration Cycle

**Without these tools** (slow):
1. Make code changes
2. `pip uninstall eeg-processor`
3. `pip install -e .`
4. Run full analysis on real data
5. Wait 5+ minutes
6. Find bug
7. Repeat

**With these tools** (fast):
```bash
# 1. Make code changes to src/eeg_processor/processing/custom_wavelet.py

# 2. Test immediately (2 seconds)
python examples/test_custom_cwt_minimal.py

# 3. If basic tests pass, test pipeline (10 seconds)
python examples/test_dev.py

# 4. Only when everything works, test with real data
```

## Test Files

### `test_custom_cwt_minimal.py`
**Purpose**: Fastest possible test of core functionality
- **Runtime**: ~2 seconds
- **Tests**: Direct function calls with synthetic data
- **Use when**: Testing changes to wavelet computation, parameter handling
- **No installation required**: Modifies `sys.path` to use local `src/`

### `test_dev.py`
**Purpose**: Full pipeline test with synthetic data
- **Runtime**: ~10 seconds
- **Tests**: Complete pipeline including config parsing, stage execution, result saving
- **Use when**: Testing stage integration, config handling, result saving
- **Options**:
  - `--config PATH`: Use custom config file
  - `--no-cleanup`: Keep temporary files for inspection

### `configs/test_custom_cwt.yml`
**Purpose**: Example configuration for custom_cwt stage
- **Use as**: Template for your own configs
- **Demonstrates**: All `custom_cwt` parameters

## Example Config Structure

```yaml
processing:
  # ... standard preprocessing ...

  - custom_cwt:
      wavelet_type: "morse"
      freq_range: [4, 40]
      n_freqs: 50

      # Morse parameters
      morse_gamma: 3.0  # Time resolution
      morse_beta: 3.0   # Frequency resolution

      # Options
      compute_itc: true
      baseline: [-0.2, 0]
      baseline_mode: "mean"
```

## Common Parameter Values

### Morse Wavelet Parameters

**Balanced** (default):
```yaml
morse_gamma: 3.0
morse_beta: 3.0
```

**High temporal resolution** (good for ERPs):
```yaml
morse_gamma: 6.0  # Higher gamma
morse_beta: 3.0
```

**High frequency resolution** (good for oscillations):
```yaml
morse_gamma: 3.0
morse_beta: 10.0  # Higher beta
```

## Tips

### Speed Up Testing

1. **Use fewer frequencies**: `n_freqs: 20` instead of `100`
2. **Shorter padding**: `pad_len: 5` instead of `20`
3. **Smaller frequency range**: `[8, 12]` instead of `[1, 50]`
4. **Skip ITC**: `compute_itc: false`

### Debugging

**Keep test results**:
```bash
python examples/test_dev.py --no-cleanup
# Check output in /tmp/eeg_processor_test_XXXXX/results/
```

**Add more logging**:
```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

**Test specific functionality**:
```python
# Edit test_custom_cwt_minimal.py to focus on one test
# Comment out other tests for faster iteration
```

## Troubleshooting

### Import Errors

```python
# ModuleNotFoundError: No module named 'clouddrift'
pip install clouddrift pywt
```

### Sys.path Issues

The test scripts add `src/` to path automatically:
```python
sys.path.insert(0, str(project_root / "src"))
```

If imports still fail, check your project structure.

### Can't Find Example Data

For `test_dev.py`, data is generated automatically. No real data needed!

## When to Reinstall Package

You only need to reinstall when testing:
- CLI commands (`eeg-processor process ...`)
- Package entry points
- Installation/deployment

For development of processing modules, **no reinstall needed** - just use these test scripts!
