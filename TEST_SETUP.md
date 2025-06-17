# EEG Processor Test Setup Instructions

## Quick Setup for Testing

The tests have been designed to work with minimal setup. Most tests use mocking and temporary files.

### 1. Install Package in Development Mode

If you have pip available:
```bash
pip install -e .
```

Or install individual dependencies as needed:
```bash
pip install pytest mne matplotlib numpy pandas pyyaml loguru click
```

### 2. Run Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test files  
python3 -m pytest tests/test_config_loader.py -v
python3 -m pytest tests/test_file_io.py -v
python3 -m pytest tests/test_pipeline.py -v

# Run tests without pytest (basic Python)
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_config_loader import *
print('Config loader tests imported successfully')
"
```

### 3. Manual Testing

If automated tests don't work, you can test manually:

```python
# Test imports
import sys
sys.path.insert(0, 'src')

from eeg_processor.utils.config_loader import load_config
from eeg_processor.utils.exceptions import ConfigurationError

# Test config loading
config = load_config('tests/test_data/test_config.yml')
print('Config loaded successfully!')
```

### 4. Test Data

Minimal test data has been created in `tests/test_data/`:
- BrainVision files (.vhdr, .vmrk, .eeg)
- Empty placeholder files for other formats
- Test configuration file

### 5. Troubleshooting

If tests fail:
1. Check that you're in the project root directory
2. Verify Python can find the eeg_processor module
3. Install missing dependencies as needed
4. Most file I/O tests use mocking, so they shouldn't need real data files

### 6. Testing Individual Modules

You can test individual modules without pytest:

```python
# Test configuration loading
python3 -c "
import sys; sys.path.insert(0, 'src')
try:
    from eeg_processor.utils.config_loader import load_config
    config = load_config({'paths': {'raw_data_dir': '/tmp'}})
    print('✅ Config loader works')
except Exception as e:
    print(f'❌ Config loader failed: {e}')
"
```
