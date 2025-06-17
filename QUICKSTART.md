# EEG Processor Quick Start Guide

Get up and running with EEG Processor in 5 minutes! 🚀

## Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install eeg-processor
```

### Option 2: Development Installation
```bash
git clone https://github.com/yourusername/eeg-processor.git
cd eeg-processor
pip install -e .
```

### Verify Installation
```bash
eeg-processor --version
eeg-processor --help
```

## Quick Setup

### 1. Prepare Your Data Structure
```
my_experiment/
├── data/
│   ├── raw/                 # Your EEG files here
│   └── processed/           # Results will go here (auto-created)
└── config.yml              # Configuration file
```

### 2. Create Configuration
**Option A: Interactive Wizard (Recommended for beginners)**
```bash
cd my_experiment
eeg-processor create-config --interactive
```

**Option B: Generate Template**
```bash
# For BrainVision data
eeg-processor create-config --format brainvision --output config.yml

# For EDF data  
eeg-processor create-config --format edf --output config.yml
```

**Option C: Minimal Manual Configuration**
Create `config.yml`:
```yaml
paths:
  raw_data_dir: "data/raw"
  results_dir: "data/processed"
  file_extension: ".vhdr"  # or ".edf"

participants: "auto"  # Auto-discover files

stages:
  - load_data
  - filter
  - epoching
  - artifact_rejection
  - save_results

filtering:
  lowpass: 40
  highpass: 0.1
  notch: 50

epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

conditions:
  - name: "condition1"
    condition_markers: ["S1", "S2"]
```

### 3. Validate Configuration
```bash
eeg-processor validate config.yml
```

### 4. Run Processing
```bash
# Preview what will be processed
eeg-processor process config.yml --dry-run

# Process all participants
eeg-processor process config.yml

# Process single participant
eeg-processor process config.yml --participant sub-01
```

### 5. Generate Quality Reports
```bash
eeg-processor quality-report data/processed/
```

## Common Data Formats

### BrainVision (.vhdr/.vmrk/.eeg)
```yaml
data_format: "brainvision"
paths:
  file_extension: ".vhdr"
```

### EDF (.edf)
```yaml
data_format: "edf"  
paths:
  file_extension: ".edf"
```

### FIF (.fif)
```yaml
data_format: "fif"
paths:
  file_extension: ".fif"
```

## Quick Examples

### Example 1: Basic ERP Analysis
```yaml
# P300 oddball paradigm
epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

conditions:
  - name: "target"
    condition_markers: ["S1"]
  - name: "standard"  
    condition_markers: ["S2"]

artifact_rejection:
  peak_to_peak: 100e-6
```

### Example 2: Resting State
```yaml
epoching:
  tmin: 0
  tmax: 2.0
  baseline: null

filtering:
  lowpass: 40
  highpass: 1.0

stages:
  - load_data
  - filter
  - bad_channels
  - epoching
  - ica
  - save_results
```

### Example 3: Sleep EEG
```yaml
data_format: "edf"

epoching:
  tmin: 0
  tmax: 30  # 30-second epochs
  baseline: null

filtering:
  lowpass: 30
  highpass: 0.5
  notch: null

conditions:
  - name: "wake"
    condition_markers: ["Wake"]
  - name: "sleep"
    condition_markers: ["N1", "N2", "N3", "REM"]
```

## Quick Commands Reference

```bash
# Configuration
eeg-processor create-config --interactive
eeg-processor validate config.yml

# Processing  
eeg-processor process config.yml
eeg-processor process config.yml --participant sub-01
eeg-processor process config.yml --dry-run

# Quality Control
eeg-processor quality-report results/
eeg-processor quality-report results/ --format pdf

# Interactive Exploration
eeg-processor explore config.yml sub-01

# Information
eeg-processor info results/
eeg-processor --help
```

## Troubleshooting

### Problem: Configuration errors
**Solution:**
```bash
eeg-processor validate config.yml
```

### Problem: No participants found  
**Solutions:**
- Check file paths in config
- Verify file extension matches your data
- Use `participants: "auto"` for auto-discovery

### Problem: Processing fails
**Solutions:**
```bash
# Use verbose mode for details
eeg-processor process config.yml --verbose

# Try single participant first
eeg-processor process config.yml --participant sub-01

# Check data format
eeg-processor validate config.yml
```

### Problem: Memory issues
**Solutions:**
```bash
# Process participants individually
eeg-processor process config.yml --participant sub-01

# Reduce epoch length in config:
epoching:
  tmin: -0.1  # Shorter epochs
  tmax: 0.5
```

## Python API Quick Start

```python
from eeg_processor import EEGPipeline

# Basic usage
pipeline = EEGPipeline("config.yml")
results = pipeline.run_all()

# Single participant
result = pipeline.run_participant("sub-01")

# Interactive processing
raw = pipeline.load_participant_data("sub-01")
filtered = pipeline.apply_stage(raw, "filter", lowpass=40)
epochs = pipeline.apply_stage(filtered, "epoching", tmin=-0.2, tmax=0.8)

# Analysis interface
analysis = pipeline.get_analysis_interface()
epochs = analysis.load_epochs("sub-01", "target")
evoked = analysis.load_evoked("sub-01", "target")
```

## Next Steps

1. **🎓 Learn More**: Read the [Getting Started Tutorial](docs/tutorials/getting_started.md)
2. **🔧 Customize**: Explore [Advanced Tutorial](docs/tutorials/advanced_tutorial.md)  
3. **📚 Reference**: Check the [API Documentation](docs/API.md)
4. **💡 Examples**: Browse the [examples/](examples/) directory
5. **🤝 Community**: Join discussions and report issues on GitHub

## File Organization Best Practices

```
project/
├── configs/
│   ├── base_config.yml
│   ├── experiment1_config.yml
│   └── experiment2_config.yml
├── data/
│   ├── raw/
│   │   ├── experiment1/
│   │   └── experiment2/
│   └── processed/
│       ├── experiment1/
│       └── experiment2/
├── scripts/
│   ├── process_all.py
│   └── generate_reports.py
└── results/
    ├── figures/
    └── reports/
```

## Pro Tips

1. **Always validate** your config before processing
2. **Start small** - test with one participant first
3. **Use dry-run** to preview processing
4. **Check quality reports** after processing
5. **Keep configs in version control**
6. **Document your processing steps**
7. **Backup original data** before processing

---

**Need help?** 
- 📖 Read the [full documentation](docs/)
- 💬 Check [GitHub Issues](https://github.com/yourusername/eeg-processor/issues)
- 🔍 Use `eeg-processor command --help` for command-specific help

Happy processing! 🧠✨