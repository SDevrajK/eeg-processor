# Development Workflow

## Cross-Platform Development Setup

This project is developed across Windows and WSL2 environments. This document outlines the workflow for maintaining code across both platforms.

### Repository Structure

**Primary Development Location**: WSL2 (`/home/sdevrajk/projects/eeg-processor`)
**Windows Mirror**: `C:\Users\sayee\Documents\Research\PythonCode\EEG_Processor`
**Remote Repository**: https://github.com/SDevrajK/eeg-processor.git

### Recommended Workflow

#### Option 1: WSL as Primary (Recommended)

Use WSL2 as your primary development environment and treat the Windows copy as a mirror for Windows-specific testing.

**Development cycle:**
```bash
# In WSL2
cd /home/sdevrajk/projects/eeg-processor

# Make changes, test, commit
git add <files>
git commit -m "Description of changes"
git push origin main

# To sync Windows copy (if needed for testing)
rsync -av --delete /home/sdevrajk/projects/eeg-processor/ \
  /mnt/c/Users/sayee/Documents/Research/PythonCode/EEG_Processor/
```

**On the other PC:**
```bash
# Clone initially
git clone https://github.com/SDevrajK/eeg-processor.git
cd eeg-processor

# Pull updates
git pull origin main
```

#### Option 2: Windows as Primary

If you prefer Windows development, use Git for Windows and WSL2 only for Linux-specific testing.

**Development cycle (Windows Command Prompt or PowerShell):**
```powershell
# In Windows
cd C:\Users\sayee\Documents\Research\PythonCode\EEG_Processor

# Make changes, test, commit
git add <files>
git commit -m "Description of changes"
git push origin main

# To sync WSL copy (if needed)
robocopy C:\Users\sayee\Documents\Research\PythonCode\EEG_Processor ^
  \\wsl$\Ubuntu\home\sdevrajk\projects\eeg-processor /MIR /XD .git
```

### Environment Setup on New PC

#### WSL2 Setup
```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n eeg-processor python=3.12 -y
conda activate eeg-processor

# Clone repository
cd ~/projects
git clone https://github.com/SDevrajK/eeg-processor.git
cd eeg-processor

# Install dependencies
pip install -e .
pip install -r requirements.txt  # If you have one

# Verify installation
python -c "from src.eeg_processor.cli import cli; cli(['--help'])"
```

#### Windows Setup (Optional)
```powershell
# Install Miniconda or Anaconda for Windows
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n eeg-processor python=3.12 -y
conda activate eeg-processor

# Clone repository
cd C:\Users\sayee\Documents\Research\PythonCode
git clone https://github.com/SDevrajK/eeg-processor.git
cd EEG_Processor

# Install dependencies
pip install -e .
```

### File Synchronization Best Practices

**Do NOT sync between Windows and WSL manually** - use Git as the source of truth:
- ✅ Make changes in one location
- ✅ Commit and push to GitHub
- ✅ Pull changes on the other machine
- ❌ Avoid manual file copying between Windows and WSL
- ❌ Don't work on both copies simultaneously

### Git Configuration

Ensure consistent line endings across platforms:

```bash
# Set in repository
git config core.autocrlf false
git config core.eol lf

# Or globally
git config --global core.autocrlf false
git config --global core.eol lf
```

### Files to Exclude from Git

The `.gitignore` file already excludes:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- IDE files (`.vscode/`, `.idea/`)
- Test data and results directories
- Environment-specific configuration

### Common Issues and Solutions

**Issue**: Line ending conflicts between Windows and Linux
**Solution**: Use `core.autocrlf=false` and `core.eol=lf` (already configured)

**Issue**: Path separators in code
**Solution**: Always use `pathlib.Path` or `os.path.join()` instead of hardcoded `/` or `\`

**Issue**: Different Python versions between environments
**Solution**: Pin to Python 3.12 in both environments

**Issue**: File permission differences
**Solution**: Git only tracks execute permission; file ownership differences are normal

### Testing Before Commit

Always test your changes before committing:

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg-processor

# Run tests
python -m pytest tests/ -v

# Check code quality (optional)
python -m black src/ tests/ --check
python -m flake8 src/

# Test CLI
python -c "from src.eeg_processor.cli import cli; cli(['list-stages'])"
```

### Deployment Checklist

Before pushing to GitHub:
- [ ] All tests pass
- [ ] Code follows project guidelines (see CLAUDE.md)
- [ ] No debug print statements or commented code
- [ ] Documentation updated if needed
- [ ] Commit message is descriptive
- [ ] No sensitive data in commit

### Repository Maintenance

**Regular cleanup:**
```bash
# Remove untracked files (be careful!)
git clean -fd

# Remove ignored files from cache
git rm -r --cached __pycache__/
git rm --cached *.pyc
```

**View commit history:**
```bash
git log --oneline --graph --all --decorate
```

**Undo uncommitted changes:**
```bash
# Discard changes to specific file
git checkout -- filename

# Discard all uncommitted changes
git reset --hard HEAD
```

### Getting Help

- Project documentation: See `CLAUDE.md` for development guidelines
- Git issues: Check https://github.com/SDevrajK/eeg-processor/issues
- Git basics: https://git-scm.com/doc
