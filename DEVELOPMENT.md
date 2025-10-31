# Development Workflow

## Cross-Platform Development Setup

This project is developed across Windows and WSL2 environments. This document outlines the workflow for maintaining code across both platforms.

### Repository Structure

**Primary Development Location**: WSL2 (`/home/sdevrajk/projects/eeg-processor`)
**Windows Mirror**: `C:\Users\sayee\Documents\Research\PythonCode\EEG_Processor`
**Remote Repository**: https://github.com/SDevrajK/eeg-processor.git

### Development Tools

**Primary IDE**: Visual Studio Code with WSL2 integration
**Git Interface**: VSCode's built-in Git integration (Source Control panel)
**Terminal**: WSL2 bash terminal within VSCode

### Recommended Workflow

#### Option 1: WSL as Primary (Recommended)

Use WSL2 as your primary development environment and treat the Windows copy as a mirror for Windows-specific testing.

**Development cycle using VSCode:**
1. Open project in VSCode using WSL: `code /home/sdevrajk/projects/eeg-processor`
2. Make changes and save files
3. Use Source Control panel (Ctrl+Shift+G) to:
   - Review changes
   - Stage files (click + icon)
   - Write commit message
   - Commit (click ✓ icon)
   - Push to GitHub (click "Sync Changes" or ... → Push)

**Alternative: Command line workflow**
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

### Git Authentication Setup

#### Configuring Your Git Identity

First, set your actual Git credentials (not Claude Code's):

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Or set only for this repository
cd /home/sdevrajk/projects/eeg-processor
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

#### GitHub Authentication Options

VSCode's Git integration sometimes fails to push due to authentication issues. Here are solutions:

**Option 1: GitHub Personal Access Token (Recommended for VSCode)**

1. Create a Personal Access Token on GitHub:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a name: "VSCode WSL2"
   - Select scopes: `repo` (all), `workflow`
   - Click "Generate token"
   - **COPY THE TOKEN** - you won't see it again!

2. Configure Git to use the token:
   ```bash
   # Store credentials (saves to ~/.git-credentials)
   git config --global credential.helper store

   # Next push will prompt for credentials
   # Username: your-github-username
   # Password: paste-your-token (not your GitHub password!)
   ```

3. Test authentication:
   ```bash
   git push origin main
   ```

**Option 2: SSH Keys (More Secure, No Expiration)**

1. Generate SSH key in WSL2:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # Press Enter to accept default location
   # Optionally set a passphrase
   ```

2. Add SSH key to GitHub:
   ```bash
   # Copy public key
   cat ~/.ssh/id_ed25519.pub
   # Copy the output
   ```
   - Go to https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your public key
   - Click "Add SSH key"

3. Change remote URL to SSH:
   ```bash
   cd /home/sdevrajk/projects/eeg-processor
   git remote set-url origin git@github.com:SDevrajK/eeg-processor.git
   ```

4. Test connection:
   ```bash
   ssh -T git@github.com
   # Should see: "Hi username! You've successfully authenticated..."
   ```

**Option 3: VSCode GitHub Extension**

VSCode can authenticate directly with GitHub:

1. Install "GitHub Pull Requests and Issues" extension (usually pre-installed)
2. Click "Sign in to GitHub" when prompted in VSCode
3. Follow the browser authentication flow
4. VSCode will handle authentication automatically

#### Troubleshooting Push Failures

**If VSCode push fails:**

1. Try pushing from integrated terminal:
   ```bash
   cd /home/sdevrajk/projects/eeg-processor
   git push origin main
   ```
   - If prompted for credentials, enter your username and token (not password)
   - Credentials will be saved for future pushes

2. Check authentication status:
   ```bash
   git config --list | grep credential
   git config --list | grep user
   git remote -v
   ```

3. Clear cached credentials if needed:
   ```bash
   # Remove stored credentials
   rm ~/.git-credentials

   # Or use credential helper to erase
   git credential reject
   # Then enter:
   # protocol=https
   # host=github.com
   # [press Enter twice]
   ```

4. Verify remote URL format:
   ```bash
   git remote -v
   # Should show either:
   # https://github.com/SDevrajK/eeg-processor.git (for token)
   # git@github.com:SDevrajK/eeg-processor.git (for SSH)
   ```

**If authentication keeps failing:**
- Close and reopen VSCode
- Reload the WSL window (Ctrl+Shift+P → "Remote-WSL: Reopen Folder in WSL")
- Use the terminal `git push` as a fallback
- Check GitHub for rate limiting or account issues

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
