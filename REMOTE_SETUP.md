# Remote PC Setup Protocol

Complete step-by-step guide to clone and set up the eeg-processor project on a new PC.

## Prerequisites Check and Installation

### Step 1: Check and Install Git

```bash
# Check if Git is installed
git --version

# If not installed, install Git:
# On Ubuntu/Debian/WSL2:
sudo apt update
sudo apt install git -y

# On Windows (PowerShell as Administrator):
# Download from https://git-scm.com/download/win
# Or use winget:
winget install --id Git.Git -e --source winget

# Verify installation
git --version
# Should show: git version 2.x.x
```

### Step 2: Check and Install VSCode

```bash
# Check if VSCode is installed
code --version

# If not installed:
# On Ubuntu/Debian/WSL2:
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt update
sudo apt install code -y

# On Windows:
# Download from https://code.visualstudio.com/
# Or use winget:
winget install -e --id Microsoft.VisualStudioCode

# Verify installation
code --version
# Should show version, commit hash, and architecture
```

### Step 3: Check and Install Miniconda

```bash
# Check if conda is installed
conda --version

# If not installed:
# On Linux/WSL2:
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
~/miniconda3/bin/conda init bash
source ~/.bashrc

# On Windows:
# Download from https://docs.conda.io/en/latest/miniconda.html
# Run the installer and follow prompts

# Verify installation
conda --version
# Should show: conda x.x.x
```

## Git Configuration

### Step 4: Configure Git Identity

```bash
# Set your name and email (replace with your actual info)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set line ending behavior
git config --global core.autocrlf false
git config --global core.eol lf

# Verify configuration
git config --list | grep user
# Should show:
# user.name=Your Name
# user.email=your.email@example.com
```

### Step 5: Set Up GitHub Authentication

Choose **ONE** of these options:

#### Option A: Personal Access Token (Easier)

```bash
# Configure credential storage
git config --global credential.helper store

# You'll be prompted for credentials on first push/pull
# When prompted:
#   Username: your-github-username
#   Password: [paste your GitHub Personal Access Token]

# To create a token:
# 1. Go to: https://github.com/settings/tokens
# 2. Click "Generate new token (classic)"
# 3. Name: "Remote PC Access"
# 4. Select scopes: repo (all), workflow
# 5. Click "Generate token"
# 6. COPY THE TOKEN (you won't see it again!)
```

#### Option B: SSH Key (More Permanent)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
# Press Enter to accept default location
# Optionally set a passphrase (or press Enter for none)

# Display public key
cat ~/.ssh/id_ed25519.pub
# Copy the entire output

# Add to GitHub:
# 1. Go to: https://github.com/settings/keys
# 2. Click "New SSH key"
# 3. Title: "Remote PC"
# 4. Paste your public key
# 5. Click "Add SSH key"

# Test connection
ssh -T git@github.com
# Should see: "Hi username! You've successfully authenticated..."
```

## Clone and Setup Project

### Step 6: Create Project Directory

```bash
# Create projects directory
mkdir -p ~/projects
cd ~/projects

# Verify location
pwd
# Should show: /home/username/projects (or similar)
```

### Step 7: Clone Repository

```bash
# For HTTPS (if using Personal Access Token):
git clone https://github.com/SDevrajK/eeg-processor.git

# For SSH (if using SSH key):
git clone git@github.com:SDevrajK/eeg-processor.git

# Navigate into project
cd eeg-processor

# Verify clone success
ls -la
# Should see: src/, tests/, config/, README.md, etc.
```

### Step 8: Create Conda Environment

```bash
# Make sure conda is initialized
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda create -n eeg-processor python=3.12 -y

# Activate environment
conda activate eeg-processor

# Verify environment
which python
conda env list
# Should show eeg-processor with an asterisk (*)
```

### Step 9: Install Dependencies

```bash
# Make sure you're in the project directory
cd ~/projects/eeg-processor

# Make sure environment is activated
conda activate eeg-processor

# Install package in development mode
pip install -e .

# Install any additional dependencies (if requirements.txt exists)
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Verify installation
python -c "from src.eeg_processor.cli import cli; cli(['--help'])"
# Should show the CLI help message
```

### Step 10: Open in VSCode

```bash
# Open project in VSCode
cd ~/projects/eeg-processor
code .

# Or if using WSL2 from Windows:
# Open VSCode on Windows, then:
# Ctrl+Shift+P → "Remote-WSL: Open Folder in WSL"
# Navigate to ~/projects/eeg-processor
```

### Step 11: Configure VSCode Python Environment

In VSCode:
1. Open Command Palette: `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose: `eeg-processor` conda environment
4. Verify in bottom-right corner: Should show `eeg-processor`

## Verification Tests

### Step 12: Run Tests

```bash
# Activate environment
conda activate eeg-processor

# Run basic tests
cd ~/projects/eeg-processor
python -m pytest tests/ -v

# Test CLI commands
python -c "from src.eeg_processor.cli import cli; cli(['list-stages'])"
python -c "from src.eeg_processor.cli import cli; cli(['list-presets'])"
```

### Step 13: Test Git Operations

```bash
# Check git status
git status
# Should show: "On branch main" and "nothing to commit, working tree clean"

# Test pull (get latest changes)
git pull origin main
# Should show: "Already up to date" or download any new changes

# Verify remote connection
git remote -v
# Should show your GitHub repository URL
```

## Quick Reference Commands

### Daily Workflow on Remote PC

```bash
# Start working session
cd ~/projects/eeg-processor
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg-processor

# Get latest changes from GitHub
git pull origin main

# Open in VSCode
code .

# After making changes, commit and push:
git add .
git commit -m "Description of changes"
git push origin main
```

### Environment Activation (Add to ~/.bashrc for convenience)

```bash
# Add this to ~/.bashrc for automatic conda initialization:
echo 'source ~/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc

# Quick activate command (add alias to ~/.bashrc):
echo 'alias eeg="cd ~/projects/eeg-processor && conda activate eeg-processor"' >> ~/.bashrc
source ~/.bashrc

# Now you can just type:
eeg
# This will navigate to project and activate environment!
```

## Troubleshooting

### Git Clone Fails

```bash
# If using HTTPS and prompted for password:
# Make sure you use your Personal Access Token, NOT your GitHub password

# If using SSH and connection fails:
ssh -T git@github.com
# Check the error message

# Permission denied? Add SSH key to ssh-agent:
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Python Import Errors

```bash
# Ensure environment is activated
conda activate eeg-processor

# Reinstall package
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
# Should include: /home/username/projects/eeg-processor/src
```

### VSCode Can't Find Python

```bash
# Install Python extension in VSCode:
# Ctrl+Shift+X, search "Python", install Microsoft's Python extension

# Reload window:
# Ctrl+Shift+P → "Developer: Reload Window"

# Select interpreter again:
# Ctrl+Shift+P → "Python: Select Interpreter"
```

## Success Checklist

After completing all steps, verify:

- [ ] `git --version` works
- [ ] `code --version` works
- [ ] `conda --version` works
- [ ] `git config user.name` shows your name
- [ ] Project cloned to `~/projects/eeg-processor`
- [ ] `conda activate eeg-processor` works
- [ ] `python -c "from src.eeg_processor.cli import cli; cli(['--help'])"` works
- [ ] `git pull origin main` works without errors
- [ ] VSCode opens project successfully
- [ ] VSCode shows correct Python interpreter (eeg-processor)

## Next Steps

Once setup is complete, see:
- `DEVELOPMENT.md` - Development workflow and best practices
- `CLAUDE.md` - Project guidelines and architecture
- `README.md` - Project overview and usage

You're ready to develop!
