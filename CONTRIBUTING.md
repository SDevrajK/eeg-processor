# Contributing to EEG Processor

Thank you for your interest in contributing to EEG Processor! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/eeg-processor.git
   cd eeg-processor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

### Running Code Quality Checks

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/

# Run all checks
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=eeg_processor

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

Example:
```python
def test_pipeline_loads_config_successfully():
    # Arrange
    config_path = "tests/fixtures/valid_config.yml"
    
    # Act
    pipeline = EEGPipeline(config_path)
    
    # Assert
    assert pipeline.config is not None
    assert len(pipeline.config.participants) > 0
```

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def process_eeg_data(raw_data: BaseRaw, filter_params: Dict) -> BaseRaw:
    """Process EEG data with specified parameters.
    
    Args:
        raw_data: The raw EEG data to process.
        filter_params: Dictionary containing filter parameters.
        
    Returns:
        Processed EEG data.
        
    Raises:
        ValueError: If filter parameters are invalid.
        
    Example:
        >>> raw = load_raw("data.fif")
        >>> processed = process_eeg_data(raw, {"l_freq": 1, "h_freq": 40})
    """
```

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import Dict, List, Optional, Union
from pathlib import Path

def load_participants(config_path: Path) -> List[Dict[str, str]]:
    """Load participant information from config."""
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a pull request**
   - Use a descriptive title
   - Explain what your changes do
   - Reference any related issues
   - Include screenshots if applicable

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Commit messages are descriptive

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Operating system and Python version
- EEG Processor version
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error messages or stack traces
- Sample data or config files (if applicable)

### Feature Requests

When requesting features, please include:

- Clear description of the feature
- Use case or motivation
- Proposed implementation (if you have ideas)
- Examples of similar features in other tools

## Code Review Process

1. All submissions require review before merging
2. Reviewers will check for:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Performance implications
3. Address reviewer feedback promptly
4. Once approved, maintainers will merge the PR

## Release Process

1. Version bumping follows semantic versioning
2. Releases are tagged and published to PyPI
3. Changelog is updated for each release

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Assume positive intent

## Getting Help

- Create an issue for bug reports or feature requests
- Start a discussion for questions or ideas
- Check existing issues and discussions first

Thank you for contributing to EEG Processor!