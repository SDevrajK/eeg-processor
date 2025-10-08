# PRD: EEG Processor Configuration System Enhancement

## Problem Statement

The current EEG Processor configuration system presents significant usability challenges:

1. **Complex Template**: Users must download a 321-line template file (`template_config.yml`) with extensive comments and documentation
2. **Manual Cleanup**: Users need to manually remove unnecessary sections and comments for their specific use case
3. **High Barrier to Entry**: New users must navigate through detailed documentation to understand configuration options
4. **Time-Consuming Setup**: Experienced users waste time creating configs for different experiments
5. **Error-Prone Process**: Complex template increases risk of configuration errors
6. **Limited Stage Discovery**: No easy way to discover available processing stages and their parameters
7. **Inadequate Help System**: Users cannot easily get help for specific processing stages

This creates friction in the user experience and slows down research workflows, particularly for users who need to quickly set up processing for different experiments.

## Current State Analysis

### Existing Configuration System
- **Template-based**: Single comprehensive template with all possible options
- **YAML Format**: Uses YAML configuration files with extensive commenting
- **Manual Process**: Users download, edit, and clean up template files manually
- **Limited Tooling**: Basic `create-config` command exists but generates complex templates

### Stage System Architecture
- **Well-organized Registry**: 17 processing stages in categorized registry
- **Comprehensive Documentation**: Processing functions have detailed docstrings with Args, Returns, Notes
- **Type Hints**: Full parameter specifications with defaults and types
- **Modular Design**: Stages are implemented as separate, well-documented functions

### Current Config Structure (from `template_config.yml`)
```yaml
# 321 lines with extensive comments covering:
study: {...}           # Study metadata
raw_data_dir: "..."    # Data paths  
participants: [...]    # Participant definitions
stages: [...]          # Processing pipeline (12+ different stage types)
conditions: [...]      # Experimental conditions
output: {...}          # Output settings
```

### Existing CLI Support
- `eeg-processor create-config`: Creates basic template
- `eeg-processor validate`: Validates configuration files  
- Partial interactive wizard in `interactive_config.py` (388 lines)

## Solution Overview

Implement a multi-tiered configuration system that provides different levels of complexity and automation based on user needs and experience levels, with integrated stage discovery and help systems.

### Core Design Principles
1. **Progressive Disclosure**: Start simple, reveal complexity as needed
2. **Preset-Driven**: Common use cases should be one command away
3. **Smart Defaults**: Automatically configure reasonable defaults based on context
4. **Maintain Flexibility**: Advanced users retain full control over all parameters
5. **Backward Compatibility**: Existing configurations continue to work
6. **Discoverable**: Users can easily find and understand available processing stages
7. **Self-Documenting**: Built-in help system reduces reliance on external documentation

## Proposed Solution Architecture

### 1. Configuration Presets Library

Create a library of pre-built configurations for common EEG processing scenarios:

#### **Basic Presets**
- `basic-erp`: Simple ERP processing (Filter → Bad channels → Re-reference → Epoch)
- `minimal`: Absolute minimum processing for testing
- `quality-focused`: Comprehensive quality control with basic processing

#### **Advanced Presets**  
- `artifact-removal`: Comprehensive artifact correction (ASR + EMCP + ICA)
- `time-frequency`: Time-frequency analysis pipeline
- `resting-state`: Continuous data processing for resting state analysis
- `clinical`: Conservative processing for clinical applications

#### **Research Presets**
- `auditory-erp`: Optimized for auditory ERP experiments
- `visual-erp`: Optimized for visual ERP experiments  
- `oddball-paradigm`: Standard oddball experimental setup
- `emcp-comparison`: Compare different blink correction methods

### 2. Enhanced CLI Interface

Expand the CLI with multiple modes and integrated help system:

```bash
# Quick preset generation
eeg-processor create-config --preset basic-erp --output my_config.yml
eeg-processor create-config --preset artifact-removal --format brainvision

# Minimal configuration
eeg-processor create-config --minimal --participants auto --stages filter,epoch

# Template generation with specific focus
eeg-processor create-config --template --focus artifact-removal
eeg-processor create-config --template --focus time-frequency

# Interactive wizard (improved)
eeg-processor create-config --interactive --help-text

# Stage discovery and help system
eeg-processor list-stages                    # List all available stages
eeg-processor list-stages --category preprocessing  # List by category
eeg-processor help-stage filter              # Show detailed help for filter stage
eeg-processor help-stage --all               # Show help for all stages
eeg-processor help-stage filter --examples   # Show usage examples

# List available options
eeg-processor list-presets
eeg-processor list-presets --category research
```

### 3. Smart Configuration Builder

#### **Data Format Detection**
- Automatically detect file format from directory structure
- Suggest appropriate default parameters based on format
- Warn about format-specific considerations

#### **Parameter Validation**  
- Real-time validation with helpful error messages
- Suggest fixes for common configuration errors
- Validate parameter combinations for compatibility

#### **Progressive Configuration**
- Start with minimal required parameters
- Progressively add complexity based on user choices
- Context-sensitive help and suggestions

### 4. Stage Documentation System

#### **Built-in Help System**
- Extract comprehensive documentation from existing docstrings
- Provide parameter specifications with types and defaults
- Show usage examples and best practices
- Categorize stages by processing type

#### **Multi-format Documentation**
- CLI help for immediate assistance
- Markdown generation for web documentation
- JSON schema for programmatic access
- Interactive examples with real usage patterns

### 5. Configuration Composition System

#### **Base + Override Pattern**
```yaml
# Base configuration
base: "presets/basic-erp"

# Experiment-specific overrides
study:
  name: "My Experiment"
  
stages:
  - filter:
      l_freq: 0.5  # Override default 0.1 Hz
```

#### **Template Inheritance**
- Allow configs to inherit from presets
- Override specific parameters while keeping base structure
- Support multiple inheritance layers

## Implementation Plan

### Phase 1: Core Infrastructure & Stage Help (Week 1-2)
1. **Stage Discovery System**
   - Implement `list-stages` CLI command with categorization
   - Add `help-stage` command with comprehensive parameter documentation
   - Extract and format existing docstrings for CLI display
   - Add stage usage examples and best practices

2. **Create Presets Library**
   - Design preset file structure
   - Implement 4-6 core presets (basic-erp, artifact-removal, minimal, etc.)
   - Add preset loading and validation system

3. **Enhanced CLI Commands**
   - Extend `create-config` with `--preset`, `--minimal`, `--template` flags
   - Add `list-presets` command
   - Implement basic preset selection logic

### Phase 2: Smart Features & Documentation (Week 3-4)  
4. **Improved Interactive Wizard**
   - Enhance UX with better prompts and help text
   - Add contextual guidance and explanations
   - Implement progressive disclosure of options
   - Integrate stage help into wizard flow

5. **Validation and Suggestions**
   - Smart parameter validation with fix suggestions
   - Data format detection and appropriate defaults
   - Configuration compatibility checking

6. **Programmatic Documentation Generation**
   - Auto-generate markdown documentation for all stages
   - Create JSON schema for stage parameters
   - Generate HTML documentation for web viewing
   - Add code examples and usage patterns

### Phase 3: Advanced Features & Polish (Week 5-6)
7. **Configuration Composition**
   - Implement base + override system
   - Add template inheritance support
   - Create config merging utilities

8. **Documentation Integration & Examples**
   - Update user documentation with new features
   - Create comprehensive tutorial examples
   - Add preset documentation with use cases
   - Generate API documentation for programmatic access

## Technical Requirements

### New Files/Modules
- `src/eeg_processor/presets/` directory with preset YAML files
- `src/eeg_processor/utils/preset_manager.py` for preset loading/management
- `src/eeg_processor/utils/config_wizard.py` enhanced interactive wizard
- `src/eeg_processor/utils/config_validator.py` smart validation system
- `src/eeg_processor/utils/stage_documentation.py` stage help and documentation system
- `src/eeg_processor/utils/doc_generator.py` programmatic documentation generation

### CLI Integration
- Extend existing `cli.py` with new commands and options:
  - `list-stages` with filtering and categorization
  - `help-stage` with comprehensive parameter help
  - Enhanced `create-config` with multiple modes
  - `list-presets` with category filtering
- Maintain backward compatibility with existing commands
- Add comprehensive help text and examples

### Configuration Schema
- Define preset schema and validation rules
- Implement inheritance and override mechanisms
- Add metadata support for presets (descriptions, use cases, etc.)
- Create JSON schema for stage parameters and validation

### Documentation System
- Stage introspection for automatic documentation extraction
- Multi-format output (CLI, Markdown, HTML, JSON)
- Example generation and validation
- Integration with existing docstring documentation

## Success Metrics

### User Experience Improvements
- **50% reduction** in time to create working configuration
- **75% reduction** in configuration-related support requests
- **90% of users** can create configs without consulting template file
- **80% reduction** in stage parameter lookup time

### Technical Metrics  
- **Zero breaking changes** to existing configurations
- **<2 seconds** to generate any preset configuration
- **100% validation coverage** for all preset combinations
- **100% stage documentation coverage** with examples

### Adoption Metrics
- **>80% of new users** use presets instead of manual template editing
- **>60% of experienced users** adopt new CLI commands
- **>70% of users** use stage help system instead of external docs
- **Positive user feedback** on configuration simplicity and discoverability

## Risk Assessment and Mitigation

### Technical Risks
- **Backward Compatibility**: Mitigate by maintaining existing config loading system
- **Preset Maintenance**: Create automated testing for all presets
- **Complexity Creep**: Maintain clear separation between simple and advanced features
- **Documentation Sync**: Use automated generation to keep docs current with code

### User Experience Risks  
- **Feature Discovery**: Add comprehensive help text and progressive disclosure
- **Migration Confusion**: Provide clear migration guide for existing users
- **Over-Simplification**: Ensure advanced users retain full control
- **Help System Overload**: Organize help by user experience level

## Future Considerations

### Potential Extensions
- **GUI Configuration Builder**: Web-based or desktop configuration interface
- **Preset Sharing**: Community-contributed preset library
- **Configuration Analytics**: Usage tracking to improve defaults
- **Integration with Data Management**: Auto-configure based on dataset metadata
- **Interactive Tutorials**: Guided workflows for common analysis patterns

### Long-term Vision
Transform EEG Processor from a template-driven system to an intelligent configuration platform that:
- Adapts to user needs and provides contextual guidance
- Self-documents all capabilities through introspection
- Learns from usage patterns to improve defaults
- Provides seamless progression from beginner to advanced usage

---

**Document Status**: Draft v1.1  
**Last Updated**: 2025-06-18  
**Next Review**: After Phase 1 completion