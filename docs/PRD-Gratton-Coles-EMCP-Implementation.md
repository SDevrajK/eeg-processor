# Product Requirements Document (PRD)
## Gratton & Coles Eye Movement Correction Procedure (EMCP) Implementation

**Version:** 1.0  
**Date:** January 17, 2025  
**Author:** Development Team  
**Status:** Draft  

---

## Executive Summary

This PRD outlines the implementation of the Gratton & Coles Eye Movement Correction Procedure (EMCP) as a new processing stage in the EEG Processor pipeline. EMCP provides a regression-based alternative to ICA for eye blink and movement artifact removal, enabling researchers to choose between different artifact correction approaches based on their specific needs.

## Background and Context

### Current State
The EEG Processor currently provides artifact removal through:
- **ICA-based removal** (`remove_artifacts` with `method: "ica"`) in `src/eeg_processor/processing/ica.py`
- **ASR cleaning** (`clean_rawdata_asr`) for transient artifacts in `src/eeg_processor/processing/artifact.py`
- **Basic regression** (`remove_blinks_regression`) implemented but not integrated as a pipeline stage, and the current implementation was not working correctly

### Problem Statement
Researchers need access to the classic Gratton & Coles EMCP method for:
1. **Compatibility** with existing research protocols and publications
2. **Comparison studies** between ICA and regression-based approaches
3. **Computational efficiency** for large datasets where ICA may be too slow
4. **Historical reproducibility** for replicating older studies
5. **Methodological alternatives** when ICA fails or is inappropriate

### Scientific Background
The Gratton & Coles EMCP (1983) is a widely-used regression-based method that:
- Calculates propagation factors between EOG and EEG channels
- Removes stimulus-linked variability before factor estimation
- Provides separate correction for blinks vs. eye movements
- Uses data from the experimental session itself (no separate calibration)
- Allows retention of all trials regardless of ocular artifact presence

## Requirements

### Functional Requirements

#### FR1: New Pipeline Stage Implementation
- **Requirement:** Implement `remove_blinks_emcp` as a new processing stage
- **Details:** 
  - Mirror the existing `remove_artifacts` stage structure
  - Support same configuration patterns as other artifact removal stages
  - Integrate seamlessly with existing pipeline architecture

#### FR2: Core EMCP Algorithm
- **Requirement:** Implement the full Gratton & Coles EMCP algorithm
- **Details:**
  - Propagation factor calculation using regression analysis
  - Separate handling of blinks and horizontal eye movements
  - Automatic EOG channel detection and validation
  - Trial-by-trial or continuous data correction modes

#### FR3: Configuration Interface
- **Requirement:** YAML configuration support matching existing patterns
- **Example Configuration:**
```yaml
stages:
  - remove_blinks_emcp:
      eog_channels: ['HEOG', 'VEOG']
      method: 'gratton_coles'  # or 'enhanced'
      correction_mode: 'continuous'  # or 'trial_based'
      propagation_method: 'least_squares'  # or 'robust'
      show_plot: false
      verbose: false
```

#### FR4: Quality Control Integration
- **Requirement:** Full integration with existing quality control system
- **Details:**
  - Store correction metrics in `raw._emcp_metrics`
  - Track propagation factors, correlation preservation, artifact reduction
  - Generate before/after comparison plots
  - Provide correction quality assessment

#### FR5: Visualization and Diagnostics
- **Requirement:** Comprehensive visualization tools
- **Features:**
  - Before/after EEG trace comparisons
  - EOG-EEG correlation plots
  - Propagation factor spatial distribution
  - Artifact reduction effectiveness metrics

### Non-Functional Requirements

#### NFR1: Performance
- **Requirement:** Efficient processing for large datasets
- **Targets:**
  - Process 32-channel, 60-minute recordings in <30 seconds
  - Memory usage comparable to existing stages
  - Support for decimation to speed up factor estimation

#### NFR2: Compatibility
- **Requirement:** Full backward compatibility with existing pipeline
- **Details:**
  - No breaking changes to existing configurations
  - Seamless integration with quality control system
  - Compatible with all supported EEG file formats

#### NFR3: Validation
- **Requirement:** Scientifically validated implementation
- **Details:**
  - Results comparable to published EMCP implementations
  - Validation against reference datasets
  - Comprehensive unit and integration testing

#### NFR4: Documentation
- **Requirement:** Complete documentation and examples
- **Details:**
  - Method description and scientific references
  - Configuration examples and best practices
  - Performance comparisons with ICA approach

## Technical Design

### Implementation Strategy

Based on comprehensive research and analysis of the existing codebase, **we will implement two complementary approaches**:

1. **MNE EOGRegression Method**: Leverages MNE-Python's `EOGRegression` class
   - **Reference Handling**: Works with any reference but follows MNE best practices
   - **Scientific Validation**: Well-tested and widely used implementation
   - **Integration**: Seamless with existing MNE-based pipeline

2. **Original Gratton-Coles Method**: Direct implementation for reference-agnostic processing
   - **Reference Agnostic**: Works without requiring specific reference schemes
   - **Historical Accuracy**: True to the original 1983 methodology
   - **Flexibility**: Alternative when MNE approach is inappropriate

**Key Research Findings:**
- EOGRegression does **NOT strictly require** average reference (contrary to initial assumption)
- MNE documentation recommends setting reference before regression (best practice)
- Existing `remove_blinks_regression` has implementation issues with forced reference changes
- DeepEEG provides reference-agnostic implementation suitable for adaptation

### Architecture Integration

#### Pipeline Integration
```python
# In DataProcessor.stage_registry
"remove_blinks_emcp": self._remove_blinks_emcp

# Stage implementation with method parameter
def _remove_blinks_emcp(self, data: BaseRaw,
                       method: str = "eog_regression", 
                       inplace: bool = False,
                       **kwargs) -> BaseRaw:
    """EMCP blink removal with method selection"""
    if method == "eog_regression":
        from ..processing.emcp import remove_blinks_eog_regression
        return remove_blinks_eog_regression(raw=data, inplace=inplace, **kwargs)
    elif method == "gratton_coles":
        from ..processing.emcp import remove_blinks_gratton_coles  
        return remove_blinks_gratton_coles(raw=data, inplace=inplace, **kwargs)
    else:
        raise ValueError(f"Unknown EMCP method: {method}")
```

#### Module Structure
```
src/eeg_processor/processing/
├── emcp.py                              # New EMCP module
│   ├── remove_blinks_eog_regression()   # MNE EOGRegression wrapper
│   ├── remove_blinks_gratton_coles()    # Reference-agnostic implementation  
│   ├── _validate_eog_channels()         # Channel validation
│   ├── _plot_emcp_comparison()          # Visualization
│   └── _calculate_emcp_metrics()        # Quality metrics
└── artifact.py                         # Replace existing implementation
```

### Implementation Approach

#### Method 1: EOG Regression (`method: "eog_regression"`)
- **Implementation**: MNE-Python's `EOGRegression` class
- **Reference Handling**: Works with current reference (respects user's choice)
- **Preprocessing**: Follows MNE best practices (reference set before regression)
- **Advantages**: 
  - Scientifically validated and widely used
  - Handles both Raw and Epochs data
  - Built-in validation and error handling
  - Consistent with MNE-based architecture
- **Use Case**: Modern implementation following current best practices

#### Method 2: Gratton-Coles Original (`method: "gratton_coles"`)  
- **Implementation**: Direct linear regression (adapted from DeepEEG)
- **Reference Handling**: Reference-agnostic (works with any reference)
- **Preprocessing**: Minimal requirements, true to original 1983 method
- **Advantages**:
  - Historical accuracy to original methodology
  - No reference scheme requirements
  - Simpler preprocessing pipeline
  - Works when MNE approach fails
- **Use Case**: Research requiring original method or problematic reference schemes

#### Configuration Examples
```yaml
# Modern MNE approach (respects current reference)
stages:
  - remove_blinks_emcp:
      method: "eog_regression"
      eog_channels: ['HEOG', 'VEOG']
      proj: true

# Original Gratton-Coles (reference-agnostic)  
stages:
  - remove_blinks_emcp:
      method: "gratton_coles"
      eog_channels: ['HEOG', 'VEOG']
      subtract_evoked: true
```

#### Quality Control Integration
```python
# Metrics storage pattern (method-specific)
raw._emcp_metrics = {
    'method': 'eog_regression',           # or 'gratton_coles'
    'implementation': 'mne_eog_regression', # or 'deepeeg_adapted'
    'eog_channels': ['HEOG', 'VEOG'],
    'reference_scheme': 'average',         # or original reference
    'regression_coefficients': {...},     # Method-specific coefficients
    'correlation_preservation': 0.95,
    'artifact_reduction_db': -12.3,
    'processing_time': 2.1,
    'correction_quality': 'excellent',
    'preprocessing_applied': ['filtering', 'reference_setting'],
    'validation_passed': True
}
```

#### Issues with Current Implementation
Analysis of `remove_blinks_regression` in `artifact.py` revealed several problems:

1. **Forced Reference Changes**: Always applies temporary average reference (line 303)
2. **Complex Reference Restoration**: Fragile logic for restoring original reference
3. **Event Detection Dependency**: May fail if no blink events detected
4. **Limited Error Handling**: Insufficient handling of edge cases
5. **Reference Mismatch**: Conflicts with user's intended reference scheme

The new implementation will address these issues by providing both reference-respecting and reference-agnostic options.

### Core Algorithm Components

#### 1. Propagation Factor Calculation
- **Input:** EEG and EOG time series
- **Process:** 
  - Remove stimulus-locked activity
  - Calculate channel-wise regression coefficients
  - Validate factor stability across segments
- **Output:** Propagation factor matrix

#### 2. Artifact Correction
- **Input:** Raw EEG data + propagation factors
- **Process:**
  - Apply regression-based correction
  - Preserve non-artifact signal components
  - Maintain spatial relationships
- **Output:** Corrected EEG data

#### 3. Quality Assessment
- **Metrics:**
  - Pre/post artifact power ratios
  - Signal-to-noise improvement
  - Spatial correlation preservation
  - Temporal continuity metrics

### Configuration Schema

```yaml
# Complete configuration example
stages:
  - remove_blinks_emcp:
      # EOG Configuration
      eog_channels: ['HEOG', 'VEOG']        # Required EOG channels
      
      # Algorithm Parameters  
      method: 'gratton_coles'               # 'gratton_coles' or 'enhanced'
      correction_mode: 'continuous'         # 'continuous' or 'trial_based'
      propagation_method: 'least_squares'   # 'least_squares' or 'robust'
      
      # Signal Processing
      filter_eog: true                      # Pre-filter EOG channels
      filter_freq: [0.1, 30]              # Filter range if enabled
      
      # Quality Control
      correlation_threshold: 0.3            # Minimum EOG-EEG correlation
      stability_check: true                 # Validate factor stability
      
      # Visualization
      show_plot: false                      # Display correction plots
      plot_duration: 10.0                   # Duration for plots (seconds)
      
      # Processing
      decim: null                          # Decimation for factor calculation
      verbose: false                       # Detailed logging
```

## Implementation Phases

### Phase 1: Core Implementation (1-2 days)
1. **Create `emcp.py` module** wrapping MNE-Python's EOGRegression
2. **Implement wrapper functionality:**
   - Configuration interface adaptation
   - MNE EOGRegression integration
   - EOG channel validation
3. **Add DataProcessor integration**
4. **Basic unit tests**

### Phase 2: Quality Control Integration (1-2 days)
1. **Metrics collection and storage**
2. **Quality assessment algorithms**
3. **Integration with existing QC system**
4. **HTML report generation updates**

### Phase 3: Visualization and Diagnostics (1-2 days)
1. **Before/after comparison plots**
2. **Propagation factor visualization**
3. **Interactive diagnostic tools**
4. **Integration with pipeline plotting system**

### Phase 4: Testing and Validation (1-2 days)
1. **Comprehensive test suite**
2. **Validation against MNE-Python examples**
3. **Performance benchmarking**
4. **Integration testing with full pipeline**

### Phase 5: Documentation and Examples (1 day)
1. **Update CLAUDE.md with EMCP information**
2. **Create configuration examples**
3. **Update CLI help and documentation**
4. **Performance comparison documentation**

## Summary of Implementation Changes

**Total Estimated Time: 5-7 days** (reduced from 7-10 days)

**Key Implementation Benefits:**
1. **Reduced Development Risk**: Using proven MNE-Python implementation
2. **Faster Time-to-Market**: No need to implement complex algorithms from scratch
3. **Better Maintenance**: MNE-Python handles algorithm updates and bug fixes
4. **Scientific Credibility**: Leverages well-established, peer-reviewed implementation
5. **Seamless Integration**: Natural fit with existing MNE-based pipeline architecture

**Risk Mitigation:**
- **No Algorithm Implementation Risk**: MNE-Python handles the complex regression mathematics
- **Validation Simplified**: Can validate against MNE-Python's own test suite and examples
- **Documentation Available**: MNE-Python provides comprehensive method documentation

## Success Criteria

### Technical Success
- [ ] EMCP stage successfully integrates with existing pipeline
- [ ] Processing performance meets NFR1 targets
- [ ] All quality control metrics properly tracked
- [ ] Comprehensive test coverage (>90%)
- [ ] Zero breaking changes to existing functionality

### Scientific Success
- [ ] Results match published EMCP implementations
- [ ] Artifact reduction effectiveness comparable to ICA
- [ ] Proper preservation of neural signals
- [ ] Validation on multiple EEG datasets

### User Experience Success
- [ ] Simple YAML configuration
- [ ] Clear diagnostic output and plots
- [ ] Seamless integration with existing workflows
- [ ] Comprehensive documentation and examples

## Risk Assessment

### Technical Risks
- **Risk:** Complex propagation factor calculation
  - **Mitigation:** Use established algorithms, comprehensive testing
- **Risk:** Performance impact on large datasets
  - **Mitigation:** Optimization, decimation options, profiling

### Scientific Risks
- **Risk:** Suboptimal artifact correction compared to ICA
  - **Mitigation:** Proper algorithm implementation, validation studies
- **Risk:** Signal distortion from over-correction
  - **Mitigation:** Quality metrics, correlation thresholds, user guidance

### Integration Risks
- **Risk:** Breaking existing pipeline functionality
  - **Mitigation:** Thorough testing, backward compatibility focus
- **Risk:** Quality control system incompatibility
  - **Mitigation:** Follow existing patterns, comprehensive QC integration

## Dependencies

### Internal Dependencies
- `src/eeg_processor/processing/artifact.py` - Existing artifact removal infrastructure
- `src/eeg_processor/state_management/data_processor.py` - Stage registry system
- `src/eeg_processor/quality_control/` - Quality tracking and reporting
- `src/eeg_processor/utils/` - Utility functions and validation

### External Dependencies
- **MNE-Python** - Core EEG processing (already available)
- **NumPy/SciPy** - Regression calculations (already available)
- **Matplotlib** - Visualization (already available)
- **Scikit-learn** - Advanced regression methods (optional enhancement)

## Future Enhancements

### Version 1.1 Features
- **Enhanced EMCP variants** (e.g., adaptive correction factors)
- **Automatic EOG channel detection** from electrode montages
- **Batch processing optimization** for large-scale studies
- **Real-time correction capabilities** for online processing

### Version 1.2 Features
- **Hybrid ICA-EMCP approach** for optimal artifact removal
- **Machine learning enhanced** propagation factor estimation
- **Automated parameter optimization** based on data characteristics
- **Advanced validation metrics** and correction quality assessment

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Technical Lead | | | |
| Product Owner | | | |
| QA Lead | | | |

---

**Document Control:**
- **File Location:** `docs/PRD-Gratton-Coles-EMCP-Implementation.md`
- **Last Modified:** January 17, 2025
- **Review Cycle:** Quarterly
- **Next Review:** April 17, 2025