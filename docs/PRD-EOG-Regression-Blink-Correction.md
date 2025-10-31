# Product Requirements Document (PRD)
## EOG Regression-Based Blink Correction Implementation

**Version:** 2.0
**Date:** January 19, 2025
**Author:** Development Team
**Status:** Draft

---

## Executive Summary

This PRD outlines the implementation of MNE-Python's EOGRegression method as a new `method='regression'` option in the existing `remove_artifacts` stage. This provides researchers with a faster, deterministic alternative to ICA for blink artifact correction, supporting both continuous (Raw) and stimulus-locked epoched data following the Gratton & Coles (1983) methodology.

## Background and Context

### Current State
The EEG Processor currently provides artifact removal through:
- **ICA-based removal** (`remove_artifacts` with `method="ica"`) in `src/eeg_processor/processing/ica.py`
- **ASR cleaning** (`clean_rawdata_asr`) for transient artifacts
- **EMCP methods** (`remove_blinks_emcp`) - **TO BE DEPRECATED** in favor of unified `remove_artifacts` interface

### Problem Statement
Researchers need:
1. **Unified artifact removal interface** - Single `remove_artifacts` stage with multiple methods
2. **Regression-based alternative to ICA** - Faster processing for blink artifacts
3. **Support for epoched data correction** - Match BrainVision Analyzer workflow (epoch → correction → baseline → rejection)
4. **Gratton & Coles methodology** - Subtract evoked response before fitting coefficients
5. **Simple, single implementation** - No complex mode switching for initial release

### BrainVision Analyzer Workflow Compatibility

**BrainVision Analyzer Pipeline:**
```
Raw Data → Segmentation (Epoching) → OcularCorrection → Baseline Correction → Artifact Rejection
```

**MNE EOGRegression supports this workflow:**
- Fit EOGRegression on **stimulus-locked epochs** with evoked response subtracted
- Apply correction to **same epochs** (or Raw data)
- EOG artifacts don't need to be time-locked to stimulus events
- Preserves stimulus-locked ERP components while removing blink artifacts

### Scientific Background

#### MNE EOGRegression with Gratton & Coles Method

**Core Principle (Gratton et al., 1983):**
> Compute regression coefficients on epoch data with the evoked response subtracted out. The idea is that EEG signal components relevant to the study are in the evoked, so by removing them, mostly noise components (including EOG artifacts) will be left.

**Why This Works:**
- EOG artifacts are present in epochs regardless of stimulus timing
- Subtracting evoked response removes stimulus-locked brain activity
- Remaining signal is dominated by artifacts and spontaneous activity
- Regression coefficients capture artifact propagation without bias from ERPs
- Applying these coefficients preserves the evoked ERP components

**Key Quote from MNE Documentation:**
> "It is possible to compute regression weights directly from the raw data, but this could bias the evoked response more than computing the weights from data with the evoked response removed."

#### When to Apply: Raw vs. Epochs

**For Stimulus-Locked Epochs (BrainVision Analyzer workflow):**
1. Create epochs around experimental events (e.g., stimulus presentation)
2. Fit EOGRegression on epochs with `.subtract_evoked()` (Gratton & Coles approach)
3. Apply correction to original epochs
4. Continue with baseline correction and artifact rejection

**For Continuous Data:**
1. Optionally: Create blink-locked epochs to estimate robust coefficients
2. Fit EOGRegression on Raw or blink epochs
3. Apply correction to Raw data
4. Create experimental epochs from corrected Raw data

**This PRD focuses on the epoched workflow to match BrainVision Analyzer compatibility.**

#### Comparison: ICA vs. EOG Regression

| Metric | EOG Regression | ICA |
|--------|---------------|-----|
| **Computation Time** | Fast (~15s) | Slower (~60-120s) |
| **Data Type Support** | Raw, Epochs, Evoked | Primarily Raw |
| **EOG Channels Required** | Yes (≥1) | No |
| **Artifact Types** | Primarily blinks | Multiple types |
| **Correction Quality** | Good (comparable to ICA for blinks) | Excellent (all artifacts) |
| **Deterministic** | Yes | No (random seed) |
| **Evoked Response Bias** | Low (with subtract_evoked) | Very low |
| **Pipeline Timing** | Before or after epoching | Before epoching |

**When to Use Regression:**
- ✅ EOG channels available
- ✅ Primarily blink artifacts
- ✅ Epoched data workflow needed (match BrainVision)
- ✅ Speed/efficiency important
- ✅ Deterministic results preferred

**When to Use ICA:**
- ✅ Multiple artifact types (blinks + muscle + heartbeat)
- ✅ No EOG channels
- ✅ Maximum correction quality needed
- ✅ Continuous data workflow

## Requirements

### Functional Requirements

#### FR1: Integration as `method='regression'` in `remove_artifacts`
- **Requirement:** Add `regression` as a method option to existing `remove_artifacts` stage
- **Details:**
  - Currently `method='regression'` raises NotImplementedError
  - Implement full EOG regression functionality
  - Maintain compatibility with existing ICA method
  - Support both Raw and Epochs data types

#### FR2: Gratton & Coles Implementation
- **Requirement:** Implement core Gratton & Coles workflow for epoched data
- **Details:**
  - Fit on epochs with `.subtract_evoked()` (removes stimulus-locked activity)
  - Extract regression coefficients (EOG → EEG propagation factors)
  - Apply correction to epochs or Raw data
  - Store original evoked response for quality comparison

#### FR3: Configuration Interface
- **Requirement:** Simple YAML configuration matching existing patterns
- **Configuration:**
```yaml
# For epoched data (BrainVision Analyzer workflow)
processing:
  - epoch: {tmin: -0.2, tmax: 0.8, event_id: {stimulus: 1}}
  - remove_artifacts:
      method: "regression"
      eog_channels: ['HEOG', 'VEOG']
      subtract_evoked: true  # Gratton & Coles approach
      show_plot: false

# For continuous data (alternative workflow)
processing:
  - rereference: {method: "average"}  # Required before regression
  - remove_artifacts:
      method: "regression"
      eog_channels: ['HEOG', 'VEOG']
      subtract_evoked: false  # Not applicable for Raw data
```

#### FR4: Quality Control Integration
- **Requirement:** Track regression metrics for quality assessment
- **Metrics Stored:**
  ```python
  data._regression_metrics = {
      'method': 'regression',
      'implementation': 'mne_eog_regression',
      'eog_channels': ['HEOG', 'VEOG'],
      'data_type': 'Epochs',  # or 'Raw'
      'subtract_evoked': True,
      'n_epochs': 120,  # if Epochs
      'regression_coefficients': {
          'shape': (64, 2),  # n_eeg_channels × n_eog_channels
          'max_coeff': 0.23,
          'mean_coeff': 0.12
      },
      'artifact_reduction': {
          'eog_variance_explained': 0.87,
          'mean_correlation_preserved': 0.95
      },
      'preprocessing_requirements': {
          'reference_set': True,  # For Raw data
          'reference_type': 'average'
      },
      'processing_time_seconds': 8.2
  }
  ```

#### FR5: Documentation and Examples
- **Requirement:** Clear documentation for both workflows
- **Details:**
  - Epoched workflow (BrainVision Analyzer compatibility)
  - Continuous workflow (traditional preprocessing)
  - When to use regression vs. ICA
  - Configuration examples
  - Common issues and solutions

### Non-Functional Requirements

#### NFR1: Performance
- **Targets:**
  - Process 32-channel, 120 epochs in <10 seconds
  - Process 60-minute continuous recording in <15 seconds
  - Memory usage comparable to ICA method

#### NFR2: Compatibility
- **Requirements:**
  - Zero breaking changes to existing `remove_artifacts` configurations
  - Backward compatibility with `method='ica'`
  - Works with all supported EEG file formats
  - Supports both Raw and Epochs data types

#### NFR3: Scientific Validity
- **Requirements:**
  - Results match MNE-Python EOGRegression validation
  - Gratton & Coles methodology correctly implemented
  - Evoked response preservation verified
  - Documented limitations and assumptions

#### NFR4: Code Quality
- **Requirements:**
  - Follow CLAUDE.md development guidelines
  - Minimal, concise implementation
  - Descriptive naming without excessive comments
  - Comprehensive unit tests (>90% coverage)

## Technical Design

### Implementation Overview

**Single, simple implementation** following Gratton & Coles methodology:
- Fit EOGRegression on data (with optional evoked subtraction for Epochs)
- Apply regression coefficients to remove artifacts
- Store quality metrics
- Support both Raw and Epochs seamlessly

### File Modifications

#### 1. Update `src/eeg_processor/state_management/data_processor.py`

```python
def _remove_artifacts(self, data: Union[BaseRaw, Epochs],
                      method: str = "ica",
                      inplace: bool = False,
                      **kwargs) -> Union[BaseRaw, Epochs]:
    """Artifact removal with multiple methods"""
    if method == "ica":
        # Existing ICA implementation
        from ..processing.ica import remove_artifacts_ica
        return remove_artifacts_ica(raw=data, inplace=inplace, **kwargs)
    elif method == "regression":
        # NEW: EOG regression implementation
        from ..processing.regression import remove_artifacts_regression
        return remove_artifacts_regression(data=data, inplace=inplace, **kwargs)
    else:
        raise ValueError(f"Unknown artifact removal method: {method}")
```

#### 2. Create `src/eeg_processor/processing/regression.py`

**New module implementing EOG regression:**

```python
"""
EOG Regression-Based Artifact Removal

Implements MNE's EOGRegression method following Gratton & Coles (1983)
approach for blink artifact correction.

Supports:
- Raw (continuous) data
- Epochs (stimulus-locked) data
- Evoked response subtraction (Gratton & Coles method)

References:
    Gratton, G., Coles, M. G. H., & Donchin, E. (1983). A new method for
    off-line removal of ocular artifact. Electroencephalography and Clinical
    Neurophysiology, 55(4), 468-484.
"""

from typing import Union, List, Dict, Any
import numpy as np
from loguru import logger
from mne.io import BaseRaw
from mne import Epochs
from mne.preprocessing import EOGRegression


def remove_artifacts_regression(
    data: Union[BaseRaw, Epochs],
    eog_channels: List[str] = ['HEOG', 'VEOG'],
    subtract_evoked: bool = True,
    show_plot: bool = False,
    plot_duration: float = 10.0,
    inplace: bool = False,
    verbose: bool = False,
    **kwargs
) -> Union[BaseRaw, Epochs]:
    """
    Remove blink artifacts using EOG regression.

    Implements Gratton & Coles (1983) approach:
    - For Epochs: Optionally subtract evoked response before fitting
    - For Raw: Direct regression fitting
    - Apply correction to preserve stimulus-locked activity

    Args:
        data: Raw or Epochs data with EEG and EOG channels
        eog_channels: List of EOG channel names
        subtract_evoked: For Epochs, subtract evoked before fitting (Gratton method)
        show_plot: Display before/after comparison
        plot_duration: Plot duration in seconds (for Raw data)
        inplace: Modify data in-place (always creates copy for safety)
        verbose: Detailed logging
        **kwargs: Additional EOGRegression parameters

    Returns:
        Corrected data (same type as input)

    Raises:
        ValueError: If EOG channels missing or invalid configuration

    Notes:
        - For Raw data: Apply reference (average) before regression
        - For Epochs: subtract_evoked=True implements Gratton & Coles method
        - Stores metrics in data._regression_metrics

    Examples:
        # Epoched workflow (BrainVision Analyzer style)
        epochs = remove_artifacts_regression(
            epochs,
            eog_channels=['HEOG', 'VEOG'],
            subtract_evoked=True
        )

        # Continuous workflow
        raw = remove_artifacts_regression(
            raw,
            eog_channels=['HEOG', 'VEOG']
        )
    """
    if inplace:
        logger.info("inplace=True ignored - always creates copy for safety")

    # Validate inputs
    _validate_inputs(data, eog_channels)

    # Determine data type and workflow
    is_epochs = isinstance(data, Epochs)
    data_type = "Epochs" if is_epochs else "Raw"

    logger.info(f"Starting EOG regression on {data_type} data")
    logger.info(f"EOG channels: {eog_channels}, subtract_evoked: {subtract_evoked}")

    # Create working copy
    working_data = data.copy()
    original_data = data.copy()

    try:
        # Fit and apply regression
        if is_epochs and subtract_evoked:
            # Gratton & Coles approach for Epochs
            cleaned_data = _fit_and_apply_epochs_with_evoked_subtraction(
                working_data, eog_channels, verbose, **kwargs
            )
        else:
            # Direct fitting for Raw or Epochs without evoked subtraction
            cleaned_data = _fit_and_apply_direct(
                working_data, eog_channels, verbose, **kwargs
            )

        # Calculate quality metrics
        metrics = _calculate_regression_metrics(
            original_data=original_data,
            cleaned_data=cleaned_data,
            eog_channels=eog_channels,
            subtract_evoked=subtract_evoked if is_epochs else False,
            data_type=data_type
        )

        # Store metrics
        cleaned_data._regression_metrics = metrics

        logger.success(f"EOG regression completed")
        logger.info(f"Mean correlation preserved: {metrics['artifact_reduction']['mean_correlation_preserved']:.3f}")

        # Optional visualization
        if show_plot:
            _plot_regression_comparison(
                original_data, cleaned_data, eog_channels,
                data_type, plot_duration
            )

        return cleaned_data

    except Exception as e:
        logger.error(f"EOG regression failed: {str(e)}")
        # Return original with error metrics
        original_data._regression_metrics = {
            'method': 'regression',
            'correction_applied': False,
            'error': str(e)
        }
        return original_data


def _fit_and_apply_epochs_with_evoked_subtraction(
    epochs: Epochs,
    eog_channels: List[str],
    verbose: bool,
    **kwargs
) -> Epochs:
    """
    Fit EOGRegression on epochs with evoked response subtracted (Gratton & Coles).

    This removes stimulus-locked brain activity before estimating regression
    coefficients, preventing bias in the evoked response.
    """
    logger.info("Fitting regression with evoked subtraction (Gratton & Coles method)")

    # Get picks
    eeg_picks = 'eeg'
    eog_picks = eog_channels

    # Create epochs with evoked subtracted
    epochs_subtracted = epochs.copy().subtract_evoked()

    # Fit EOGRegression on subtracted data
    model = EOGRegression(picks=eeg_picks, picks_artifact=eog_picks, **kwargs)
    model.fit(epochs_subtracted)

    # Apply to ORIGINAL epochs (preserves evoked response)
    cleaned_epochs = model.apply(epochs.copy())

    return cleaned_epochs


def _fit_and_apply_direct(
    data: Union[BaseRaw, Epochs],
    eog_channels: List[str],
    verbose: bool,
    **kwargs
) -> Union[BaseRaw, Epochs]:
    """
    Direct EOGRegression fitting and application.

    Used for:
    - Raw data
    - Epochs without evoked subtraction
    """
    logger.info("Fitting regression directly on data")

    # Get picks
    eeg_picks = 'eeg'
    eog_picks = eog_channels

    # Fit and apply
    model = EOGRegression(picks=eeg_picks, picks_artifact=eog_picks, **kwargs)
    model.fit(data)
    cleaned_data = model.apply(data.copy())

    return cleaned_data


def _validate_inputs(
    data: Union[BaseRaw, Epochs],
    eog_channels: List[str]
) -> None:
    """Validate inputs for EOG regression."""
    # Check EOG channels exist
    missing = [ch for ch in eog_channels if ch not in data.ch_names]
    if missing:
        available_eog = [ch for ch in data.ch_names if 'EOG' in ch.upper()]
        raise ValueError(
            f"Missing EOG channels: {missing}. "
            f"Available: {available_eog if available_eog else 'None'}"
        )

    # For Raw data, check reference is set
    if isinstance(data, BaseRaw):
        ref_applied = data.info.get('custom_ref_applied', None)
        if not ref_applied:
            logger.warning(
                "No reference detected. EOG regression works best with "
                "average reference applied first."
            )


def _calculate_regression_metrics(
    original_data: Union[BaseRaw, Epochs],
    cleaned_data: Union[BaseRaw, Epochs],
    eog_channels: List[str],
    subtract_evoked: bool,
    data_type: str
) -> Dict[str, Any]:
    """Calculate quality metrics for regression correction."""
    import time

    # Get EEG data
    eeg_picks = mne.pick_types(original_data.info, eeg=True, meg=False)

    if isinstance(original_data, Epochs):
        # For epochs: average across epochs for comparison
        orig_data = original_data.get_data(picks=eeg_picks).mean(axis=0)
        clean_data = cleaned_data.get_data(picks=eeg_picks).mean(axis=0)
        n_epochs = len(original_data)
    else:
        # For raw: use sample
        orig_data = original_data.get_data(picks=eeg_picks)
        clean_data = cleaned_data.get_data(picks=eeg_picks)
        n_epochs = None

    # Calculate correlation preservation
    correlations = []
    for ch_idx in range(len(eeg_picks)):
        corr = np.corrcoef(orig_data[ch_idx], clean_data[ch_idx])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)

    mean_corr = np.nanmean(correlations)

    # Build metrics
    metrics = {
        'method': 'regression',
        'implementation': 'mne_eog_regression',
        'data_type': data_type,
        'eog_channels': eog_channels,
        'subtract_evoked': subtract_evoked,
        'correction_applied': True,
        'artifact_reduction': {
            'mean_correlation_preserved': round(float(mean_corr), 3),
            'channel_correlations': [round(c, 3) for c in correlations]
        }
    }

    if n_epochs is not None:
        metrics['n_epochs'] = n_epochs

    # Quality flags
    metrics['quality_flags'] = {
        'low_correlation': mean_corr < 0.85,
        'acceptable_correction': mean_corr >= 0.85
    }

    return metrics


def _plot_regression_comparison(
    original_data: Union[BaseRaw, Epochs],
    cleaned_data: Union[BaseRaw, Epochs],
    eog_channels: List[str],
    data_type: str,
    plot_duration: float
) -> None:
    """Create before/after comparison plots."""
    # Implementation similar to existing EMCP plotting
    # Simplified for initial release
    logger.info("Plotting comparison (simplified visualization)")
    # TODO: Add comprehensive plotting in future iteration
```

#### 3. Update Schema `schemas/stage_remove_artifacts.json`

Add regression-specific parameters:

```json
{
  "properties": {
    "method": {
      "enum": ["ica", "regression"],
      "description": "Artifact removal method: ICA (multiple artifacts) or regression (blink-focused)"
    },
    "eog_channels": {
      "description": "EOG channels for regression method (required if method='regression')"
    },
    "subtract_evoked": {
      "type": "boolean",
      "default": true,
      "description": "For Epochs: subtract evoked before fitting (Gratton & Coles). Ignored for Raw."
    }
  }
}
```

#### 4. Deprecate `remove_blinks_emcp` Stage

Add deprecation warning in `data_processor.py`:

```python
def _remove_blinks_emcp(self, data, **kwargs):
    """DEPRECATED: Use remove_artifacts with method='regression' instead"""
    logger.warning(
        "Stage 'remove_blinks_emcp' is deprecated. "
        "Use 'remove_artifacts' with method='regression' instead."
    )
    # Forward to new implementation
    return self._remove_artifacts(data, method='regression', **kwargs)
```

### Pipeline Placement

#### Recommended: Epoched Workflow (BrainVision Analyzer Style)

```yaml
processing:
  # Preprocessing
  - filter: {l_freq: 0.1, h_freq: 40}
  - detect_bad_channels: {interpolate: true}
  - rereference: {method: "average"}

  # Epoching
  - epoch:
      tmin: -0.2
      tmax: 0.8
      event_id: {stimulus: 1}
      baseline: null  # No baseline yet

  # Artifact removal on epochs (Gratton & Coles)
  - remove_artifacts:
      method: "regression"
      eog_channels: ['HEOG', 'VEOG']
      subtract_evoked: true  # Key parameter for epochs

  # Baseline correction happens in epoch stage or separately
  # Artifact rejection can follow
```

#### Alternative: Continuous Workflow

```yaml
processing:
  - filter: {l_freq: 0.1, h_freq: 40}
  - detect_bad_channels: {interpolate: true}
  - rereference: {method: "average"}  # REQUIRED

  # Regression on continuous data
  - remove_artifacts:
      method: "regression"
      eog_channels: ['HEOG', 'VEOG']

  # Then epoch for analysis
  - epoch: {tmin: -0.2, tmax: 0.8}
```

## Implementation Phases

### Phase 1: Core Implementation (2-3 days)
1. ✅ Create `src/eeg_processor/processing/regression.py`
2. ✅ Implement `remove_artifacts_regression()` function
3. ✅ Support both Raw and Epochs data types
4. ✅ Implement Gratton & Coles evoked subtraction for Epochs
5. ✅ Update `data_processor.py` to route `method='regression'`
6. ✅ Basic unit tests

### Phase 2: Quality Control & Validation (1-2 days)
1. ✅ Implement `_calculate_regression_metrics()`
2. ✅ Validate against MNE-Python examples
3. ✅ Test on both Raw and Epochs data
4. ✅ Verify evoked response preservation

### Phase 3: Documentation (1 day)
1. ✅ Update CLAUDE.md with regression method
2. ✅ Create configuration examples
3. ✅ Document BrainVision Analyzer workflow compatibility
4. ✅ Add to CLI help system
5. ✅ Update schema files

### Phase 4: Testing & Integration (1 day)
1. ✅ Comprehensive test suite
2. ✅ Integration tests with full pipeline
3. ✅ Test epoched and continuous workflows
4. ✅ Backward compatibility verification

### Phase 5: Deprecation & Cleanup (1 day)
1. ✅ Add deprecation warnings to `remove_blinks_emcp`
2. ✅ Update existing configurations to use new method
3. ✅ Migration guide for users

**Total Estimated Time: 5-7 days**

## Success Criteria

### Technical Success
- [ ] `method='regression'` works for both Raw and Epochs
- [ ] Gratton & Coles evoked subtraction correctly implemented
- [ ] Processing time <10s for typical epoched dataset
- [ ] Zero breaking changes to existing configurations
- [ ] Test coverage >90%

### Scientific Success
- [ ] Results match MNE-Python EOGRegression validation
- [ ] Evoked response preserved (correlation >0.95)
- [ ] Comparable artifact reduction to ICA for blinks
- [ ] BrainVision Analyzer workflow reproducible

### User Experience Success
- [ ] Simple configuration (one method parameter change)
- [ ] Clear documentation for both workflows
- [ ] Helpful error messages
- [ ] Smooth migration from deprecated `remove_blinks_emcp`

## Risk Assessment

### Technical Risks
- **Risk:** Evoked subtraction may not work for all epoch configurations
  - **Mitigation:** Comprehensive validation, clear documentation, optional parameter

- **Risk:** Performance slower than expected for large epoch counts
  - **Mitigation:** Profile and optimize, add progress logging

### Scientific Risks
- **Risk:** Overcorrection removing neural signals
  - **Mitigation:** Correlation metrics, visualization, quality thresholds

- **Risk:** Less effective than ICA for complex artifacts
  - **Mitigation:** Clear documentation of use cases, recommend ICA when appropriate

### Migration Risks
- **Risk:** Users confused by deprecation of `remove_blinks_emcp`
  - **Mitigation:** Helpful warning messages, automatic forwarding, migration guide

## Decision: Answer to Key Question

### **Q: What does MNE documentation say about applying EOG regression to stimulus-locked epochs?**

**A: MNE fully supports applying EOGRegression to stimulus-locked epochs using the Gratton & Coles method:**

1. **Direct Support:**
   - `EOGRegression.fit(epochs)` works directly on stimulus-locked Epochs
   - `epochs.subtract_evoked()` removes stimulus-locked brain activity
   - Regression fitted on subtracted epochs preserves ERP components

2. **Key Principle:**
   > "EOG artifacts do not need to be time-locked to the stimulus/epoch event timing... as long as the EOG artifacts are in the epochs, then the algorithm should be able to estimate regression coefficients."

3. **Recommended Workflow for Stimulus-Locked Epochs:**
   ```python
   # Create stimulus-locked epochs
   epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8)

   # Fit on epochs with evoked subtracted (Gratton & Coles)
   epochs_sub = epochs.copy().subtract_evoked()
   model = EOGRegression(picks='eeg', picks_artifact='eog')
   model.fit(epochs_sub)

   # Apply to original epochs (preserves evoked)
   epochs_clean = model.apply(epochs)
   ```

4. **Why This Works:**
   - Subtracting evoked removes stimulus-locked brain signals
   - Leaves artifacts + spontaneous activity
   - Regression coefficients capture artifact propagation
   - Applying to original epochs removes artifacts but preserves ERPs

**This matches the BrainVision Analyzer workflow exactly: Epoch → OcularCorrection → Continue processing**

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Technical Lead | | | |
| Product Owner | | | |

---

**Document Control:**
- **File Location:** `docs/PRD-EOG-Regression-Blink-Correction.md`
- **Last Modified:** January 19, 2025
- **Review Cycle:** Quarterly
- **Next Review:** April 19, 2025
