# EOG Regression Implementation Summary

**Quick Reference for Regression Module Development**

---

## Critical Findings for Implementation

### 1. Two Distinct Methodologies

**Direct Regression (for continuous data):**
```python
model = EOGRegression(picks='eeg', picks_artifact='eog')
model.fit(raw)
raw_clean = model.apply(raw)
```

**Gratton Method (for epoched ERP data with evoked subtraction):**
```python
# CRITICAL: Fit on evoked-subtracted data
epochs_no_evoked = epochs.copy().subtract_evoked()
model = EOGRegression(picks='eeg', picks_artifact='eog')
model.fit(epochs_no_evoked)

# Apply to ORIGINAL epochs (preserves ERPs)
epochs_clean = model.apply(epochs)
```

---

## 2. Mathematical Implementation (From MNE Source)

### Fit Phase:
```python
# 1. Mean center artifact channels
ref_data = artifact_data - np.mean(artifact_data, axis=-1, keepdims=True)

# 2. Handle epochs: concatenate trials
if ref_data.ndim == 3:  # Epochs
    ref_data = ref_data.transpose(1, 0, 2)
    ref_data = ref_data.reshape(len(picks_artifact), -1)

# 3. Compute artifact covariance
cov_ref = ref_data @ ref_data.T

# 4. Compute regression coefficients for each EEG channel
for each EEG channel:
    cov_data = EEG_channel - np.mean(EEG_channel)
    coef[i] = np.linalg.solve(cov_ref, ref_data @ cov_data.T)
```

### Apply Phase:
```python
# 1. Mean center artifact channels (same as fit)
ref_data = artifact_data - np.mean(artifact_data, axis=-1, keepdims=True)

# 2. Subtract weighted artifacts from each EEG channel
for each EEG channel:
    EEG_channel -= coef[i] @ ref_data
```

---

## 3. Mandatory Prerequisites

### BEFORE Regression:
1. **EEG Referencing (REQUIRED)**
   ```python
   raw.set_eeg_reference(ref_channels='average', projection=True)
   ```
   - MNE raises RuntimeError if not set
   - Artifact propagation depends on reference scheme

2. **Data Preloading**
   ```python
   raw.load_data()
   ```
   - Regression requires data in memory

3. **EOG Channels Present**
   - Standard: `['HEOG', 'VEOG']`
   - Alternative: `['Fp1', 'Fp2']` (frontal electrodes)

---

## 4. Raw vs Epochs: Critical Differences

### Raw (Continuous) Data:
- **Shape:** `(n_channels, n_samples)`
- **Mean centering:** Removes DC offset and slow drifts
- **Use case:** Resting-state, spontaneous activity
- **Method:** Direct regression

### Epochs (Segmented) Data:
- **Shape:** `(n_epochs, n_channels, n_samples)`
- **Mean centering:** Removes DC offset + **EVOKED RESPONSES**
- **Problem:** If evoked responses present, regression treats brain activity as artifact
- **Solution:** Subtract evoked response before fitting (Gratton method)

---

## 5. Decision Tree: Which Method to Use?

```
Is data continuous (Raw)?
├─ YES → Use direct regression
└─ NO (Epochs) → Continue

    Does data contain stimulus-locked ERPs?
    ├─ NO → Use direct regression
    └─ YES → Continue

        Do frontal channels show evoked responses?
        ├─ NO → Use direct regression
        └─ YES → Use Gratton method (evoked subtraction)
```

---

## 6. Implementation Pattern for EEG Processor

### Suggested Configuration Options:

```yaml
# Option 1: Direct regression (for Raw or non-ERP epochs)
remove_blinks_emcp:
  method: eog_regression
  eog_channels: [HEOG, VEOG]
  picks: eeg

# Option 2: Gratton method (for ERP epochs)
remove_blinks_emcp:
  method: gratton_coles  # Signals evoked subtraction
  eog_channels: [HEOG, VEOG]
  picks: eeg
```

### Processing Logic:

```python
def remove_blinks_emcp(inst, method='eog_regression', eog_channels=None, ...):
    """
    Remove blink artifacts using EOG regression.

    Parameters
    ----------
    method : str
        'eog_regression' - Direct regression
        'gratton_coles' - Evoked subtraction method (for ERP epochs)
    """

    # 1. Validate prerequisites
    if not inst.info.get('custom_ref_applied'):
        raise ValueError("EEG reference must be set before regression")

    if eog_channels is None:
        eog_channels = 'eog'

    # 2. Create regression model
    model = EOGRegression(picks='eeg', picks_artifact=eog_channels)

    # 3. Fit based on method
    if method == 'gratton_coles' and isinstance(inst, mne.Epochs):
        # Gratton method: fit on evoked-subtracted data
        inst_no_evoked = inst.copy().subtract_evoked()
        model.fit(inst_no_evoked)
    else:
        # Direct regression: fit on original data
        model.fit(inst)

    # 4. Apply to original data
    inst_clean = model.apply(inst, copy=True)

    # 5. Store regression coefficients for QC
    return inst_clean, model.coef_
```

---

## 7. Quality Control Metrics

### Validation Checks:

1. **Coefficient Topography:**
   ```python
   model.plot()  # Visualize regression weights
   ```
   - Expect: Decreasing frontal → parietal
   - Larger for horizontal vs vertical movement

2. **Residual Artifact:**
   ```python
   # Compare EOG-EEG correlation before/after
   corr_before = np.corrcoef(eog_channel, eeg_channel)[0, 1]
   corr_after = np.corrcoef(eog_channel, eeg_clean)[0, 1]
   reduction = (corr_before - corr_after) / corr_before * 100
   ```

3. **ERP Preservation (for Gratton method):**
   ```python
   # Compare evoked responses before/after
   evoked_before = epochs.average()
   evoked_after = epochs_clean.average()
   # Verify posterior channels unchanged
   ```

---

## 8. Common Errors and Solutions

### Error 1: "No average reference for the EEG channels"
**Solution:**
```python
inst.set_eeg_reference(ref_channels='average', projection=True)
```

### Error 2: "Selected data channels are not compatible"
**Cause:** Channel mismatch between fit and apply
**Solution:** Ensure identical channel sets and order

### Error 3: Over-correction of genuine brain activity
**Cause:** Fitting on epochs with evoked responses
**Solution:** Use Gratton method with `subtract_evoked()`

---

## 9. Verified Citations

1. **Gratton, G., Coles, M.G., & Donchin, E. (1983).** A new method for off-line removal of ocular artifact. *Electroencephalography and Clinical Neurophysiology*, 55(4), 468-484. [PMID: 6187540]

2. **Croft, R.J., et al. (2005).** EOG correction: a comparison of four methods. *Psychophysiology*, 42(1), 16-24. [PMID: 15720577]

3. **Hoffmann, S., & Falkenstein, M. (2008).** The correction of eye blink artefacts in the EEG: a comparison of two prominent methods. *PLoS ONE*, 3(8), e3004. [PMID: 18714341]

---

## 10. Next Development Steps

- [ ] Implement `_fit_regression()` function (direct method)
- [ ] Implement `_fit_gratton()` function (evoked subtraction)
- [ ] Add prerequisite validation (reference, EOG channels)
- [ ] Create coefficient visualization
- [ ] Add quality metrics (correlation reduction, etc.)
- [ ] Write unit tests with simulated data
- [ ] Document usage examples in module docstring
- [ ] Add to configuration schema

---

**Key Takeaway:** The critical distinction is whether to subtract evoked responses before fitting. For epoched ERP data, the Gratton method (evoked subtraction) is scientifically essential to avoid removing genuine brain activity.
