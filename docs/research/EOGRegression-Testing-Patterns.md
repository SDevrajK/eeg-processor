# EOGRegression Testing Patterns and Strategies

**Research Report**
**Date:** 2025-10-19
**Objective:** Document MNE-Python EOGRegression testing patterns, edge cases, and validation strategies for implementing comprehensive unit tests.

---

## Executive Summary

This report analyzes MNE-Python's test suite for `EOGRegression`, official tutorials, and existing EEG Processor test patterns to guide implementation of comprehensive unit tests for our EOG regression functionality. The research reveals systematic testing approaches covering basic functionality, numerical validation, edge cases, and integration testing.

**Key Findings:**
- MNE's test suite (`test_regress.py`) provides comprehensive coverage of EOGRegression functionality
- Tests emphasize numerical validation using `assert_allclose` with tight tolerances (10+ decimal places)
- Both unit tests (mocked) and integration tests (real data) are essential
- Edge cases include: bad channels, projection requirements, channel ordering, data types (Raw/Epochs/Evoked)
- Quality metrics validation requires independent computation for verification

---

## 1. MNE-Python Test Suite Analysis

### 1.1 Test File Location and Structure

**Primary Test File:**
```
mne/preprocessing/tests/test_regress.py
```

**Key Imports:**
```python
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from mne import pick_types
from mne.datasets import testing
from mne.io import read_raw_fif
from mne.preprocessing import (
    EOGRegression,
    create_eog_epochs,
    read_eog_regression,
    regress_artifact,
)
```

**Test Data:**
```python
data_path = testing.data_path(download=False)
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
```

### 1.2 Core Test Functions

#### Test 1: `test_regress_artifact()` - Basic Regression Functionality

**Purpose:** Validate core regression algorithm and artifact removal

**Key Test Cases:**

1. **Basic artifact regression on epochs:**
```python
epochs = create_eog_epochs(raw)
epochs.apply_baseline((None, None))
orig_data = epochs.get_data("eeg")
orig_norm = np.linalg.norm(orig_data)

epochs_clean, betas = regress_artifact(epochs)
clean_data = epochs_clean.get_data("eeg")
clean_norm = np.linalg.norm(clean_data)

# Validation: Signal should be reduced but not eliminated
assert orig_norm / 2 > clean_norm > orig_norm / 10
```

**Validation Strategy:**
- Measure L2 norm before/after regression
- Ensure signal reduction is within reasonable bounds (2x to 10x reduction)
- Avoid over-correction (signal shouldn't disappear completely)

2. **In-place operation with pre-computed betas:**
```python
regress_artifact(epochs, betas=betas, copy=False)  # inplace
assert_allclose(epochs_clean.get_data(copy=False), epochs.get_data(copy=False))
```

3. **Self-regression validation:**
```python
# Regressing channels onto themselves should make them constant
epochs, betas = regress_artifact(epochs, picks="eog", picks_artifact="eog")
assert np.ptp(epochs.get_data("eog")) < 1e-15  # constant value
assert_allclose(betas, 1)
```

4. **Error handling - invalid beta shape:**
```python
with pytest.raises(ValueError, match=r"Invalid value.*betas\.shape.*"):
    regress_artifact(epochs, betas=betas[:-1])
```

5. **Projection requirements:**
```python
raw.set_eeg_reference(projection=True)
model = EOGRegression(proj=False, picks="meg", picks_artifact="eog")
model.fit(raw)  # Works - projections not required for MEG

model = EOGRegression(proj=False, picks="eeg", picks_artifact="eog")
with pytest.raises(RuntimeError, match="Projections need to be applied"):
    model.fit(raw)  # Fails - EEG requires applied projections
```

6. **Reference requirement for EEG:**
```python
raw.del_proj()
with pytest.raises(RuntimeError, match="No average reference for the EEG"):
    model.fit(raw)
```

#### Test 2: `test_eog_regression()` - EOGRegression Class

**Purpose:** Comprehensive testing of EOGRegression class functionality

**Key Test Patterns:**

1. **String representation validation:**
```python
model = EOGRegression()
assert str(model) == "<EOGRegression | not fitted>"
model.fit(raw)
assert str(model) == "<EOGRegression | fitted to 1 artifact channel>"
```

2. **Coefficient shape validation:**
```python
model.fit(raw)
assert model.coef_.shape == (59, 1)  # 59 EEG channels, 1 EOG channel
```

3. **Signal reduction verification:**
```python
raw_clean = model.apply(raw)
assert np.ptp(raw_clean.get_data("eeg")) < np.ptp(raw.get_data("eeg"))
```

4. **Testing on different data types:**
```python
# On epochs
epochs = create_eog_epochs(raw)
model = EOGRegression().fit(epochs)
epochs = model.apply(epochs)
assert np.ptp(epochs.get_data("eeg")) < 1e-4  # Blinks mostly gone

# On evoked
evoked = epochs.average("all")
model = EOGRegression().fit(evoked)
evoked = model.apply(evoked)
assert np.ptp(evoked.get_data("eeg")) < 1e-4
```

5. **Channel ordering compatibility:**
```python
# Reorder channels - should fail
raw_ = raw.copy().drop_channels(["EEG 001"])
raw_ = raw_.add_channels([raw.copy().pick(["EEG 001"])])
model = EOGRegression().fit(evoked)
with pytest.raises(ValueError, match="data channels are not compatible"):
    model.apply(raw_)
```

6. **In-place vs copy operation:**
```python
raw_ = model.apply(raw, copy=False)
assert raw_ is raw
assert raw_._data is raw._data

raw_ = model.apply(raw, copy=True)
assert raw_ is not raw
assert raw_._data is not raw._data
```

7. **Visualization testing:**
```python
# Single channel type
fig = model.plot()
assert len(fig.axes) == 2  # topomap + colorbar
assert fig.axes[0].title.get_text() == "eeg/EOG 061"

# Multiple channel types
raw_meg_eeg.load_data()
fig = EOGRegression().fit(raw_meg_eeg).plot()
assert len(fig.axes) == 6  # 3 topomaps + 3 colorbars

# Multiple regressors
m = EOGRegression(picks_artifact=["EEG 001", "EOG 061"]).fit(raw_meg_eeg)
assert str(m) == "<EOGRegression | fitted to 2 artifact channels>"
fig = m.plot()
assert len(fig.axes) == 12  # 6 topomaps + 6 colorbars
```

#### Test 3: `test_read_eog_regression()` - Serialization

**Purpose:** Test saving/loading regression models

```python
model = EOGRegression().fit(raw)
model.save(tmp_path / "weights.h5", overwrite=True)
model2 = read_eog_regression(tmp_path / "weights.h5")

# Verify all attributes preserved
assert_array_equal(model.picks, model2.picks)
assert_array_equal(model.picks_artifact, model2.picks_artifact)
assert_array_equal(model.exclude, model2.exclude)
assert_array_equal(model.coef_, model2.coef_)
assert model.proj == model2.proj
assert model.info_.keys() == model2.info_.keys()
```

#### Test 4: `test_regress_artifact_bads()` - Bad Channel Handling

**Purpose:** Ensure bad channels are properly handled during regression

**Key Validation:**

1. **Signal suppression measurement:**
```python
picks = pick_types(raw.info, eeg=True)
norms = np.linalg.norm(raw.get_data(picks), axis=1)
raw_reg, _ = regress_artifact(raw, picks=picks, picks_artifact="eog")
norms_reg = np.linalg.norm(raw_reg.get_data(picks), axis=1)
suppression = 20 * np.log10(norms / norms_reg)
assert_array_less(3, suppression)  # at least 3 dB suppression
```

2. **Bad channel invariance:**
```python
# Adding bad channels shouldn't affect results when picks are supplied
raw.info["bads"] = raw.ch_names[:2] + raw.ch_names[-2:-1]
raw_reg, _ = regress_artifact(raw, picks=picks, picks_artifact="eog")
data_reg_new = raw_reg.get_data()
assert_allclose(data_reg, data_reg_new)
```

### 1.3 Test Markers and Decorators

**Testing Data Requirement:**
```python
@testing.requires_testing_data
def test_eog_regression():
    """Skips test if testing data not available"""
```

---

## 2. MNE-Python Tutorial Examples

### 2.1 Basic Example (`eog_regression.py`)

**Workflow Pattern:**

```python
# 1. Load and filter data
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(0.3, None, picks="all")  # Highpass to eliminate drifts

# 2. Fit regression model
weights = EOGRegression().fit(raw)

# 3. Apply to data
raw_clean = weights.apply(raw, copy=True)

# 4. Visualize weights
weights.plot()

# 5. Compare before/after using evoked potentials
evoked_before = mne.Epochs(raw, events, event_id, tmin, tmax,
                           baseline=(tmin, 0)).average()
evoked_after = mne.Epochs(raw_clean, events, event_id, tmin, tmax,
                          baseline=(tmin, 0)).average()
```

**Key Parameters:**
- Highpass filter: 0.3 Hz (remove slow drifts for stable coefficients)
- Always use `copy=True` when comparing before/after
- Apply same filtering to both EEG and EOG channels

### 2.2 Advanced Tutorial (`35_artifact_correction_regression.py`)

**Scientific Workflow:**

1. **Data Preparation:**
```python
raw.pick(["eeg", "eog", "stim"])
raw.load_data()
raw.set_eeg_reference("average")  # Required before regression
raw.filter(0.3, 40)  # Remove slow drifts
```

2. **Basic Regression:**
```python
model_plain = EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs)
epochs_clean_plain = model_plain.apply(epochs)
epochs_clean_plain.apply_baseline()  # Redo baseline after regression
```

3. **Gratton et al. (1983) Method - Subtract Evoked:**
```python
# Subtract evoked to leave mostly noise + EOG artifacts
epochs_sub = epochs.copy().subtract_evoked()

# Fit on subtracted epochs
model_sub = EOGRegression(picks="eeg", picks_artifact="eog").fit(epochs_sub)

# Apply to original epochs
epochs_clean_sub = model_sub.apply(epochs).apply_baseline()
```

**Rationale:** Removing evoked response leaves mostly noise, so EOG artifacts dominate and regression coefficients are more robust.

4. **Croft & Barry (2000) Method - EOG Evoked:**
```python
# Create epochs time-locked to blinks
eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_evoked = eog_epochs.average("all")

# Fit on blink-evoked response
model_evoked = EOGRegression(picks="eeg", picks_artifact="eog").fit(eog_evoked)

# Apply to regular epochs
epochs_clean_evoked = model_evoked.apply(epochs).apply_baseline()
```

**Rationale:** Blink-locked averaging suppresses non-time-locked EEG, amplifying EOG artifacts for better coefficient estimation.

5. **Visualization Comparison:**
```python
plot_kwargs = dict(picks="all", ylim=dict(eeg=(-10, 10), eog=(-5, 15)))
epochs.average("all").plot(**plot_kwargs)  # Before
epochs_clean.average("all").plot(**plot_kwargs)  # After
```

---

## 3. Existing EEG Processor Test Patterns

### 3.1 Test Organization Structure

**Test File Hierarchy:**
```
tests/
├── test_emcp.py                 # Unit tests with mocks
├── test_emcp_validation.py      # Validation against MNE reference
├── test_emcp_integration.py     # Integration tests
├── test_utils/
│   └── mne_sample_data.py      # Test data management
└── fixtures/
    └── test_multi_event_config.yml
```

### 3.2 Mock Data Creation Patterns

**Pattern 1: Basic Mock Raw with EOG**

```python
def _create_mock_raw_with_eog(self):
    """Create mock Raw object with EEG and EOG channels."""
    n_channels = 10
    n_times = 1000
    sfreq = 250.0

    ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
    ch_types = ['eeg'] * 8 + ['eog'] * 2

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.randn(n_channels, n_times) * 1e-5

    return RawArray(data, info)
```

**Pattern 2: Mock Raw with Simulated Blinks**

```python
def _create_mock_raw_with_blinks(self):
    """Create mock raw data with simulated blinks."""
    n_channels = 10
    n_times = 1000
    sfreq = 250.0

    ch_names = [f'EEG{i:03d}' for i in range(1, 9)] + ['HEOG', 'VEOG']
    ch_types = ['eeg'] * 8 + ['eog'] * 2
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    data = np.random.randn(n_channels, n_times) * 1e-5

    # Add simulated blinks with known correlation pattern
    blink_times = [100, 300, 500]
    for blink_time in blink_times:
        if blink_time < n_times - 50:
            # VEOG blink (exponential decay)
            veog_artifact = -100e-6 * np.exp(-0.1 * np.arange(50))
            data[9, blink_time:blink_time+50] += veog_artifact

            # Correlated artifacts in EEG with known coefficients
            for eeg_ch in range(8):
                beta = 0.1 + 0.05 * eeg_ch  # Channel-specific correlation
                data[eeg_ch, blink_time:blink_time+50] += beta * veog_artifact

    return RawArray(data, info)
```

**Pattern 3: Mock Raw Without EOG**

```python
def _create_mock_raw_without_eog(self):
    """Create mock Raw object without EOG channels."""
    n_channels = 8
    n_times = 1000
    sfreq = 250.0

    ch_names = [f'EEG{i:03d}' for i in range(1, 9)]
    ch_types = ['eeg'] * 8

    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.random.randn(n_channels, n_times) * 1e-5

    return RawArray(data, info)
```

### 3.3 Test Class Organization

**Unit Test Classes:**

1. **TestEMCPValidation** - Input validation tests
2. **TestEMCPMetrics** - Quality metrics calculation
3. **TestEOGRegressionMethod** - EOG regression functionality
4. **TestGrattonColesMethod** - Gratton & Coles implementation
5. **TestEMCPQualitySummary** - Quality summary functions
6. **TestEMCPErrorHandling** - Error handling and edge cases

**Example Class Structure:**

```python
class TestEOGRegressionMethod:
    """Test EOG regression method implementation."""

    @patch('eeg_processor.processing.emcp.find_eog_events')
    @patch('eeg_processor.processing.emcp.EOGRegression')
    def test_eog_regression_basic_functionality(self, mock_eog_regression, mock_find_events):
        """Test basic EOG regression functionality with mocks."""
        raw = self._create_mock_raw_with_blinks()

        # Mock blink events
        mock_events = np.array([[100, 0, 1], [300, 0, 1], [500, 0, 1]])
        mock_find_events.return_value = mock_events

        # Mock EOGRegression behavior
        mock_regressor = MagicMock()
        mock_regressor.apply.return_value = raw.copy()
        mock_eog_regression.return_value = mock_regressor

        result = remove_blinks_eog_regression(
            raw=raw,
            eog_channels=['HEOG', 'VEOG'],
            show_plot=False,
            verbose=False
        )

        # Verify method calls
        mock_find_events.assert_called_once()
        mock_eog_regression.assert_called_once()
        mock_regressor.fit.assert_called_once()
        mock_regressor.apply.assert_called_once()

        # Check result metrics
        assert hasattr(result, '_emcp_metrics')
        assert result._emcp_metrics['method'] == 'eog_regression'
```

### 3.4 Validation Test Patterns

**Pattern 1: Numerical Validation Against MNE Reference**

```python
def test_eog_regression_matches_mne_reference(self, sample_raw, eog_channels):
    """Test that our EOG regression produces identical results to MNE."""

    # Apply our method
    our_result = remove_blinks_eog_regression(
        sample_raw.copy(),
        eog_channels=eog_channels,
        show_plot=False,
        verbose=False
    )

    # Apply MNE reference
    eeg_picks = mne.pick_types(sample_raw.info, eeg=True, meg=False)
    eog_picks = [sample_raw.ch_names.index(ch) for ch in eog_channels]

    mne_regressor = EOGRegression(picks=eeg_picks, picks_artifact=eog_picks)
    mne_regressor.fit(sample_raw)
    mne_result = mne_regressor.apply(sample_raw.copy())

    # Compare with tight numerical tolerance
    our_eeg_data = our_result.get_data(picks=eeg_picks)
    mne_eeg_data = mne_result.get_data(picks=eeg_picks)

    np.testing.assert_array_almost_equal(
        our_eeg_data,
        mne_eeg_data,
        decimal=10,
        err_msg="EOG regression results do not match MNE reference"
    )
```

**Pattern 2: Data Integrity Validation**

```python
def test_emcp_methods_preserve_data_integrity(self, sample_raw, eog_channels):
    """Test that EMCP methods preserve data integrity."""

    original_raw = sample_raw.copy()

    result = remove_blinks_eog_regression(
        original_raw.copy(),
        eog_channels=eog_channels,
        show_plot=False,
        verbose=False
    )

    # Check dimensions preserved
    assert result.get_data().shape == original_raw.get_data().shape
    assert result.info['sfreq'] == original_raw.info['sfreq']
    assert result.ch_names == original_raw.ch_names

    # Check EOG channels unchanged
    eog_picks = [result.ch_names.index(ch) for ch in eog_channels]
    np.testing.assert_array_equal(
        result.get_data(picks=eog_picks),
        original_raw.get_data(picks=eog_picks)
    )

    # Check EEG data was modified
    eeg_picks = mne.pick_types(result.info, eeg=True)
    assert not np.array_equal(
        original_raw.get_data(picks=eeg_picks),
        result.get_data(picks=eeg_picks)
    )

    # Check high correlation (not over-corrected)
    correlations = []
    for ch_idx in range(len(eeg_picks)):
        orig = original_raw.get_data(picks=eeg_picks)[ch_idx]
        clean = result.get_data(picks=eeg_picks)[ch_idx]
        corr = np.corrcoef(orig, clean)[0, 1]
        correlations.append(corr)

    mean_corr = np.mean(correlations)
    assert mean_corr > 0.8, f"Mean correlation {mean_corr} too low"
```

**Pattern 3: Quality Metrics Validation**

```python
def test_emcp_quality_metrics_accuracy(self, sample_raw, eog_channels):
    """Test that quality metrics accurately reflect processing."""

    cleaned_raw = remove_blinks_gratton_coles(
        sample_raw.copy(),
        eog_channels=eog_channels,
        show_plot=False,
        verbose=False
    )

    metrics = get_emcp_quality_summary(cleaned_raw)

    # Independently verify blink detection
    primary_eog = eog_channels[0]
    eog_events = find_eog_events(sample_raw, ch_name=primary_eog, verbose=False)
    expected_blinks = len(eog_events)

    assert metrics['blink_events'] == expected_blinks

    # Independently verify correlation
    eeg_picks = mne.pick_types(sample_raw.info, eeg=True)
    original_eeg = sample_raw.get_data(picks=eeg_picks)
    cleaned_eeg = cleaned_raw.get_data(picks=eeg_picks)

    manual_correlations = []
    for ch_idx in range(len(eeg_picks)):
        corr = np.corrcoef(original_eeg[ch_idx], cleaned_eeg[ch_idx])[0, 1]
        if not np.isnan(corr):
            manual_correlations.append(corr)

    manual_mean_corr = np.mean(manual_correlations)
    stored_mean_corr = metrics['mean_correlation']

    np.testing.assert_almost_equal(
        stored_mean_corr,
        manual_mean_corr,
        decimal=3
    )
```

### 3.5 Pytest Fixtures

**Sample Data Fixture:**

```python
@pytest.fixture(scope="class")
def sample_raw(self):
    """Load MNE sample data for validation testing."""
    if not MNE_AVAILABLE:
        pytest.skip("MNE not available")

    manager = get_sample_manager()
    if not manager.is_available():
        pytest.skip("MNE sample data not available")

    raw = get_sample_raw(preload=True)
    if raw is None:
        pytest.skip("Failed to load MNE sample data")

    # Prepare data for testing
    raw.pick_types(eeg=True, eog=True, meg=False, stim=False)
    raw.crop(tmax=60)  # Use first 60 seconds
    raw.set_eeg_reference('average', projection=True)

    return raw

@pytest.fixture
def eog_channels(self, sample_raw):
    """Get available EOG channels from sample data."""
    eog_channels = [ch for ch in sample_raw.ch_names if 'EOG' in ch]
    if not eog_channels:
        pytest.skip("No EOG channels found")
    return eog_channels
```

### 3.6 Test Data Management

**MNE Sample Data Manager Pattern:**

```python
class MNESampleDataManager:
    """Manages MNE sample dataset download and access."""

    def __init__(self, test_data_dir: Optional[Union[str, Path]] = None):
        if test_data_dir is None:
            self.test_data_dir = Path(__file__).parent.parent / "test_data"
        else:
            self.test_data_dir = Path(test_data_dir)

        self.mne_sample_dir = self.test_data_dir / "mne_sample"
        self.mne_sample_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """Check if MNE sample data is available locally."""
        sample_file = self.get_sample_raw_file()
        return sample_file is not None and sample_file.exists()

    def download_sample_data(self, force: bool = False) -> bool:
        """Download MNE sample dataset if not available."""
        # Implementation details...
```

---

## 4. Edge Cases and Validation Strategies

### 4.1 Critical Edge Cases

#### 1. Missing or Invalid EOG Channels

**Test Pattern:**
```python
def test_missing_eog_channels(self):
    """Test handling when EOG channels are missing."""
    raw = create_raw_without_eog()

    with pytest.raises(ValueError, match="Missing EOG channels"):
        remove_blinks_eog_regression(
            raw=raw,
            eog_channels=['HEOG', 'VEOG']
        )

def test_empty_eog_channel_list(self):
    """Test validation of empty channel list."""
    raw = create_raw_with_eog()

    with pytest.raises(ValueError, match="At least one EOG channel"):
        remove_blinks_eog_regression(
            raw=raw,
            eog_channels=[]
        )
```

#### 2. No Blinks Detected

**Test Pattern:**
```python
@patch('find_eog_events')
def test_no_blinks_detected(self, mock_find_events):
    """Test when no blink events are found."""
    raw = create_raw_with_eog()

    # Mock no blink events
    mock_find_events.return_value = np.array([]).reshape(0, 3)

    result = remove_blinks_eog_regression(
        raw=raw,
        eog_channels=['VEOG']
    )

    # Should return original data with appropriate metrics
    assert hasattr(result, '_emcp_metrics')
    assert result._emcp_metrics['blink_events_found'] == 0
    assert result._emcp_metrics['correction_applied'] is False
```

#### 3. Projection Requirements

**Test Pattern:**
```python
def test_projection_requirements_eeg(self):
    """Test that EEG requires applied projections."""
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()
    raw.del_proj()
    raw.set_eeg_reference(projection=True)

    model = EOGRegression(proj=False, picks="eeg", picks_artifact="eog")

    with pytest.raises(RuntimeError, match="Projections need to be applied"):
        model.fit(raw)

def test_projection_requirements_meg(self):
    """Test that MEG doesn't require projections."""
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()

    model = EOGRegression(proj=False, picks="meg", picks_artifact="eog")
    model.fit(raw)  # Should work without error
```

#### 4. Reference Requirements

**Test Pattern:**
```python
def test_eeg_reference_required(self):
    """Test that EEG requires average reference."""
    raw = read_raw_fif(raw_fname).load_data()
    raw.del_proj()

    model = EOGRegression(proj=False, picks="eeg", picks_artifact="eog")

    with pytest.raises(RuntimeError, match="No average reference for the EEG"):
        model.fit(raw)
```

#### 5. Channel Ordering Compatibility

**Test Pattern:**
```python
def test_channel_ordering_mismatch(self):
    """Test that channel ordering must match between fit and apply."""
    raw = read_raw_fif(raw_fname).pick(["eeg", "eog"]).load_data()

    # Fit on original ordering
    model = EOGRegression().fit(raw)

    # Reorder channels
    raw_reordered = raw.copy().drop_channels(["EEG 001"])
    raw_reordered = raw_reordered.add_channels([raw.copy().pick(["EEG 001"])])

    with pytest.raises(ValueError, match="data channels are not compatible"):
        model.apply(raw_reordered)
```

#### 6. Bad Channels Handling

**Test Pattern:**
```python
def test_bad_channels_invariance(self):
    """Test that bad channels don't affect regression when picks specified."""
    raw = create_raw_with_eog()
    picks = pick_types(raw.info, eeg=True)

    # Regression without bad channels
    raw_reg1, _ = regress_artifact(raw.copy(), picks=picks, picks_artifact="eog")

    # Add bad channels
    raw.info["bads"] = raw.ch_names[:2] + raw.ch_names[-2:-1]

    # Regression with bad channels
    raw_reg2, _ = regress_artifact(raw.copy(), picks=picks, picks_artifact="eog")

    # Results should be identical
    assert_allclose(raw_reg1.get_data(), raw_reg2.get_data())
```

#### 7. Data Type Compatibility

**Test Pattern:**
```python
@pytest.mark.parametrize("data_type", ["raw", "epochs", "evoked"])
def test_data_type_compatibility(self, data_type):
    """Test EOGRegression works with Raw, Epochs, and Evoked."""
    raw = create_raw_with_eog()

    if data_type == "raw":
        data = raw
    elif data_type == "epochs":
        events = create_synthetic_events(raw)
        data = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, preload=True)
    else:  # evoked
        events = create_synthetic_events(raw)
        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, preload=True)
        data = epochs.average()

    model = EOGRegression().fit(data)
    result = model.apply(data)

    assert result is not None
    assert hasattr(model, 'coef_')
```

#### 8. Insufficient Data Length

**Test Pattern:**
```python
def test_insufficient_data_length(self):
    """Test handling of very short data segments."""
    raw = create_raw_with_eog()
    raw.crop(tmax=0.1)  # Only 100ms

    # Should either work or provide meaningful error
    # (Implementation-dependent)
```

#### 9. NaN/Inf Values in Data

**Test Pattern:**
```python
def test_nan_in_eog_channels(self):
    """Test handling of NaN values in EOG data."""
    raw = create_raw_with_eog()

    # Introduce NaN values
    data = raw.get_data()
    data[-1, 50:100] = np.nan  # NaN in EOG channel

    # Should either clean NaN or raise informative error
    with pytest.raises((ValueError, RuntimeError), match="NaN|invalid"):
        model = EOGRegression().fit(raw)
```

#### 10. Beta Coefficient Shape Mismatch

**Test Pattern:**
```python
def test_beta_shape_mismatch(self):
    """Test error handling for mismatched beta shapes."""
    raw = create_raw_with_eog()
    epochs = create_eog_epochs(raw)

    epochs_clean, betas = regress_artifact(epochs)

    # Wrong shape
    with pytest.raises(ValueError, match="Invalid value.*betas.shape"):
        regress_artifact(epochs, betas=betas[:-1])
```

### 4.2 Validation Approaches

#### 1. Signal Reduction Validation

```python
# L2 norm approach
orig_norm = np.linalg.norm(original_data)
clean_norm = np.linalg.norm(cleaned_data)
reduction_factor = orig_norm / clean_norm
assert 2 < reduction_factor < 10  # Reasonable reduction

# Peak-to-peak approach
orig_ptp = np.ptp(original_data)
clean_ptp = np.ptp(cleaned_data)
assert clean_ptp < orig_ptp

# dB suppression approach
suppression_db = 20 * np.log10(orig_norm / clean_norm)
assert suppression_db > 3  # At least 3 dB suppression
```

#### 2. Correlation Validation

```python
# Channel-wise correlation
correlations = []
for ch_idx in range(n_channels):
    corr = np.corrcoef(original[ch_idx], cleaned[ch_idx])[0, 1]
    if not np.isnan(corr):
        correlations.append(corr)

mean_corr = np.mean(correlations)
assert 0.7 <= mean_corr <= 1.0  # High correlation (minimal over-correction)
```

#### 3. Regression Coefficient Validation

```python
# Coefficient magnitude
assert 0 < np.max(np.abs(model.coef_)) < 1.0

# Coefficient shape
n_picks = len(mne.pick_types(raw.info, eeg=True))
n_artifact_picks = len(eog_channels)
assert model.coef_.shape == (n_picks, n_artifact_picks)

# Spatial pattern plausibility
# (Frontal channels should have higher coefficients for blinks)
```

#### 4. Self-Regression Test

```python
# Regressing channels onto themselves should produce identity
model = EOGRegression(picks="eog", picks_artifact="eog").fit(raw)
assert_allclose(model.coef_, 1.0, rtol=1e-10)

# Result should be constant
result = model.apply(raw)
assert np.ptp(result.get_data("eog")) < 1e-15
```

---

## 5. Test Data Creation Recommendations

### 5.1 Synthetic Data with Known Properties

**Advantages:**
- Full control over artifact characteristics
- Known ground truth for validation
- Fast execution (no file I/O)
- No external dependencies

**Implementation:**

```python
def create_synthetic_eeg_with_blinks(
    n_eeg_channels: int = 8,
    n_eog_channels: int = 2,
    n_times: int = 1000,
    sfreq: float = 250.0,
    n_blinks: int = 5,
    blink_amplitude: float = 100e-6,
    propagation_coeffs: Optional[np.ndarray] = None,
    noise_level: float = 1e-5
) -> mne.io.RawArray:
    """
    Create synthetic EEG data with known blink artifacts.

    Parameters
    ----------
    n_eeg_channels : int
        Number of EEG channels
    n_eog_channels : int
        Number of EOG channels (1 or 2)
    n_times : int
        Number of time samples
    sfreq : float
        Sampling frequency in Hz
    n_blinks : int
        Number of blinks to simulate
    blink_amplitude : float
        Amplitude of blink artifacts in V
    propagation_coeffs : array_like, shape (n_eeg_channels,)
        Known propagation coefficients from EOG to each EEG channel
        If None, random coefficients between 0.1 and 0.5
    noise_level : float
        Background noise level in V

    Returns
    -------
    raw : mne.io.RawArray
        Synthetic raw data with blinks
    """
    # Create channel info
    ch_names = ([f'EEG{i:03d}' for i in range(1, n_eeg_channels + 1)] +
                [f'EOG{i:03d}' for i in range(1, n_eog_channels + 1)])
    ch_types = ['eeg'] * n_eeg_channels + ['eog'] * n_eog_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Initialize data with background noise
    data = np.random.randn(n_eeg_channels + n_eog_channels, n_times) * noise_level

    # Generate propagation coefficients if not provided
    if propagation_coeffs is None:
        # Frontal channels have higher coefficients
        propagation_coeffs = np.linspace(0.5, 0.1, n_eeg_channels)

    # Generate blink times
    min_isi = int(0.5 * sfreq)  # Minimum 500ms between blinks
    blink_duration = int(0.2 * sfreq)  # 200ms blink duration

    blink_times = []
    current_time = min_isi
    for _ in range(n_blinks):
        if current_time + blink_duration < n_times:
            blink_times.append(current_time)
            current_time += min_isi + np.random.randint(0, int(2 * sfreq))

    # Generate blink waveforms
    for blink_time in blink_times:
        t = np.arange(blink_duration)

        # VEOG blink (negative deflection with exponential decay)
        veog_waveform = -blink_amplitude * np.exp(-0.1 * t)

        # Add to primary EOG channel
        eog_idx = n_eeg_channels  # First EOG channel
        data[eog_idx, blink_time:blink_time + blink_duration] += veog_waveform

        # Propagate to EEG channels with known coefficients
        for eeg_ch in range(n_eeg_channels):
            data[eeg_ch, blink_time:blink_time + blink_duration] += (
                propagation_coeffs[eeg_ch] * veog_waveform
            )

    # Create RawArray
    raw = mne.io.RawArray(data, info)

    # Store ground truth as metadata
    raw._blink_times = blink_times
    raw._propagation_coeffs = propagation_coeffs

    return raw
```

**Usage in Tests:**

```python
def test_known_coefficient_recovery(self):
    """Test that regression recovers known propagation coefficients."""
    # Create synthetic data with known coefficients
    true_coeffs = np.array([0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05])
    raw = create_synthetic_eeg_with_blinks(
        n_eeg_channels=8,
        n_eog_channels=1,
        propagation_coeffs=true_coeffs,
        n_blinks=10
    )

    # Fit EOG regression
    model = EOGRegression(picks='eeg', picks_artifact='eog').fit(raw)

    # Recovered coefficients should match ground truth
    np.testing.assert_allclose(
        model.coef_.flatten(),
        true_coeffs,
        rtol=0.1,  # 10% tolerance (noise effects)
        err_msg="Failed to recover known propagation coefficients"
    )
```

### 5.2 MNE Sample Dataset

**Advantages:**
- Real EEG data with authentic artifacts
- Established ground truth from MNE tutorials
- Standard reference for validation

**Setup Pattern:**

```python
# Download and cache sample data
manager = MNESampleDataManager()
if not manager.is_available():
    manager.download_sample_data()

# Load for testing
raw = manager.get_sample_raw(preload=True)
raw.pick_types(eeg=True, eog=True)
raw.crop(tmax=60)  # Use subset for faster tests
raw.set_eeg_reference('average', projection=True)
```

### 5.3 Parametrized Test Data

**Pattern:**

```python
@pytest.mark.parametrize("n_channels,n_blinks,expected_reduction", [
    (8, 5, 2.0),      # Standard case
    (16, 10, 2.5),    # More channels, more blinks
    (4, 2, 1.5),      # Minimal case
    (32, 20, 3.0),    # High-density array
])
def test_parametrized_correction(self, n_channels, n_blinks, expected_reduction):
    """Test correction across different data configurations."""
    raw = create_synthetic_eeg_with_blinks(
        n_eeg_channels=n_channels,
        n_blinks=n_blinks
    )

    model = EOGRegression().fit(raw)
    cleaned = model.apply(raw, copy=True)

    orig_norm = np.linalg.norm(raw.get_data('eeg'))
    clean_norm = np.linalg.norm(cleaned.get_data('eeg'))
    reduction = orig_norm / clean_norm

    assert reduction >= expected_reduction
```

---

## 6. Performance Benchmarking Approaches

### 6.1 Timing Tests

```python
import time
import pytest

@pytest.mark.benchmark
def test_eog_regression_performance(self, benchmark_data):
    """Benchmark EOG regression fitting and application."""
    raw = create_large_raw_dataset()  # e.g., 64 channels, 60s

    # Time fitting
    start = time.time()
    model = EOGRegression().fit(raw)
    fit_time = time.time() - start

    # Time application
    start = time.time()
    cleaned = model.apply(raw, copy=True)
    apply_time = time.time() - start

    # Performance assertions
    assert fit_time < 5.0, f"Fitting took {fit_time}s (expected < 5s)"
    assert apply_time < 1.0, f"Application took {apply_time}s (expected < 1s)"

    # Log for regression tracking
    print(f"Fit: {fit_time:.3f}s, Apply: {apply_time:.3f}s")
```

### 6.2 Memory Profiling

```python
import tracemalloc

def test_memory_usage(self):
    """Test that memory usage is reasonable during regression."""
    raw = create_large_raw_dataset()

    tracemalloc.start()

    # Measure peak memory during fit
    model = EOGRegression().fit(raw)
    current, peak = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # Memory should be reasonable (< 500 MB for typical dataset)
    assert peak < 500 * 1024 * 1024, f"Peak memory: {peak / 1024**2:.1f} MB"
```

### 6.3 Scalability Tests

```python
@pytest.mark.parametrize("n_channels", [8, 16, 32, 64, 128])
def test_scalability_with_channels(self, n_channels):
    """Test that performance scales linearly with channel count."""
    raw = create_synthetic_eeg_with_blinks(n_eeg_channels=n_channels)

    start = time.time()
    model = EOGRegression().fit(raw)
    elapsed = time.time() - start

    # Should scale approximately linearly
    expected_time = 0.01 * n_channels  # ~10ms per channel
    assert elapsed < expected_time * 2  # Allow 2x overhead
```

---

## 7. Recommended Test Implementation Strategy

### 7.1 Test Coverage Priorities

**Priority 1 (Essential):**
1. Basic functionality test (fit, apply, copy behavior)
2. Numerical validation against MNE reference
3. Data integrity preservation
4. Input validation (missing channels, empty lists)
5. Error handling (no blinks, bad data)

**Priority 2 (Important):**
6. Coefficient shape and magnitude validation
7. Bad channel handling
8. Projection and reference requirements
9. Data type compatibility (Raw, Epochs, Evoked)
10. Quality metrics accuracy

**Priority 3 (Nice-to-have):**
11. Visualization tests (plot methods)
12. Serialization (save/load)
13. Performance benchmarks
14. Channel ordering compatibility
15. Self-regression test

### 7.2 Recommended Test File Structure

```
tests/
├── test_eog_regression_unit.py          # Unit tests with mocks
│   ├── TestBasicFunctionality
│   ├── TestInputValidation
│   ├── TestCoefficientComputation
│   ├── TestCopyBehavior
│   └── TestErrorHandling
│
├── test_eog_regression_validation.py    # Validation against MNE
│   ├── TestNumericalAccuracy
│   ├── TestDataIntegrity
│   ├── TestQualityMetrics
│   └── TestReferenceComparison
│
├── test_eog_regression_integration.py   # Integration tests
│   ├── TestWithRealData
│   ├── TestPipelineIntegration
│   └── TestWorkflows
│
└── test_utils/
    ├── synthetic_data.py                # Synthetic data generators
    └── mne_sample_data.py              # Sample data management
```

### 7.3 Pytest Configuration

**pytest.ini:**
```ini
[pytest]
markers =
    unit: Unit tests with mocks (fast, no external dependencies)
    validation: Validation tests against MNE reference (requires sample data)
    integration: Integration tests with real data (slow)
    benchmark: Performance benchmark tests

# Test discovery
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Coverage
addopts =
    --strict-markers
    --verbose
    --tb=short
    --cov=src/eeg_processor
    --cov-report=html
    --cov-report=term-missing
```

**Example Test Invocations:**

```bash
# Run all tests
pytest tests/

# Run only unit tests (fast)
pytest tests/ -m unit

# Run validation tests (requires sample data)
pytest tests/ -m validation

# Run specific test file
pytest tests/test_eog_regression_unit.py -v

# Run with coverage
pytest tests/ --cov=src/eeg_processor/processing/emcp --cov-report=html

# Run benchmarks
pytest tests/ -m benchmark --benchmark-only
```

### 7.4 Pytest Parametrization Strategy

```python
@pytest.mark.parametrize("data_type", ["raw", "epochs", "evoked"])
@pytest.mark.parametrize("n_eog_channels", [1, 2])
@pytest.mark.parametrize("copy", [True, False])
def test_eog_regression_combinations(self, data_type, n_eog_channels, copy):
    """Test EOG regression across multiple parameter combinations."""
    # Create appropriate data
    raw = create_synthetic_eeg_with_blinks(n_eog_channels=n_eog_channels)

    if data_type == "epochs":
        data = create_epochs_from_raw(raw)
    elif data_type == "evoked":
        data = create_epochs_from_raw(raw).average()
    else:
        data = raw

    # Run test
    model = EOGRegression().fit(data)
    result = model.apply(data, copy=copy)

    # Validate based on copy parameter
    if copy:
        assert result is not data
    else:
        assert result is data
```

---

## 8. Key Takeaways

### 8.1 Testing Best Practices from MNE

1. **Use tight numerical tolerances** for validation:
   - `assert_array_almost_equal(decimal=10)` for critical comparisons
   - `assert_allclose(rtol=1e-10)` for relative comparisons

2. **Test both functionality and behavior**:
   - Not just "does it work" but "does it work correctly"
   - Verify coefficient shapes, magnitudes, and spatial patterns

3. **Separate unit and integration tests**:
   - Unit tests with mocks for fast iteration
   - Integration tests with real data for validation

4. **Validate signal quality metrics**:
   - Signal reduction (L2 norm, dB suppression)
   - Correlation preservation
   - Artifact-specific metrics (blink detection)

5. **Test edge cases explicitly**:
   - Missing channels, no events, bad channels
   - Projection/reference requirements
   - Data type compatibility

### 8.2 Critical Validation Patterns

1. **Numerical Validation Against MNE:**
   ```python
   np.testing.assert_array_almost_equal(our_result, mne_result, decimal=10)
   ```

2. **Data Integrity Preservation:**
   ```python
   assert result.shape == original.shape
   assert_array_equal(result['eog'], original['eog'])  # EOG unchanged
   assert not np.array_equal(result['eeg'], original['eeg'])  # EEG modified
   ```

3. **Quality Metrics Validation:**
   ```python
   # Independent computation
   manual_metric = compute_metric_independently(data)
   stored_metric = get_stored_metric(data)
   np.testing.assert_almost_equal(manual_metric, stored_metric, decimal=3)
   ```

4. **Signal Reduction Bounds:**
   ```python
   reduction = orig_norm / clean_norm
   assert 2 < reduction < 10  # Reasonable reduction range
   ```

### 8.3 Recommended pytest Fixtures

```python
@pytest.fixture
def synthetic_raw():
    """Create synthetic data with known properties."""
    return create_synthetic_eeg_with_blinks(n_blinks=5)

@pytest.fixture
def synthetic_raw_no_blinks():
    """Create synthetic data without blinks."""
    return create_synthetic_eeg_with_blinks(n_blinks=0)

@pytest.fixture
def synthetic_raw_no_eog():
    """Create synthetic data without EOG channels."""
    return create_eeg_only_data()

@pytest.fixture(scope="session")
def mne_sample_raw():
    """Load MNE sample data (cached for session)."""
    manager = get_sample_manager()
    raw = manager.get_sample_raw(preload=True)
    raw.pick_types(eeg=True, eog=True)
    raw.crop(tmax=60)
    raw.set_eeg_reference('average', projection=True)
    return raw

@pytest.fixture
def eog_channels(mne_sample_raw):
    """Get EOG channel names from sample data."""
    return [ch for ch in mne_sample_raw.ch_names if 'EOG' in ch]
```

---

## 9. References

### 9.1 MNE-Python Resources

- **Test File:** `mne/preprocessing/tests/test_regress.py`
  - [GitHub Link](https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/tests/test_regress.py)

- **Tutorial:** "Repairing artifacts with regression"
  - [Documentation](https://mne.tools/stable/auto_tutorials/preprocessing/35_artifact_correction_regression.html)

- **Example:** "Reduce EOG artifacts through regression"
  - [Documentation](https://mne.tools/stable/auto_examples/preprocessing/eog_regression.html)

- **API Reference:** `mne.preprocessing.EOGRegression`
  - [Documentation](https://mne.tools/stable/generated/mne.preprocessing.EOGRegression.html)

### 9.2 Scientific References

- **Gratton et al. (1983)**: Gratton, G., Coles, M. G., & Donchin, E. (1983). A new method for off-line removal of ocular artifact. *Electroencephalography and clinical neurophysiology*, 55(4), 468-484.

- **Croft & Barry (2000)**: Croft, R. J., & Barry, R. J. (2000). Removal of ocular artifact from the EEG: a review. *Neurophysiologie Clinique/Clinical Neurophysiology*, 30(1), 5-19.

### 9.3 Internal EEG Processor Resources

- **Existing Tests:**
  - `/home/sdevrajk/projects/eeg-processor/tests/test_emcp.py`
  - `/home/sdevrajk/projects/eeg-processor/tests/test_emcp_validation.py`
  - `/home/sdevrajk/projects/eeg-processor/tests/test_emcp_integration.py`

- **Test Utilities:**
  - `/home/sdevrajk/projects/eeg-processor/tests/test_utils/mne_sample_data.py`

---

## 10. Next Steps

### 10.1 Immediate Actions

1. **Create synthetic data generator** for EOG regression tests
   - Implement `create_synthetic_eeg_with_blinks()` function
   - Add ground truth coefficient tracking

2. **Implement unit test suite** (`test_eog_regression_unit.py`)
   - Basic functionality tests
   - Input validation tests
   - Error handling tests

3. **Implement validation test suite** (`test_eog_regression_validation.py`)
   - Numerical comparison with MNE reference
   - Data integrity tests
   - Quality metrics validation

4. **Set up pytest configuration**
   - Add test markers (unit, validation, integration)
   - Configure coverage reporting
   - Add parametrization for common test patterns

### 10.2 Future Enhancements

1. **Performance benchmarking suite**
   - Timing tests for different data sizes
   - Memory profiling
   - Scalability tests

2. **Integration tests**
   - Full pipeline integration
   - Multi-method workflows (ASR + EOG regression + ICA)
   - Real-world data scenarios

3. **Continuous validation**
   - Regular comparison against MNE updates
   - Regression test suite for numerical accuracy
   - Performance regression tracking

---

## Appendix A: Quick Reference - Test Patterns

### A.1 Basic Test Structure

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import mne
from mne.preprocessing import EOGRegression
from unittest.mock import patch, MagicMock

class TestEOGRegressionBasic:
    """Basic EOG regression functionality tests."""

    def test_fit_and_apply(self):
        """Test basic fit and apply workflow."""
        raw = create_synthetic_raw_with_blinks()

        # Fit
        model = EOGRegression(picks='eeg', picks_artifact='eog')
        model.fit(raw)

        # Check model state
        assert hasattr(model, 'coef_')
        assert model.coef_.shape[0] > 0

        # Apply
        cleaned = model.apply(raw, copy=True)

        # Verify signal reduction
        orig_ptp = np.ptp(raw.get_data('eeg'))
        clean_ptp = np.ptp(cleaned.get_data('eeg'))
        assert clean_ptp < orig_ptp

    def test_missing_eog_channels(self):
        """Test error on missing EOG channels."""
        raw = create_raw_without_eog()

        model = EOGRegression(picks='eeg', picks_artifact='eog')

        with pytest.raises(ValueError, match="EOG"):
            model.fit(raw)
```

### A.2 Validation Pattern

```python
def test_numerical_accuracy(self):
    """Validate against MNE reference implementation."""
    raw = load_mne_sample_data()

    # Our implementation
    our_model = OurEOGRegression().fit(raw)
    our_result = our_model.apply(raw.copy())

    # MNE reference
    mne_model = EOGRegression().fit(raw)
    mne_result = mne_model.apply(raw.copy())

    # Compare
    np.testing.assert_array_almost_equal(
        our_result.get_data('eeg'),
        mne_result.get_data('eeg'),
        decimal=10
    )
```

### A.3 Parametrized Test Pattern

```python
@pytest.mark.parametrize("n_channels,n_blinks", [
    (8, 5),
    (16, 10),
    (32, 20),
])
def test_scalability(self, n_channels, n_blinks):
    """Test across different data configurations."""
    raw = create_synthetic_eeg_with_blinks(
        n_eeg_channels=n_channels,
        n_blinks=n_blinks
    )

    model = EOGRegression().fit(raw)
    cleaned = model.apply(raw)

    assert cleaned is not None
```

---

**End of Report**
