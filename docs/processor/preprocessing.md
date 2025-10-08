# Signal filtering and artifact removal

*8 stages in this category*

## Overview

Preprocessing stages clean and prepare the raw EEG data for analysis. This includes filtering, artifact removal, and channel management.

## Typical Pipeline Order

1. [`filter`](preprocessing/filter.md)
2. [`detect_bad_channels`](preprocessing/detect_bad_channels.md)
3. [`rereference`](preprocessing/rereference.md)
4. [`remove_artifacts`](preprocessing/remove_artifacts.md)
5. [`clean_rawdata_asr`](preprocessing/clean_rawdata_asr.md)

## Available Stages

### [`clean_rawdata_asr`](clean_rawdata_asr.md)

Clean raw EEG data using Artifact Subspace Reconstruction (ASR). ASR is an intermediate data cleaning step designed to be applied after bad channel detection/interpolation but before ICA. It identifies and corrects brief high-amplitude artifacts by reconstructing corrupted signal subspaces using a calibration-based approach. **Recommended Pipeline Position:** 1. Bad channel detection and interpolation 2. ASR data cleaning (this function) ← Intermediate step 3. ICA artifact removal (for remaining component-based artifacts)

**Key Parameters:**
- `raw` (*BaseRaw*)
- `cutoff` (*int | float*)
- `method` (*str*)

---

### [`compute_eog`](compute_eog.md)

Compute HEOG/VEOG from electrode pairs.

**Key Parameters:**
- `raw` (*BaseRaw*)
- `heog_pair` (*tuple | None*)
- `veog_pair` (*tuple | None*)

---

### [`detect_bad_channels`](detect_bad_channels.md)

Detect and optionally interpolate bad channels using MNE's LOF method. Uses Local Outlier Factor to identify channels that are outliers compared to their spatial neighbors, avoiding false positives from physiological artifacts like eyeblinks.

**Key Parameters:**
- `raw` (*BaseRaw*)
- `n_neighbors` (*int*)
- `threshold` (*float*)

---

### [`filter`](filter.md)

Apply filtering with optional in-place operation

**Key Parameters:**
- `raw` (*BaseRaw*)
- `l_freq` (*float | None*)
- `h_freq` (*float | None*)

---

### [`remove_artifacts`](remove_artifacts.md)

Remove artifacts using Independent Component Analysis (ICA). Combines automatic classification (ICALabel) with correlation-based detection to identify and remove eye blinks, muscle artifacts, cardiac artifacts, and line noise components.

**Key Parameters:**
- `raw` (*BaseRaw*)
- `n_components` (*float | int*)
- `method` (*str*)

---

### [`remove_blinks_emcp`](remove_blinks_emcp.md)

Remove blink artifacts using MNE's EOGRegression method. This method uses MNE's EOGRegression class to identify and remove blink artifacts from EEG data. **IMPORTANT: For EEG data, apply the desired reference (typically average reference) before using this method, as recommended by MNE.**

**Key Parameters:**
- `raw` (*BaseRaw*)
- `eog_channels` (*list*)
- `show_plot` (*bool*)

---

### [`rereference`](rereference.md)

Apply rereferencing to raw EEG data with robust exclude handling. Preserves existing projections (e.g., for blink correction) while adding new reference.

**Key Parameters:**
- `raw` (*BaseRaw*)
- `method` (*str | list*)
- `exclude` (*list | None*)

---

[← Back to Overview](README.md)