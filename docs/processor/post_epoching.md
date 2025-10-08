# Analysis of epoched data

*3 stages in this category*

## Overview

Post-epoching stages operate on segmented data to extract meaningful information and perform advanced analyses.

## Available Stages

### [`time_frequency`](time_frequency.md)

Compute averaged time-frequency representation from epochs

**Key Parameters:**
- `epochs` (*Epochs*)
- `freq_range` (*list*)
- `n_freqs` (*int*)

---

### [`time_frequency_average`](time_frequency_average.md)

Convert RawTFR to AverageTFR by averaging across time dimension

**Key Parameters:**
- `raw_tfr` (*Any*)
- `method` (*str*)
- `kwargs` (*Any*)

---

### [`time_frequency_raw`](time_frequency_raw.md)

Compute baseline power spectrum from continuous raw data

**Key Parameters:**
- `raw` (*BaseRaw*)
- `freq_range` (*list*)
- `n_freqs` (*int*)

---

[← Back to Overview](README.md)