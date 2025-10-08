# Data preparation and event management

*3 stages in this category*

## Overview

Data handling stages manage the loading, saving, and initial preparation of EEG data. These stages typically run at the beginning or end of the pipeline.

## Typical Pipeline Order

1. [`crop`](data_handling/crop.md)
2. [`adjust_events`](data_handling/adjust_events.md)
3. [`correct_triggers`](data_handling/correct_triggers.md)

## Available Stages

### [`adjust_events`](adjust_events.md)

Adjust event times with optional in-place operation

**Key Parameters:**
- `raw` (*BaseRaw*)
- `shift_ms` (*float*)
- `target_events` (*list | None*)

---

### [`correct_triggers`](correct_triggers.md)

Correct incorrectly coded triggers in EEG data.

**Key Parameters:**
- `raw` (*BaseRaw*)
- `condition` (*dict*)
- `method` (*str*)

---

### [`crop`](crop.md)

Crop raw EEG data using either absolute times or event markers. Parameters ---------- crop_before : str | int | None Event code (e.g., 51, 'Stimulus/S 51', 'Stimulus/S  1') crop_after : str | int | None Event code to crop after last occurrence

**Key Parameters:**
- `raw` (*BaseRaw*)
- `t_min` (*float | None*)
- `t_max` (*float | None*)

---

[← Back to Overview](README.md)