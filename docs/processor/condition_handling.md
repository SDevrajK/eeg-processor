# Experimental condition processing and epoching

*2 stages in this category*

## Overview

Condition handling stages work with experimental conditions and events. They segment data around events and organize it for analysis.

## Typical Pipeline Order

1. [`segment_condition`](condition_handling/segment_condition.md)
2. [`epoch`](condition_handling/epoch.md)

## Available Stages

### [`epoch`](epoch.md)

Create epochs from Raw data - always returns new Epochs object

**Key Parameters:**
- `raw` (*BaseRaw*)
- `condition` (*dict*)
- `tmin` (*float*)

---

### [`segment_condition`](segment_condition.md)

Segment Raw data based on condition markers with optional in-place operation

**Key Parameters:**
- `raw` (*BaseRaw*)
- `condition` (*dict*)
- `padding` (*float*)

---

[← Back to Overview](README.md)