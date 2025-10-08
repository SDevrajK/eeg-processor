# Task List: Multi-Event Time-Frequency Analysis with Memory Optimization

## Project Metadata

- **Project Name**: EEG Processor Multi-Event Time-Frequency Analysis
- **PRD Source**: [docs/tasks/prd-multi-event-time-frequency-analysis.md](./prd-multi-event-time-frequency-analysis.md)
- **Generated Date**: 2025-01-13
- **Estimated Duration**: 8-12 days
- **Total Tasks**: 16 tasks across 5 phases

## Files the Implementation Agent Must Read

### Key Reference Files (Must Read First)

| File Path | Purpose | Key Focus Areas |
|-----------|---------|-----------------|
| `src/eeg_processor/pipeline.py` | Main orchestration logic | Current processing flow (lines 342-351) |
| `src/eeg_processor/state_management/data_processor.py` | Stage registry and processing | `apply_processing_stage` method |
| `src/eeg_processor/state_management/result_saver.py` | Current saving logic | BIDS naming conventions (line 125+) |
| `src/eeg_processor/processing/time_frequency.py` | Time-frequency analysis | Memory estimation bug (lines 50-76) |
| `src/eeg_processor/utils/memory_tools.py` | Memory utilities | Pressure monitoring functions |
| `/mnt/c/Users/sayee/Documents/Research/DudaLab/DevelopmentalGap/Analysis/Python/configs/Adults_EOA_processing_params.yml` | Example configuration | Trigger structure (lines 187-200) |

### Files to Modify

| File Path | Modification Type | Description |
|-----------|-------------------|-------------|
| `src/eeg_processor/pipeline.py` | **Modify** | Add event-specific processing logic |
| `src/eeg_processor/processing/time_frequency.py` | **Modify** | Fix memory estimation formula |
| `src/eeg_processor/state_management/result_saver.py` | **Modify** | Add BIDS-compliant event naming |

### Files to Create

| File Path | Type | Description |
|-----------|------|-------------|
| `tests/test_multi_event_tfr.py` | Test | Integration tests for multi-event processing |
| `tests/test_bids_naming.py` | Test | BIDS-compliant naming validation |
| `tests/fixtures/test_multi_event_config.yml` | Config | Test configuration with multiple triggers |

---

## Phase 1: Memory Estimation Fix (1-2 days)

**Goal**: Correct the critical memory estimation error from 330GB to accurate ~40MB calculation

### Task 1: Fix memory estimation formula in time_frequency.py
- **Priority**: High
- **Estimated Hours**: 3-4
- **Dependencies**: None

**Description**: Correct the memory estimation in `compute_epochs_tfr_average` function (line 54) to account for MNE's sequential processing with `average=True` instead of assuming all epochs stored simultaneously.

**Acceptance Criteria**:
- [x] Memory estimation formula changed from `n_epochs × n_channels × n_freqs × n_times` to `n_channels × n_freqs × n_times`
- [x] Estimation for test dataset (8500 epochs, 67 channels, 800ms, 49 freqs) shows ~40MB instead of 330GB
- [x] Memory estimation includes explanation comment about MNE's sequential processing
- [x] Logging output shows corrected memory estimates with proper formatting

### Task 2: Update memory warning thresholds
- **Priority**: Medium
- **Estimated Hours**: 2-3
- **Dependencies**: Task 1

**Description**: Adjust memory warning logic to use appropriate thresholds for the corrected smaller memory estimates.

**Acceptance Criteria**:
- [x] Memory warnings trigger at reasonable thresholds (not 70% of available memory for 40MB)
- [x] Warning messages are clear and actionable for users
- [x] Critical memory pressure detection still works appropriately

### Task 3: Create memory estimation unit tests
- **Priority**: Low
- **Estimated Hours**: 1-2
- **Dependencies**: Task 1

**Description**: Add comprehensive tests for memory estimation accuracy across different dataset sizes.

**Acceptance Criteria**:
- [x] Tests validate estimation accuracy within 10% of expected values
- [x] Tests cover various combinations of epochs, channels, frequencies, and time points
- [x] Tests verify that ITC computation doubles memory estimate when enabled

---

## Phase 2: Pipeline Architecture Enhancement (2-3 days)

**Goal**: Implement event-specific processing workflow while maintaining existing pipeline architecture

### Task 4: Add stage separation logic to pipeline.py
- **Priority**: High
- **Estimated Hours**: 4-6
- **Dependencies**: None

**Description**: Implement helper methods to split processing stages into pre-epoching, epoching, and post-epoching phases.

**Acceptance Criteria**:
- [x] `_split_stages_by_epoching` method correctly identifies epoch stage as transition point
- [x] Pre-epoching stages include filter, detect_bad_channels, rereference, remove_artifacts
- [x] Post-epoching stages include time_frequency, time_frequency_raw, time_frequency_average
- [x] Method handles edge cases (no epoching stage, multiple epoching stages)

### Task 5: Implement event-specific processing loop
- **Priority**: High
- **Estimated Hours**: 4-5
- **Dependencies**: Task 4

**Description**: Modify `_process_condition` method to process each trigger type separately for post-epoching stages.

**Acceptance Criteria**:
- [x] Pipeline processes pre-epoching stages normally on full raw data
- [x] After epoching, pipeline loops through each trigger type from condition configuration
- [x] Each trigger type is processed independently through post-epoching stages
- [x] Single `apply_processing_stage` call architecture is maintained
- [x] Existing memory tracking and cleanup logic is preserved

### Task 6: Add event extraction helper method
- **Priority**: Medium
- **Estimated Hours**: 2-3
- **Dependencies**: Task 5

**Description**: Create `_extract_event_epochs` method to safely extract specific trigger types from full epochs object.

**Acceptance Criteria**:
- [x] Method uses MNE's built-in `epochs[trigger_name]` selection
- [x] Handles missing trigger types gracefully with appropriate warnings
- [x] Returns valid Epochs object for downstream processing
- [x] Preserves all epoch metadata and structure

---

## Phase 3: BIDS Compliance Implementation (1-2 days)

**Goal**: Implement BIDS-compliant file naming using desc entity for event types

### Task 7: Add event-specific saving methods to result_saver.py
- **Priority**: High
- **Estimated Hours**: 3-4
- **Dependencies**: None

**Description**: Create methods to save processed data with BIDS-compliant event-specific naming.

**Acceptance Criteria**:
- [x] Enhanced `save_data_object` method accepts `event_type` parameter
- [x] File naming follows pattern: `sub-{id}_task-{task}_desc-{event}_tfr.h5`
- [x] Event names are cleaned for BIDS compliance (lowercase, no spaces/special chars)
- [x] Method handles all supported data types (AverageTFR, Spectrum, RawTFR)

### Task 8: Update filename generation for BIDS compliance
- **Priority**: Medium
- **Estimated Hours**: 2-3
- **Dependencies**: Task 7

**Description**: Modify `_get_filename` method to support event-specific desc entity while maintaining backward compatibility.

**Acceptance Criteria**:
- [x] Method accepts optional `event_type` parameter
- [x] Generates BIDS-compliant names with desc entity when `event_type` provided
- [x] Maintains existing naming convention for non-event-specific files
- [x] Handles both BIDS and legacy naming conventions

### Task 9: Create BIDS naming validation tests
- **Priority**: Low
- **Estimated Hours**: 2
- **Dependencies**: Task 8

**Description**: Add tests to ensure generated filenames are BIDS-compliant and unique.

**Acceptance Criteria**:
- [ ] Tests validate filename format against BIDS specification
- [ ] Tests ensure event names are properly sanitized
- [ ] Tests verify no filename collisions occur between different event types

---

## Phase 4: Event Processing Logic (2-3 days)

**Goal**: Implement robust event processing logic with proper error handling and validation

### Task 10: Integrate event processing into pipeline workflow
- **Priority**: High
- **Estimated Hours**: 3-4
- **Dependencies**: Tasks 5, 6, 7

**Description**: Connect event-specific processing loop with existing pipeline error handling and quality tracking.

**Acceptance Criteria**:
- [x] Event processing integrates seamlessly with existing memory tracking
- [x] Quality tracking records metrics for each event type separately
- [x] Error handling isolates failures to specific event types
- [x] Processing continues to other event types if one fails

### Task 11: Add configuration validation for trigger types
- **Priority**: Medium
- **Estimated Hours**: 2-3
- **Dependencies**: Task 10

**Description**: Validate that configuration trigger definitions are compatible with data events.

**Acceptance Criteria**:
- [ ] Validation checks that configured triggers exist in epoch data
- [ ] Warnings generated for missing trigger types
- [ ] Processing continues gracefully when some triggers are missing
- [ ] Clear error messages when no triggers match configuration

### Task 12: Create comprehensive integration tests
- **Priority**: Medium
- **Estimated Hours**: 3
- **Dependencies**: Tasks 10, 11

**Description**: Test full pipeline with multi-event configuration to ensure end-to-end functionality.

**Acceptance Criteria**:
- [ ] Test processes complete multi-event configuration successfully
- [ ] Verifies correct number of output files generated (one per event type)
- [ ] Validates that each output file contains expected data structure
- [ ] Tests backward compatibility with single-event configurations

---

## Phase 5: Validation and Testing (2 days)

**Goal**: Comprehensive testing and validation to ensure scientific validity and backward compatibility

### Task 13: Create multi-event test configuration and fixtures
- **Priority**: High
- **Estimated Hours**: 3-4
- **Dependencies**: None

**Description**: Develop test configuration mimicking real multi-event experimental setup.

**Acceptance Criteria**:
- [ ] Test config includes 8 trigger types matching real experimental design
- [ ] Fixture data contains appropriate number of epochs per trigger type
- [ ] Configuration tests both broadband task and backward compatibility

### Task 14: Validate scientific accuracy of event separation
- **Priority**: High
- **Estimated Hours**: 4
- **Dependencies**: Tasks 12, 13

**Description**: Ensure that event-specific processing produces scientifically valid results comparable to manual event selection.

**Acceptance Criteria**:
- [ ] Event-specific TFR results match manual `epochs[event]` selection and processing
- [ ] Statistical properties (means, variances) are preserved correctly
- [ ] Time-frequency characteristics are consistent across processing methods
- [ ] ITC calculations are accurate for each event type

### Task 15: Performance and memory validation testing
- **Priority**: Medium
- **Estimated Hours**: 2-3
- **Dependencies**: Task 14

**Description**: Verify that multi-event processing doesn't significantly impact performance or memory usage.

**Acceptance Criteria**:
- [ ] Total processing time comparable to single-event processing × number of events
- [ ] Peak memory usage remains within expected bounds
- [ ] Memory estimation accuracy validated against actual usage
- [ ] No memory leaks detected during multi-event processing

### Task 16: Backward compatibility validation
- **Priority**: Medium
- **Estimated Hours**: 2
- **Dependencies**: Task 14

**Description**: Ensure existing single-event configurations continue to work without modification.

**Acceptance Criteria**:
- [ ] Existing configuration files process without errors
- [ ] Output file formats and naming remain consistent for single-event cases
- [ ] No breaking changes to existing API or data structures
- [ ] Regression tests pass for all existing functionality

---

## Task Completion Tracking

| Task ID | Status | Completed Date | Completed By | Notes |
|---------|--------|---------------|--------------|-------|
| 1 | ⏳ Pending | | | |
| 2 | ⏳ Pending | | | |
| 3 | ⏳ Pending | | | |
| 4 | ⏳ Pending | | | |
| 5 | ⏳ Pending | | | |
| 6 | ⏳ Pending | | | |
| 7 | ⏳ Pending | | | |
| 8 | ⏳ Pending | | | |
| 9 | ⏳ Pending | | | |
| 10 | ⏳ Pending | | | |
| 11 | ⏳ Pending | | | |
| 12 | ⏳ Pending | | | |
| 13 | ⏳ Pending | | | |
| 14 | ⏳ Pending | | | |
| 15 | ⏳ Pending | | | |
| 16 | ⏳ Pending | | | |

**Status Legend**: ⏳ Pending | 🔄 In Progress | ✅ Completed | ❌ Blocked

---

## Critical Implementation Notes

### Testing Setup
- Run tests using: `.venv/Scripts/python.exe -m pytest tests/ -v`
- Memory estimation tests should use realistic EEG dataset parameters
- Integration tests require test configuration with multiple trigger types

### Implementation Details
- **Memory Bug Location**: The current memory estimation formula is in `time_frequency.py` line 54 and incorrectly calculates `n_epochs * n_channels * n_freqs * n_times` for averaged TFR
- **Pipeline Constraint**: The `apply_processing_stage` call on `pipeline.py` line 351 must remain unchanged - all modifications should be around this call
- **BIDS Compliance**: Must use `desc` entity for event types, NOT `run` entity (run implies different recordings)
- **Test Configuration**: Should mirror `Adults_EOA_processing_params.yml` structure with Broadband task and 8 trigger types

### Scientific Considerations
- Event-specific processing must preserve statistical properties of each stimulus condition
- Memory optimization should not compromise scientific accuracy or reproducibility
- File naming must be descriptive enough for researchers to identify stimulus conditions
- Backward compatibility is crucial - existing analyses must continue to work unchanged

### Architecture Principles
- Maintain separation of concerns - pipeline orchestrates, modules implement
- Preserve existing error handling and quality tracking mechanisms
- Use MNE's built-in epoch selection rather than custom filtering logic
- Keep memory estimation conservative but accurate to prevent system crashes

---

## Expected Output Files

After successful implementation, each participant will generate 8 BIDS-compliant files:

```
results/processed/
├── sub-S001_task-broadband_desc-standard_tfr.h5
├── sub-S001_task-broadband_desc-2ms_tfr.h5
├── sub-S001_task-broadband_desc-5ms_tfr.h5
├── sub-S001_task-broadband_desc-7ms_tfr.h5
├── sub-S001_task-broadband_desc-10ms_tfr.h5
├── sub-S001_task-broadband_desc-20ms_tfr.h5
├── sub-S001_task-broadband_desc-30ms_tfr.h5
└── sub-S001_task-broadband_desc-40ms_tfr.h5
```