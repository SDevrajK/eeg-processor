# Product Requirements Document: Multi-Event Time-Frequency Analysis with Memory Optimization

## Introduction

This feature enables the EEG Processor to handle multiple event types within a single experimental task, providing separate time-frequency analysis for each stimulus type while implementing proper memory estimation and BIDS-compliant file organization. The feature addresses critical limitations in processing auditory gap detection experiments where multiple stimulus conditions (different gap durations) need to be analyzed separately.

## Goals

1. **Enable event-specific processing**: Process each event type (Standard, 2ms gap, 5ms gap, 7ms gap, 10ms gap, 20ms gap, 30ms gap, 40ms gap) separately for time-frequency analysis
2. **Fix memory estimation**: Correct the current 330GB memory estimation error to provide accurate memory usage predictions
3. **Ensure BIDS compliance**: Implement proper BIDS naming conventions using the `desc` entity for event-specific derivatives
4. **Maintain scientific validity**: Preserve all existing functionality while adding multi-event capability
5. **Prevent memory crashes**: Implement reliable memory estimation and warning systems

## User Stories

**As an EEG researcher**, I want to process multiple stimulus events separately so that I can analyze neural responses to different gap durations independently.

**As a computational neuroscientist**, I want accurate memory estimation so that I can process large datasets without system crashes or unexpected memory exhaustion.

**As a data analyst**, I want BIDS-compliant file organization so that my processed data integrates seamlessly with other neuroimaging tools and meets publication standards.

## Functional Requirements

1. **Event Detection and Separation**
   - The system MUST automatically detect all event types defined in the configuration file's `triggers` section
   - The system MUST create separate epoch subsets for each event type using MNE's built-in epoch selection
   - The system MUST handle missing event types gracefully with appropriate warnings

2. **Pipeline Architecture Enhancement**
   - The system MUST separate processing stages into pre-epoching, epoching, and post-epoching phases
   - The system MUST maintain the existing single `apply_processing_stage()` call architecture
   - The system MUST process each event type independently for post-epoching stages only

3. **Memory Estimation Correction**
   - The system MUST provide accurate memory estimation for averaged time-frequency analysis
   - The system MUST account for MNE's sequential processing with `average=True` (not simultaneous epoch storage)
   - The system MUST warn users when available memory is insufficient for processing

4. **BIDS-Compliant File Organization**
   - The system MUST use the `desc` entity to distinguish between different event types
   - The system MUST follow the naming pattern: `sub-{id}_task-{task}_desc-{event}_tfr.h5`
   - The system MUST NOT use non-standard entities like `epochtype` or inappropriate `run` entities

5. **Result Management**
   - The system MUST save one time-frequency file per event type per participant
   - The system MUST preserve all existing metadata and processing information
   - The system MUST handle ITC (Inter-Trial Coherence) data appropriately for each event type

6. **Backward Compatibility**
   - The system MUST continue to work with existing single-condition configurations
   - The system MUST maintain all existing quality control and logging functionality
   - The system MUST preserve all existing error handling and recovery mechanisms

## Non-Goals

- **Modifying core time-frequency algorithms**: The underlying MNE time-frequency computation methods will not be changed
- **Changing configuration file structure**: Existing configuration files should continue to work without modification
- **Adding new time-frequency methods**: Only the processing architecture will be enhanced, not the analysis methods themselves
- **Implementing custom BIDS entities**: Will use standard BIDS entities only, no custom extensions

## Data Specifications

### Input Data
- **File formats**: EEG data in BrainVision (.dat), EDF, FIF, or other MNE-supported formats
- **Epochs structure**: MNE Epochs objects containing multiple event types with trigger codes
- **Configuration**: YAML files defining trigger mappings (e.g., Standard: 1, 2ms: 2, 5ms: 3, etc.)

### Output Data
- **File format**: HDF5 (.h5) files containing AverageTFR objects
- **Naming convention**: BIDS-compliant using `desc` entity for event types
- **Metadata**: JSON sidecar files with processing parameters and event information

### Memory Requirements
- **Typical dataset**: ~8500 epochs, 67 channels, 800ms duration, 1000Hz sampling rate
- **Corrected estimation**: ~40MB per averaged TFR (not 330GB as currently calculated)
- **Processing memory**: Sequential processing minimizes peak memory usage

## Technical Considerations

### Architecture Changes
- **Pipeline separation**: Clean division between pre-epoching (Raw→Raw), epoching (Raw→Epochs), and post-epoching (Epochs→derivatives) stages
- **Event iteration**: Loop through event types after epoching stage completion
- **Memory management**: Leverage existing memory tracking and cleanup mechanisms

### Dependencies
- **MNE-Python**: Built-in epoch selection by event name (`epochs[event_name]`)
- **Existing codebase**: Minimal changes to core processing functions
- **BIDS validation**: Ensure compatibility with BIDS validator tools

### Integration Points
- **pipeline.py**: Main orchestration logic for event-specific processing
- **time_frequency.py**: Memory estimation corrections (no algorithm changes)
- **result_saver.py**: BIDS-compliant naming implementation
- **data_processor.py**: Maintain existing stage registry and processing methods

### Scientific Validity
- **Event separation**: Each stimulus condition analyzed independently to preserve experimental design
- **Statistical power**: Separate analysis maintains appropriate sample sizes for each condition
- **Temporal precision**: Event-specific analysis preserves timing-related neural responses

## Output Specifications

### File Organization
```
results/
└── processed/
    ├── sub-S001_task-broadband_desc-standard_tfr.h5
    ├── sub-S001_task-broadband_desc-2ms_tfr.h5
    ├── sub-S001_task-broadband_desc-5ms_tfr.h5
    ├── sub-S001_task-broadband_desc-7ms_tfr.h5
    ├── sub-S001_task-broadband_desc-10ms_tfr.h5
    ├── sub-S001_task-broadband_desc-20ms_tfr.h5
    ├── sub-S001_task-broadband_desc-30ms_tfr.h5
    └── sub-S001_task-broadband_desc-40ms_tfr.h5
```

### Quality Control
- **Memory tracking**: Detailed logging of memory usage per event type
- **Processing logs**: Event-specific processing metrics and timing
- **Error handling**: Graceful handling of missing events or processing failures

## Success Metrics

1. **Functional success**: 8 separate TFR files generated per participant (one per event type)
2. **Memory accuracy**: Memory estimation within 10% of actual usage (vs. current 8000x overestimation)
3. **BIDS compliance**: All output files pass BIDS validator checks
4. **Performance**: No increase in total processing time compared to single-event processing
5. **Backward compatibility**: Existing single-condition configurations continue to work unchanged
6. **Scientific validity**: Event-specific TFR results match manual event selection and processing

## Open Questions

1. **Event ordering**: Should event types be processed in a specific order for consistency?
2. **Missing events**: How should the system handle participants with missing event types (beyond warnings)?
3. **Memory thresholds**: What specific memory percentage should trigger warnings (currently 70% safety margin)?
4. **ITC handling**: Should Inter-Trial Coherence be saved separately for each event type or combined?
5. **Configuration validation**: Should the system validate that all configured event types are present in the data?
6. **Batch optimization**: Would processing multiple participants in parallel benefit from event-specific batching strategies?

---

**Target Audience**: Junior developers familiar with Python and basic EEG processing concepts
**Implementation Priority**: High - addresses critical memory and scientific validity issues
**Estimated Complexity**: Medium - requires careful architecture changes but leverages existing functionality