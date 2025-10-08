# EEG Processor - Troubleshooting Guide

Common issues and solutions when using processing stages.

## General Issues

### Configuration file not loading

1. Check YAML syntax using `eeg-processor validate config.yml`
2. Ensure all required fields are present
3. Verify file paths exist and are accessible

### Stage not found errors

1. Use `eeg-processor list-stages` to see available stages
2. Check spelling of stage names in your configuration
3. Ensure you're using the correct stage names, not function names

### Memory errors during processing

1. Reduce the number of parallel jobs
2. Process participants individually using `--participant`
3. Enable intermediate file saving to avoid recomputation

### Slow processing

1. Use parallel processing with `-j` flag
2. Consider using ASR for artifact removal instead of ICA
3. Filter data early in the pipeline to reduce computational load

## Stage-Specific Issues

### filter

**Filtering removes too much data**

Check your frequency ranges. High-pass filters above 1 Hz may remove important slow components. Low-pass filters below 30 Hz may remove important frequency content for some analyses.

**Edge artifacts**

Filtering can introduce artifacts at the beginning and end of recordings. Consider cropping data or using longer recordings.

### detect_bad_channels

**Too many channels detected as bad**

Lower the threshold parameter or check if your data has systemic issues. More than 20% bad channels often indicates recording problems.

**Good channels marked as bad**

Increase the threshold parameter or check the n_neighbors setting. Dense electrode arrays may need higher n_neighbors values.

### epoch

**Not enough epochs after rejection**

Check your event markers and epoch timing. Consider relaxing rejection thresholds or improving preprocessing steps.

**Baseline period issues**

Ensure baseline period is within the epoch window and doesn't overlap with your events of interest.

### clean_rawdata_asr

**Over-correction of data**

Increase the cutoff parameter. Values too low (< 10) may remove valid data. Start with cutoff=20 and adjust based on your data quality.

**Insufficient artifact removal**

Decrease the cutoff parameter, but be careful not to go below 10. Also ensure you have sufficient calibration data.

## Data Quality Issues

### Too many bad channels detected

- Check recording setup and electrode impedances
- Verify reference electrode is properly connected
- Consider if high bad channel count is due to participant factors
- Adjust bad channel detection threshold if appropriate

### Excessive artifact rejection

- Review rejection thresholds - they may be too strict
- Check if artifacts are due to systematic issues (line noise, movement)
- Consider using ASR before epoching to clean continuous data
- Verify event timing is correct

### Poor signal quality after preprocessing

- Check filter settings - avoid over-filtering
- Verify reference choice is appropriate for your data
- Consider artifact removal order (ASR → EMCP → ICA)
- Review original data quality

## Getting Help

If you're still experiencing issues:

1. **Use CLI help commands:**
   ```bash
   eeg-processor help-stage <stage_name>
   eeg-processor validate <config_file>
   ```

2. **Check configuration validation:**
   ```bash
   eeg-processor validate config.yml --detailed
   ```

3. **Run with verbose output:**
   ```bash
   eeg-processor process config.yml --verbose
   ```

4. **Test with minimal configuration:**
   ```bash
   eeg-processor create-config --minimal
   ```

[🏠 Back to Main Index](README.md)