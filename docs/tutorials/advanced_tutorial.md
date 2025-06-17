# Advanced EEG Processor Tutorial

## Introduction

This tutorial covers advanced features and customization options for power users who need more control over their EEG processing workflows.

## Advanced Configuration

### Hierarchical Configuration

Organize complex experiments with hierarchical configurations:

```yaml
# base_config.yml - Common settings
data_format: "brainvision"
paths:
  raw_data_dir: "data/raw"
  results_dir: "data/processed"

filtering:
  lowpass: 40
  highpass: 0.1
  notch: 50

stages:
  - load_data
  - montage
  - filter
  - bad_channels
  - epoching
  - artifact_rejection
  - ica
  - evoked
  - save_results

quality_control:
  enabled: true
  generate_plots: true
```

```yaml
# experiment1_config.yml - Specific experiment
# Inherits from base_config.yml
dataset_name: "Experiment1_P300"

participants:
  sub-01: "exp1_sub-01_task-oddball_eeg.vhdr"
  sub-02: "exp1_sub-02_task-oddball_eeg.vhdr"

epoching:
  tmin: -0.2
  tmax: 0.8
  baseline: [-0.2, 0]

conditions:
  - name: "target"
    condition_markers: ["S1", "S11"]
    description: "Target stimuli in oddball paradigm"
  - name: "standard"
    condition_markers: ["S2", "S12"]
    description: "Standard stimuli in oddball paradigm"
```

### Parameter Overrides

Override specific parameters programmatically:

```python
from eeg_processor import EEGPipeline, load_config

# Load base configuration
base_config = load_config("base_config.yml")

# Define parameter overrides for different groups
young_adults_params = {
    "filtering": {"lowpass": 45},  # Higher frequency range
    "artifact_rejection": {"peak_to_peak": 120e-6}  # More lenient
}

older_adults_params = {
    "filtering": {"lowpass": 35},  # Lower frequency range
    "artifact_rejection": {"peak_to_peak": 80e-6}   # More strict
}

# Process different groups with different parameters
for group_name, params in [("young", young_adults_params), ("older", older_adults_params)]:
    pipeline = EEGPipeline(base_config, params)
    pipeline.config.dataset_name = f"Study_{group_name}"
    results = pipeline.run_all()
    print(f"Processed {group_name} adults: {len(results)} participants")
```

## Custom Processing Stages

### Creating Custom Stages

Define your own processing functions:

```python
from eeg_processor import EEGPipeline
from eeg_processor.processing.base import ProcessingStage
import mne
import numpy as np

class CustomSpectralAnalysis(ProcessingStage):
    """Custom stage for spectral analysis."""
    
    def __init__(self, freqs=None, method='welch'):
        self.freqs = freqs or np.arange(1, 40, 1)
        self.method = method
    
    def apply(self, data, **kwargs):
        """Apply spectral analysis to epochs."""
        if not isinstance(data, mne.Epochs):
            raise ValueError("Spectral analysis requires epochs")
        
        # Compute power spectral density
        psds, freqs = mne.time_frequency.psd_welch(
            data, fmin=self.freqs[0], fmax=self.freqs[-1],
            n_fft=2048, n_overlap=1024
        )
        
        # Store results in epochs metadata
        data.metadata['psd_alpha'] = np.mean(psds[:, :, (freqs >= 8) & (freqs <= 12)], axis=2)
        data.metadata['psd_beta'] = np.mean(psds[:, :, (freqs >= 13) & (freqs <= 30)], axis=2)
        
        return data

# Register custom stage
pipeline = EEGPipeline("config.yml")
pipeline.processor.register_stage("spectral_analysis", CustomSpectralAnalysis())

# Use in configuration
config = {
    "stages": [
        "load_data",
        "filter", 
        "epoching",
        "spectral_analysis",  # Your custom stage
        "save_results"
    ],
    "spectral_analysis": {
        "freqs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "method": "welch"
    }
}
```

### Pipeline Hooks

Add custom functions at specific pipeline stages:

```python
def custom_preprocessing_hook(raw, config):
    """Custom preprocessing before main pipeline."""
    # Apply custom filters
    if 'custom_notch' in config:
        raw.notch_filter(config['custom_notch'])
    
    # Custom bad channel detection
    if config.get('advanced_bad_channel_detection', False):
        from mne.preprocessing import find_bad_channels_lof
        bads = find_bad_channels_lof(raw)
        raw.info['bads'].extend(bads)
    
    return raw

def custom_quality_hook(epochs, tracker):
    """Custom quality metrics."""
    # Calculate custom quality metrics
    variance_ratio = np.var(epochs.get_data(), axis=2).mean()
    tracker.add_metric("variance_ratio", variance_ratio, "custom")
    
    # Detect unusual patterns
    if variance_ratio > 1000:
        tracker.add_flag("high_variance", "Unusually high variance detected")

# Register hooks
pipeline = EEGPipeline("config.yml")
pipeline.add_preprocessing_hook(custom_preprocessing_hook)
pipeline.add_quality_hook(custom_quality_hook)
```

## Advanced Quality Control

### Custom Quality Metrics

Define domain-specific quality metrics:

```python
from eeg_processor.quality_control import QualityMetricsAnalyzer

class CustomQualityAnalyzer(QualityMetricsAnalyzer):
    """Extended quality analyzer with custom metrics."""
    
    def analyze_erp_quality(self, evoked, condition_name):
        """Analyze ERP-specific quality metrics."""
        quality_metrics = {}
        
        # Signal-to-noise ratio for ERP
        baseline_data = evoked.copy().crop(tmin=-0.2, tmax=0)
        signal_data = evoked.copy().crop(tmin=0.1, tmax=0.5)
        
        baseline_std = np.std(baseline_data.data, axis=1).mean()
        signal_peak = np.max(np.abs(signal_data.data), axis=1).mean()
        
        snr = signal_peak / baseline_std
        quality_metrics['erp_snr'] = snr
        
        # Peak latency consistency
        peak_latencies = []
        for ch_data in signal_data.data:
            peak_idx = np.argmax(np.abs(ch_data))
            peak_latency = signal_data.times[peak_idx]
            peak_latencies.append(peak_latency)
        
        latency_consistency = 1 - (np.std(peak_latencies) / np.mean(peak_latencies))
        quality_metrics['peak_latency_consistency'] = latency_consistency
        
        # Topographic consistency
        correlation_matrix = np.corrcoef(evoked.data)
        topo_consistency = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        quality_metrics['topographic_consistency'] = topo_consistency
        
        return quality_metrics
    
    def assess_data_completeness(self, epochs, expected_trials):
        """Assess completeness of data collection."""
        actual_trials = len(epochs)
        completeness = actual_trials / expected_trials
        
        quality_score = "excellent" if completeness >= 0.9 else \
                       "good" if completeness >= 0.8 else \
                       "acceptable" if completeness >= 0.7 else "poor"
        
        return {
            'completeness_ratio': completeness,
            'actual_trials': actual_trials,
            'expected_trials': expected_trials,
            'completeness_score': quality_score
        }

# Use custom analyzer
pipeline = EEGPipeline("config.yml")
pipeline.quality_tracker.analyzer = CustomQualityAnalyzer()
```

### Automated Quality Thresholds

Set up automated quality control with thresholds:

```yaml
quality_control:
  enabled: true
  automatic_rejection: true
  
  thresholds:
    # Channel quality
    bad_channels_max: 0.15        # Max 15% bad channels
    channel_correlation_min: 0.7  # Min correlation with neighbors
    
    # Artifact rejection
    artifact_rejection_max: 0.3   # Max 30% rejected epochs
    peak_to_peak_threshold: 100e-6
    
    # Signal quality
    snr_min: 3.0                  # Minimum signal-to-noise ratio
    variance_ratio_max: 10.0      # Maximum variance ratio
    
    # Processing time limits
    max_processing_time: 3600     # 1 hour max per participant
    memory_limit: 0.8            # 80% of available memory
    
  actions:
    on_threshold_exceeded:
      - "log_warning"
      - "generate_report"
      # - "skip_participant"  # Uncomment to auto-skip
      # - "email_notification"
    
    on_critical_failure:
      - "stop_processing"
      - "save_intermediate"
      - "generate_error_report"

  notifications:
    email:
      enabled: false
      smtp_server: "smtp.gmail.com"
      recipients: ["researcher@university.edu"]
```

## Batch Processing and Parallelization

### Multi-Dataset Processing

Process multiple datasets efficiently:

```python
from eeg_processor import EEGPipeline
from pathlib import Path
import concurrent.futures
from typing import List, Dict

class BatchProcessor:
    """Efficient batch processing of multiple datasets."""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        
    def process_dataset(self, dataset_config: Dict) -> Dict:
        """Process a single dataset."""
        try:
            # Merge base config with dataset-specific config
            pipeline = EEGPipeline(self.base_config_path, dataset_config)
            
            # Set dataset-specific output directory
            if 'dataset_name' in dataset_config:
                output_dir = Path(pipeline.config.results_dir) / dataset_config['dataset_name']
                pipeline.config.results_dir = output_dir
            
            # Process all participants
            results = pipeline.run_all()
            
            return {
                'dataset': dataset_config.get('dataset_name', 'unnamed'),
                'status': 'success',
                'participants': len(results),
                'results_dir': str(pipeline.config.results_dir)
            }
            
        except Exception as e:
            return {
                'dataset': dataset_config.get('dataset_name', 'unnamed'),
                'status': 'error',
                'error': str(e)
            }
    
    def process_multiple_datasets(self, datasets: List[Dict], max_workers: int = 2) -> List[Dict]:
        """Process multiple datasets in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all datasets for processing
            future_to_dataset = {
                executor.submit(self.process_dataset, dataset): dataset
                for dataset in datasets
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✓ Completed: {result['dataset']}")
                except Exception as e:
                    print(f"✗ Failed: {dataset.get('dataset_name', 'unnamed')} - {e}")
        
        return results

# Usage example
datasets = [
    {
        'dataset_name': 'Young_Adults',
        'paths': {'raw_data_dir': 'data/young_adults'},
        'filtering': {'lowpass': 45}
    },
    {
        'dataset_name': 'Older_Adults', 
        'paths': {'raw_data_dir': 'data/older_adults'},
        'filtering': {'lowpass': 35}
    },
    {
        'dataset_name': 'Clinical_Group',
        'paths': {'raw_data_dir': 'data/clinical'},
        'artifact_rejection': {'peak_to_peak': 80e-6}
    }
]

processor = BatchProcessor("base_config.yml")
results = processor.process_multiple_datasets(datasets, max_workers=3)

# Summary report
for result in results:
    if result['status'] == 'success':
        print(f"✓ {result['dataset']}: {result['participants']} participants -> {result['results_dir']}")
    else:
        print(f"✗ {result['dataset']}: {result['error']}")
```

### Cluster Computing Integration

Integration with SLURM or other job schedulers:

```python
# slurm_processor.py
import os
import sys
import argparse
from eeg_processor import EEGPipeline

def process_participant_on_cluster(config_path: str, participant_id: str, output_dir: str):
    """Process single participant on cluster node."""
    
    # Load configuration with cluster-specific overrides
    overrides = {
        'paths': {'results_dir': output_dir},
        'parallel': {'n_jobs': int(os.environ.get('SLURM_CPUS_PER_TASK', 1))}
    }
    
    pipeline = EEGPipeline(config_path, overrides)
    
    # Process single participant
    result = pipeline.run_participant(participant_id)
    
    print(f"Completed processing {participant_id} on node {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("participant_id") 
    parser.add_argument("output_dir")
    
    args = parser.parse_args()
    process_participant_on_cluster(args.config_path, args.participant_id, args.output_dir)
```

```bash
#!/bin/bash
# submit_batch.sh - SLURM submission script

#SBATCH --job-name=eeg_processing
#SBATCH --array=1-50%10          # Process 50 participants, max 10 concurrent
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eeg_%A_%a.out

# Get participant ID from array index
PARTICIPANT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participant_list.txt)

# Run processing
python slurm_processor.py config.yml $PARTICIPANT_ID results/
```

## Advanced Analysis Features

### Time-Frequency Analysis

Comprehensive time-frequency analysis:

```yaml
time_frequency:
  enabled: true
  methods:
    - method: "morlet"
      freqs: [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]
      n_cycles: 3
      baseline: [-0.5, -0.1]
      baseline_mode: "percent"
      
    - method: "stockwell"
      freqs: [1, 2, 3, 4, 5, 6, 7, 8]
      width: 1.0
      
    - method: "hilbert"
      freqs: [[8, 12], [13, 30]]  # Alpha and beta bands
      filter_length: "auto"

  connectivity:
    enabled: true
    methods: ["plv", "pli", "wpli"]
    frequency_bands:
      theta: [4, 8]
      alpha: [8, 12]
      beta: [13, 30]
```

### Source Localization Setup

Prepare data for source localization:

```yaml
source_localization:
  enabled: true
  
  head_model:
    subjects_dir: "/path/to/freesurfer/subjects"
    subject: "fsaverage"  # Or participant-specific
    surfaces: ["white", "pial"]
    
  forward_model:
    spacing: "ico4"  # Source space resolution
    mindist: 5.0     # Minimum distance from skull
    
  inverse_method:
    method: "dSPM"
    lambda2: 1.0 / 9.0
    pick_ori: "normal"
    
  source_epochs:
    apply_baseline: true
    baseline: [-0.2, 0]
```

### Group-Level Analysis

Automated group-level statistics:

```python
from eeg_processor.analysis import GroupAnalyzer

class AdvancedGroupAnalyzer(GroupAnalyzer):
    """Advanced group-level analysis."""
    
    def compute_group_statistics(self, results_dir: str, conditions: List[str]):
        """Compute comprehensive group statistics."""
        
        # Load all participants' data
        all_evoked = {}
        for condition in conditions:
            all_evoked[condition] = self.load_all_evoked(results_dir, condition)
        
        # Compute group-level ERPs
        group_erps = {}
        for condition, evoked_list in all_evoked.items():
            group_erps[condition] = mne.grand_average(evoked_list)
        
        # Statistical analysis
        stats_results = self.compute_cluster_statistics(all_evoked)
        
        # Topographic analysis
        topo_results = self.analyze_topographies(group_erps)
        
        # Save results
        self.save_group_results(group_erps, stats_results, topo_results, results_dir)
        
        return {
            'group_erps': group_erps,
            'statistics': stats_results,
            'topography': topo_results
        }
    
    def compute_cluster_statistics(self, all_evoked: Dict):
        """Compute cluster-based permutation statistics."""
        from mne.stats import spatio_temporal_cluster_test
        
        conditions = list(all_evoked.keys())
        if len(conditions) != 2:
            raise ValueError("Cluster statistics requires exactly 2 conditions")
        
        # Prepare data arrays
        X = [np.array([evoked.data for evoked in all_evoked[cond]]) 
             for cond in conditions]
        
        # Compute connectivity for cluster correction
        connectivity, ch_names = mne.channels.find_ch_adjacency(
            all_evoked[conditions[0]][0].info, ch_type='eeg'
        )
        
        # Run cluster test
        t_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
            X, n_permutations=1000, threshold=None, tail=0,
            connectivity=connectivity, n_jobs=4
        )
        
        return {
            't_obs': t_obs,
            'clusters': clusters,
            'cluster_p_values': cluster_p_values,
            'significant_clusters': clusters[cluster_p_values < 0.05]
        }

# Usage
analyzer = AdvancedGroupAnalyzer()
group_results = analyzer.compute_group_statistics("results/", ["target", "standard"])
```

## Performance Optimization

### Memory Management

Optimize memory usage for large datasets:

```python
from eeg_processor.utils.memory_tools import MemoryManager

class MemoryOptimizedPipeline(EEGPipeline):
    """Memory-optimized version of EEG pipeline."""
    
    def __init__(self, config_path, memory_limit=0.8):
        super().__init__(config_path)
        self.memory_manager = MemoryManager(memory_limit)
        
    def run_participant(self, participant_id):
        """Run processing with memory monitoring."""
        
        # Monitor memory before processing
        self.memory_manager.check_memory_pressure()
        
        try:
            # Load data
            raw = self.load_participant_data(participant_id)
            
            # Process in chunks if data is large
            if self.memory_manager.estimate_memory_usage(raw) > 0.5:
                return self._process_in_chunks(raw, participant_id)
            else:
                return super().run_participant(participant_id)
                
        except MemoryError:
            # Fallback to chunk processing
            return self._process_in_chunks(raw, participant_id)
    
    def _process_in_chunks(self, raw, participant_id):
        """Process data in smaller chunks."""
        chunk_duration = 60  # 60 seconds per chunk
        results = []
        
        for start_time in range(0, int(raw.times[-1]), chunk_duration):
            end_time = min(start_time + chunk_duration, raw.times[-1])
            
            # Create chunk
            chunk = raw.copy().crop(tmin=start_time, tmax=end_time)
            
            # Process chunk
            chunk_result = self._process_chunk(chunk, participant_id, start_time)
            results.append(chunk_result)
            
            # Clean up memory
            del chunk
            self.memory_manager.force_garbage_collection()
        
        # Combine chunk results
        return self._combine_chunk_results(results, participant_id)
```

### Parallel Processing Optimization

Optimize parallel processing:

```yaml
parallel:
  enabled: true
  
  # Participant-level parallelization
  n_jobs_participants: 4  # Process 4 participants simultaneously
  
  # Stage-level parallelization  
  n_jobs_stages: 2        # Parallelize within each stage
  
  # Memory per job
  memory_per_job: "4G"
  
  # Backend selection
  backend: "multiprocessing"  # or "threading", "loky"
  
  # Chunk size for large operations
  chunk_size: 1000
  
  # Timeout settings
  timeout: 3600  # 1 hour timeout per participant
```

## Integration with Other Tools

### MNE-Python Integration

Seamless integration with MNE-Python:

```python
from eeg_processor import EEGPipeline
import mne

# Process with EEG Processor
pipeline = EEGPipeline("config.yml")
results = pipeline.run_participant("sub-01")

# Continue with MNE-Python
epochs = results['epochs']['target']

# Source localization with MNE
forward = mne.make_forward_solution(
    epochs.info, trans="fsaverage", src=src, bem=bem
)

noise_cov = mne.compute_covariance(epochs, tmax=0.0)
inverse_operator = mne.minimum_norm.make_inverse_operator(
    epochs.info, forward, noise_cov
)

# Apply inverse solution
stc = mne.minimum_norm.apply_inverse(
    results['evoked']['target'], inverse_operator, lambda2=1.0/9.0
)
```

### FieldTrip Export

Export data for FieldTrip analysis:

```python
def export_to_fieldtrip(epochs, output_path):
    """Export epochs to FieldTrip format."""
    from scipy.io import savemat
    
    # Prepare FieldTrip structure
    ft_data = {
        'trial': [epochs.get_data()[i, :, :] for i in range(len(epochs))],
        'time': [epochs.times for _ in range(len(epochs))],
        'label': epochs.ch_names,
        'fsample': epochs.info['sfreq'],
        'trialinfo': epochs.metadata.values if epochs.metadata is not None else None
    }
    
    # Save as .mat file
    savemat(output_path, {'data': ft_data})

# Usage
pipeline = EEGPipeline("config.yml")
results = pipeline.run_participant("sub-01")
export_to_fieldtrip(results['epochs']['target'], "sub-01_target_ft.mat")
```

## Troubleshooting Advanced Issues

### Debug Mode

Enable comprehensive debugging:

```yaml
debugging:
  enabled: true
  save_intermediate: true
  detailed_logging: true
  
  breakpoints:
    - stage: "artifact_rejection"
      condition: "rejection_rate > 0.5"
      action: "interactive_debug"
    
    - stage: "ica"
      condition: "n_components < 10"
      action: "save_debug_info"

  profiling:
    memory: true
    cpu: true
    output_dir: "debug/"
```

### Performance Profiling

Profile your processing pipeline:

```python
import cProfile
import pstats
from eeg_processor import EEGPipeline

def profile_pipeline(config_path, participant_id):
    """Profile pipeline performance."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run pipeline
    pipeline = EEGPipeline(config_path)
    results = pipeline.run_participant(participant_id)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    # Save profile
    stats.dump_stats(f"profile_{participant_id}.prof")
    
    return results

# Usage
results = profile_pipeline("config.yml", "sub-01")
```

This advanced tutorial covers sophisticated use cases and customization options. For specific implementation details, refer to the API documentation and example scripts.