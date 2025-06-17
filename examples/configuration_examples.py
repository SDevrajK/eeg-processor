#!/usr/bin/env python3
"""
Configuration examples for EEG Processor.

This script shows various ways to configure the EEG processing pipeline
for different use cases and data formats.
"""

import yaml
from pathlib import Path


def example_brainvision_config():
    """Example configuration for BrainVision data."""
    config = {
        "data_format": "brainvision",
        "paths": {
            "raw_data_dir": "data/raw_brainvision",
            "results_dir": "data/processed/brainvision_results",
            "interim_dir": "data/interim",
            "figures_dir": "data/figures",
            "file_extension": ".vhdr"
        },
        "participants": {
            "sub-01": "sub-01_task-rest_eeg.vhdr",
            "sub-02": "sub-02_task-rest_eeg.vhdr",
            "sub-03": "sub-03_task-rest_eeg.vhdr"
        },
        "dataset_name": "RestingState_BrainVision",
        "stages": [
            "load_data",
            "montage",
            "filter", 
            "bad_channels",
            "epoching",
            "artifact_rejection",
            "ica",
            "evoked",
            "save_results"
        ],
        "montage": {
            "montage_type": "standard_1020",
            "channel_positions": "auto"
        },
        "filtering": {
            "lowpass": 40,
            "highpass": 0.1,
            "notch": 50,
            "filter_length": "auto",
            "l_trans_bandwidth": "auto",
            "h_trans_bandwidth": "auto"
        },
        "bad_channels": {
            "method": "auto",
            "threshold": 3.0,
            "max_bad_ratio": 0.2
        },
        "epoching": {
            "tmin": -0.2,
            "tmax": 0.8,
            "baseline": [-0.2, 0],
            "preload": True,
            "reject_by_annotation": True
        },
        "artifact_rejection": {
            "peak_to_peak": 100e-6,
            "flat_threshold": 1e-6,
            "reject_criteria": {
                "eeg": 100e-6,
                "eog": 150e-6
            }
        },
        "ica": {
            "n_components": 20,
            "method": "fastica",
            "max_iter": 200,
            "random_state": 42,
            "exclude_components": "auto"
        },
        "conditions": [
            {
                "name": "target",
                "description": "Target stimuli",
                "condition_markers": ["S1", "S11"],
                "baseline": [-0.2, 0],
                "tmin": -0.2,
                "tmax": 0.8
            },
            {
                "name": "standard", 
                "description": "Standard stimuli",
                "condition_markers": ["S2", "S12"],
                "baseline": [-0.2, 0],
                "tmin": -0.2,
                "tmax": 0.8
            }
        ],
        "quality_control": {
            "enabled": True,
            "generate_plots": True,
            "thresholds": {
                "bad_channels_max": 0.2,
                "artifact_rejection_max": 0.3,
                "min_epochs_per_condition": 30
            }
        }
    }
    return config


def example_edf_config():
    """Example configuration for EDF data."""
    config = {
        "data_format": "edf",
        "paths": {
            "raw_data_dir": "data/raw_edf",
            "results_dir": "data/processed/edf_results",
            "file_extension": ".edf"
        },
        "participants": "auto",  # Auto-discover all .edf files
        "dataset_name": "SleepStudy_EDF",
        "stages": [
            "load_data",
            "filter",
            "bad_channels", 
            "epoching",
            "save_results"
        ],
        "filtering": {
            "lowpass": 30,
            "highpass": 0.5,
            "notch": None  # No notch filter
        },
        "bad_channels": {
            "method": "ransac",
            "threshold": 4.0
        },
        "epoching": {
            "tmin": 0,
            "tmax": 30,  # 30-second epochs for sleep data
            "baseline": None,
            "overlap": 0  # No overlap
        },
        "conditions": [
            {
                "name": "wake",
                "description": "Wake epochs",
                "condition_markers": ["Wake"]
            },
            {
                "name": "nrem",
                "description": "Non-REM sleep",
                "condition_markers": ["N1", "N2", "N3"]
            },
            {
                "name": "rem",
                "description": "REM sleep", 
                "condition_markers": ["REM"]
            }
        ]
    }
    return config


def example_fif_config():
    """Example configuration for FIF (MNE-Python native) data."""
    config = {
        "data_format": "fif",
        "paths": {
            "raw_data_dir": "data/raw_fif",
            "results_dir": "data/processed/fif_results",
            "file_extension": "_raw.fif"
        },
        "participants": {
            "subject_01": "subject_01_meg_raw.fif",
            "subject_02": "subject_02_meg_raw.fif"
        },
        "dataset_name": "MEG_Experiment",
        "stages": [
            "load_data",
            "maxwell_filter",  # MEG-specific preprocessing
            "filter",
            "epoching",
            "artifact_rejection",
            "save_results"
        ],
        "maxwell_filter": {
            "st_duration": 10,
            "origin": "auto",
            "coord_frame": "head"
        },
        "filtering": {
            "lowpass": 40,
            "highpass": 1,
            "notch": [50, 100]  # Multiple notch frequencies
        },
        "epoching": {
            "tmin": -0.5,
            "tmax": 1.5,
            "baseline": [-0.5, -0.1],
            "decim": 4  # Downsample by factor of 4
        },
        "conditions": [
            {
                "name": "visual",
                "condition_markers": [1, 2],
                "description": "Visual stimuli"
            },
            {
                "name": "auditory", 
                "condition_markers": [3, 4],
                "description": "Auditory stimuli"
            }
        ]
    }
    return config


def example_minimal_config():
    """Minimal configuration example."""
    config = {
        "paths": {
            "raw_data_dir": "data/raw",
            "results_dir": "data/processed"
        },
        "stages": ["load_data", "filter", "save_results"],
        "filtering": {
            "lowpass": 40,
            "highpass": 0.1
        }
    }
    return config


def example_advanced_config():
    """Advanced configuration with all options."""
    config = {
        "data_format": "brainvision",
        "paths": {
            "raw_data_dir": "data/raw",
            "results_dir": "data/processed/advanced",
            "interim_dir": "data/interim", 
            "figures_dir": "data/figures",
            "file_extension": ".vhdr"
        },
        "participants": "auto",
        "dataset_name": "AdvancedProcessing",
        "stages": [
            "load_data",
            "montage",
            "filter",
            "bad_channels", 
            "reref",
            "epoching",
            "artifact_rejection",
            "ica",
            "time_frequency",
            "evoked",
            "save_results"
        ],
        "montage": {
            "montage_type": "easycap-M1",
            "channel_positions": "data/montages/custom_montage.elc"
        },
        "filtering": {
            "lowpass": 45,
            "highpass": 0.5,
            "notch": [50, 100],
            "filter_length": "10s",
            "phase": "zero",
            "fir_window": "hamming"
        },
        "bad_channels": {
            "method": "ransac",
            "threshold": 4.0,
            "max_bad_ratio": 0.15,
            "correlation_threshold": 0.75,
            "noise_threshold": 5.0
        },
        "reref": {
            "reference": "average",
            "exclude_bads": True
        },
        "epoching": {
            "tmin": -0.5,
            "tmax": 2.0,
            "baseline": [-0.5, -0.1], 
            "preload": True,
            "reject_by_annotation": True,
            "metadata_columns": ["condition", "response_time"]
        },
        "artifact_rejection": {
            "method": "autoreject",
            "peak_to_peak": 150e-6,
            "flat_threshold": 1e-6,
            "reject_criteria": {
                "eeg": 150e-6,
                "eog": 250e-6,
                "emg": 300e-6
            },
            "interpolation": True,
            "consensus": 0.1
        },
        "ica": {
            "n_components": 25,
            "method": "infomax",
            "max_iter": 500,
            "random_state": 42,
            "exclude_components": "auto",
            "eog_channels": ["EOG1", "EOG2"],
            "ecg_channels": ["ECG"],
            "threshold": 3.0
        },
        "time_frequency": {
            "method": "morlet",
            "freqs": [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40],
            "n_cycles": 3,
            "baseline": [-0.5, -0.1],
            "baseline_mode": "percent"
        },
        "conditions": [
            {
                "name": "congruent",
                "description": "Congruent trials",
                "condition_markers": ["S1", "S11"],
                "baseline": [-0.5, -0.1],
                "tmin": -0.5,
                "tmax": 2.0,
                "metadata_query": "condition == 'congruent'"
            },
            {
                "name": "incongruent",
                "description": "Incongruent trials", 
                "condition_markers": ["S2", "S12"],
                "baseline": [-0.5, -0.1],
                "tmin": -0.5,
                "tmax": 2.0,
                "metadata_query": "condition == 'incongruent'"
            }
        ],
        "quality_control": {
            "enabled": True,
            "generate_plots": True,
            "save_intermediate": True,
            "thresholds": {
                "bad_channels_max": 0.15,
                "artifact_rejection_max": 0.25,
                "min_epochs_per_condition": 50,
                "max_processing_time": 3600,  # 1 hour max
                "memory_limit": 0.8  # 80% of available memory
            },
            "plots": {
                "raw_psd": True,
                "epochs_image": True,
                "evoked_comparison": True,
                "ica_components": True,
                "time_frequency": True
            }
        },
        "parallel": {
            "n_jobs": 4,
            "backend": "multiprocessing"
        },
        "logging": {
            "level": "INFO",
            "save_log": True,
            "log_file": "processing.log"
        }
    }
    return config


def save_config_examples():
    """Save all configuration examples to YAML files."""
    examples = {
        "brainvision_config.yml": example_brainvision_config(),
        "edf_config.yml": example_edf_config(), 
        "fif_config.yml": example_fif_config(),
        "minimal_config.yml": example_minimal_config(),
        "advanced_config.yml": example_advanced_config()
    }
    
    # Create examples directory
    examples_dir = Path("examples/configs")
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, config in examples.items():
        config_path = examples_dir / filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Saved {filename}")
    
    print(f"\nConfiguration examples saved to: {examples_dir}")


if __name__ == "__main__":
    print("EEG Processor - Configuration Examples")
    print("=" * 50)
    
    save_config_examples()
    
    print("\nExample configurations:")
    print("- brainvision_config.yml: Complete BrainVision setup")
    print("- edf_config.yml: EDF data processing") 
    print("- fif_config.yml: FIF/MEG data processing")
    print("- minimal_config.yml: Minimal required settings")
    print("- advanced_config.yml: All available options")
    
    print("\nTo use these configurations:")
    print("1. Copy the relevant example to your project")
    print("2. Modify paths and parameters for your data")
    print("3. Load with: pipeline = EEGPipeline('path/to/config.yml')")