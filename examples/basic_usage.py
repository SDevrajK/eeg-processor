#!/usr/bin/env python3
"""
Basic usage examples for EEG Processor.

This script demonstrates the most common use cases for the EEG processing pipeline.
"""

from pathlib import Path
from eeg_processor import EEGPipeline, load_config

def example_1_basic_pipeline():
    """Example 1: Basic pipeline usage with configuration file."""
    print("=== Example 1: Basic Pipeline Usage ===")
    
    # Create pipeline from config file
    pipeline = EEGPipeline("config/processing_params.yml")
    
    # Process all participants
    results = pipeline.run_all()
    
    print(f"Processing complete. Results saved to: {pipeline.config.results_dir}")
    print(f"Processed {len(results)} participants")


def example_2_single_participant():
    """Example 2: Process a single participant."""
    print("=== Example 2: Single Participant Processing ===")
    
    pipeline = EEGPipeline("config/processing_params.yml")
    
    # Process specific participant
    participant_id = "sub-01"
    result = pipeline.run_participant(participant_id)
    
    print(f"Processed participant {participant_id}")
    print(f"Final data shape: {result['epochs'].get_data().shape}")


def example_3_programmatic_config():
    """Example 3: Create configuration programmatically."""
    print("=== Example 3: Programmatic Configuration ===")
    
    config_dict = {
        "paths": {
            "raw_data_dir": "data/raw",
            "results_dir": "data/processed",
            "file_extension": ".vhdr"
        },
        "participants": ["sub-01", "sub-02", "sub-03"],
        "stages": [
            "load_data",
            "filter",
            "epoching", 
            "artifact_rejection",
            "save_results"
        ],
        "filtering": {
            "lowpass": 40,
            "highpass": 0.1,
            "notch": 50
        },
        "epoching": {
            "tmin": -0.2,
            "tmax": 0.8,
            "baseline": [-0.2, 0]
        },
        "conditions": [
            {
                "name": "target",
                "condition_markers": ["S1", "S2"]
            },
            {
                "name": "standard", 
                "condition_markers": ["S3", "S4"]
            }
        ]
    }
    
    # Create pipeline from dictionary
    pipeline = EEGPipeline(config_dict)
    
    # Process data
    results = pipeline.run_all()
    print(f"Processed {len(results)} participants with programmatic config")


def example_4_quality_control():
    """Example 4: Using quality control features."""
    print("=== Example 4: Quality Control ===")
    
    # Load config with quality control enabled
    config_dict = {
        "paths": {
            "raw_data_dir": "data/raw",
            "results_dir": "data/processed"
        },
        "quality_control": {
            "enabled": True,
            "generate_plots": True,
            "thresholds": {
                "bad_channels_max": 0.2,  # Max 20% bad channels
                "artifact_rejection_max": 0.3  # Max 30% rejected epochs
            }
        }
    }
    
    pipeline = EEGPipeline(config_dict)
    
    # Access quality tracker
    quality_tracker = pipeline.quality_tracker
    
    # Process data with quality tracking
    results = pipeline.run_all()
    
    # Generate quality report
    from eeg_processor.quality_control import generate_quality_reports
    reports = generate_quality_reports(pipeline.config.results_dir)
    
    print(f"Quality reports generated: {len(reports)} participants")


def example_5_interactive_processing():
    """Example 5: Interactive step-by-step processing."""
    print("=== Example 5: Interactive Processing ===")
    
    pipeline = EEGPipeline("config/processing_params.yml")
    
    # Load specific participant data
    participant_id = "sub-01"
    raw_data = pipeline.load_participant_data(participant_id)
    print(f"Loaded raw data: {raw_data.info}")
    
    # Apply individual processing stages
    filtered_data = pipeline.apply_stage(raw_data, "filter", 
                                       lowpass=40, highpass=0.1)
    print(f"Applied filtering: {filtered_data.info['lowpass']:.1f} Hz")
    
    # Create epochs for specific condition
    condition = {"name": "target", "condition_markers": ["S1", "S2"]}
    epochs = pipeline.apply_stage(filtered_data, "epoching", 
                                condition=condition, tmin=-0.2, tmax=0.8)
    print(f"Created epochs: {len(epochs)} trials")
    
    # Apply artifact rejection
    clean_epochs = pipeline.apply_stage(epochs, "artifact_rejection",
                                      peak_to_peak=100e-6)
    print(f"After artifact rejection: {len(clean_epochs)} trials remaining")


def example_6_analysis_interface():
    """Example 6: Using the analysis interface for post-processing."""
    print("=== Example 6: Analysis Interface ===")
    
    pipeline = EEGPipeline("config/processing_params.yml")
    
    # Get analysis interface
    analysis = pipeline.get_analysis_interface()
    
    # Load processed data
    epochs = analysis.load_epochs("sub-01", "target")
    evoked = analysis.load_evoked("sub-01", "target") 
    
    print(f"Loaded epochs: {epochs.get_data().shape}")
    print(f"Loaded evoked: {evoked.data.shape}")
    
    # Compute grand average across participants
    grand_avg = analysis.compute_grand_average("target", 
                                             participants=["sub-01", "sub-02", "sub-03"])
    print(f"Grand average: {grand_avg.data.shape}")
    
    # Generate comparison plots
    analysis.plot_condition_comparison(["target", "standard"],
                                     save_path="figures/condition_comparison.png")


def example_7_custom_processing():
    """Example 7: Custom processing with override parameters."""
    print("=== Example 7: Custom Processing ===")
    
    # Load base config
    base_config = load_config("config/processing_params.yml")
    
    # Override specific parameters
    custom_params = {
        "filtering": {"lowpass": 30},  # Different filter settings
        "epoching": {"tmin": -0.5, "tmax": 1.0},  # Longer epochs
        "artifact_rejection": {"peak_to_peak": 150e-6}  # More lenient rejection
    }
    
    # Create pipeline with overrides
    pipeline = EEGPipeline(base_config, custom_params)
    
    # Process with custom settings
    results = pipeline.run_all()
    print(f"Processed with custom parameters: {len(results)} participants")


def example_8_batch_processing():
    """Example 8: Batch processing multiple datasets."""
    print("=== Example 8: Batch Processing ===")
    
    datasets = [
        {"config": "config/dataset1_params.yml", "name": "Dataset1"},
        {"config": "config/dataset2_params.yml", "name": "Dataset2"},
        {"config": "config/dataset3_params.yml", "name": "Dataset3"}
    ]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"Processing {dataset['name']}...")
        
        pipeline = EEGPipeline(dataset["config"])
        results = pipeline.run_all()
        
        all_results[dataset["name"]] = {
            "participants": len(results),
            "results_dir": pipeline.config.results_dir
        }
        
        print(f"  Completed: {len(results)} participants")
    
    print("\nBatch processing summary:")
    for name, info in all_results.items():
        print(f"  {name}: {info['participants']} participants -> {info['results_dir']}")


if __name__ == "__main__":
    print("EEG Processor - Usage Examples")
    print("=" * 50)
    
    # Run examples (commented out to avoid actual execution)
    # Uncomment the examples you want to run:
    
    # example_1_basic_pipeline()
    # example_2_single_participant() 
    # example_3_programmatic_config()
    # example_4_quality_control()
    # example_5_interactive_processing()
    # example_6_analysis_interface()
    # example_7_custom_processing()
    # example_8_batch_processing()
    
    print("\nTo run examples, uncomment the desired function calls above.")
    print("Make sure you have valid configuration files and data paths.")