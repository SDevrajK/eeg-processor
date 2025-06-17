#!/usr/bin/env python3
"""
Standard ERP Analysis Workflow

This script demonstrates a complete ERP analysis workflow using EEG Processor,
from raw data processing to group-level statistics and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from eeg_processor import EEGPipeline
from eeg_processor.quality_control import generate_quality_reports


def run_standard_erp_analysis():
    """Complete ERP analysis workflow."""
    
    # Configuration for P300 oddball paradigm
    config = {
        "data_format": "brainvision",
        "paths": {
            "raw_data": "data/raw_erp",
            "results": "data/results_erp"
        },
        "participants": "auto",
        "dataset_name": "P300_Oddball_Study",
        
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
            "montage_type": "standard_1020"
        },
        
        "filtering": {
            "lowpass": 40,
            "highpass": 0.1,
            "notch": 50
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
            "exclude_components": "auto"
        },
        
        "conditions": [
            {
                "name": "target",
                "description": "Target stimuli (rare)",
                "condition_markers": ["S1", "S11"],
                "baseline": [-0.2, 0],
                "tmin": -0.2,
                "tmax": 0.8
            },
            {
                "name": "standard",
                "description": "Standard stimuli (frequent)",
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
                "artifact_rejection_max": 0.4,
                "min_epochs_per_condition": 30
            }
        }
    }
    
    print("🧠 Starting Standard ERP Analysis")
    print("=" * 50)
    
    # Step 1: Process all participants
    print("Step 1: Processing participants...")
    pipeline = EEGPipeline(config)
    results = pipeline.run_all()
    
    print(f"✓ Processed {len(results)} participants")
    print(f"Results saved to: {pipeline.config.results_dir}")
    
    # Step 2: Generate quality reports
    print("\nStep 2: Generating quality reports...")
    quality_reports = generate_quality_reports(pipeline.config.results_dir)
    print(f"✓ Generated quality reports for {len(quality_reports)} participants")
    
    # Step 3: Group-level analysis
    print("\nStep 3: Group-level analysis...")
    analysis = pipeline.get_analysis_interface()
    
    # Load all participants' data
    participant_ids = list(results.keys())
    
    # Compute grand averages
    grand_avg_target = analysis.compute_grand_average("target", participant_ids)
    grand_avg_standard = analysis.compute_grand_average("standard", participant_ids)
    
    print(f"✓ Computed grand averages")
    print(f"  Target: {grand_avg_target.data.shape}")
    print(f"  Standard: {grand_avg_standard.data.shape}")
    
    # Step 4: Statistical analysis
    print("\nStep 4: Statistical analysis...")
    stats_results = analysis.compute_condition_contrast(
        ["target", "standard"], 
        participant_ids,
        method="cluster_permutation"
    )
    
    print("✓ Completed statistical analysis")
    print(f"  Found {len(stats_results['significant_clusters'])} significant clusters")
    
    # Step 5: Visualization
    print("\nStep 5: Creating visualizations...")
    figures_dir = Path(pipeline.config.results_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Plot grand averages
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Butterfly plots
    grand_avg_target.plot(axes=axes[0, 0], show=False, spatial_colors=True)
    axes[0, 0].set_title("Target - Butterfly Plot")
    
    grand_avg_standard.plot(axes=axes[0, 1], show=False, spatial_colors=True)
    axes[0, 1].set_title("Standard - Butterfly Plot")
    
    # Topographic maps at P300 peak (~300ms)
    grand_avg_target.plot_topomap(times=0.3, axes=axes[1, 0], show=False)
    axes[1, 0].set_title("Target - 300ms Topography")
    
    grand_avg_standard.plot_topomap(times=0.3, axes=axes[1, 1], show=False)
    axes[1, 1].set_title("Standard - 300ms Topography")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "grand_averages.png", dpi=300)
    plt.close()
    
    # Plot difference wave
    difference_wave = grand_avg_target.copy()
    difference_wave.data = grand_avg_target.data - grand_avg_standard.data
    difference_wave.comment = "Target - Standard"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    difference_wave.plot(axes=ax, show=False, spatial_colors=True)
    ax.set_title("Difference Wave (Target - Standard)")
    plt.savefig(figures_dir / "difference_wave.png", dpi=300)
    plt.close()
    
    # Plot statistical results
    if stats_results['significant_clusters']:
        fig = analysis.plot_statistical_results(stats_results, save_path=figures_dir / "statistics.png")
    
    print(f"✓ Visualizations saved to: {figures_dir}")
    
    # Step 6: Generate summary report
    print("\nStep 6: Generating summary report...")
    summary = {
        "study": "P300 Oddball Paradigm",
        "participants": len(results),
        "conditions": ["target", "standard"],
        "processing_stages": len(config["stages"]),
        "quality_control": "enabled",
        "statistical_analysis": "cluster-based permutation test",
        "significant_effects": len(stats_results['significant_clusters']) > 0
    }
    
    # Save summary
    import json
    with open(pipeline.config.results_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n📊 Analysis Summary")
    print("-" * 30)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Standard ERP analysis completed!")
    print(f"📁 All results saved to: {pipeline.config.results_dir}")
    print(f"📊 Figures saved to: {figures_dir}")
    
    return {
        "config": config,
        "results": results,
        "grand_averages": {
            "target": grand_avg_target,
            "standard": grand_avg_standard
        },
        "statistics": stats_results,
        "summary": summary
    }


def create_analysis_report(results_dict, output_path):
    """Create a comprehensive analysis report."""
    
    report_content = f"""
# ERP Analysis Report

## Study Overview
- **Study**: {results_dict['summary']['study']}
- **Participants**: {results_dict['summary']['participants']}
- **Conditions**: {', '.join(results_dict['summary']['conditions'])}
- **Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Processing Pipeline
- **Stages**: {results_dict['summary']['processing_stages']} processing stages
- **Quality Control**: {results_dict['summary']['quality_control']}
- **Data Format**: {results_dict['config']['data_format']}

## Results Summary
- **Statistical Analysis**: {results_dict['summary']['statistical_analysis']}
- **Significant Effects**: {'Yes' if results_dict['summary']['significant_effects'] else 'No'}
- **Number of Clusters**: {len(results_dict['statistics']['significant_clusters'])}

## Quality Control
Quality control was enabled throughout the processing pipeline with the following thresholds:
- Maximum bad channels: {results_dict['config']['quality_control']['thresholds']['bad_channels_max'] * 100}%
- Maximum artifact rejection: {results_dict['config']['quality_control']['thresholds']['artifact_rejection_max'] * 100}%
- Minimum epochs per condition: {results_dict['config']['quality_control']['thresholds']['min_epochs_per_condition']}

## Files Generated
- Grand average ERPs for each condition
- Difference waves (target - standard)
- Statistical analysis results
- Quality control reports
- Topographic maps
- Individual participant results

## Recommendations
1. Review quality control reports for each participant
2. Examine grand average waveforms for expected components
3. Verify statistical results align with hypotheses
4. Consider additional analyses based on findings

Generated by EEG Processor v{eeg_processor.__version__}
"""
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Analysis report saved to: {output_path}")


if __name__ == "__main__":
    # Run the complete analysis
    analysis_results = run_standard_erp_analysis()
    
    # Create detailed report
    create_analysis_report(
        analysis_results, 
        Path(analysis_results['config']['paths']['results_dir']) / "analysis_report.md"
    )
    
    print("\n🎉 Analysis workflow completed successfully!")
    print("\nNext steps:")
    print("1. Review quality control reports")
    print("2. Examine generated figures")
    print("3. Read the analysis report")
    print("4. Consider additional analyses or adjustments")