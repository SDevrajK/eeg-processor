from pathlib import Path
import matplotlib.pyplot as plt
from src.eeg_processor.pipeline import EEGPipeline
from typing import List
from mne import grand_average

def plot_individual_spectrum(data_interface, participant_id: str, condition_name: str):
    """Plot spectrum for individual participant - diagnostic style"""
    spectrum = data_interface.load_data(participant_id, condition_name, "spectrum")
    if spectrum is None:
        print(f"No data found for {participant_id}, {condition_name}")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    spectrum.plot(axes=ax, show=False)
    ax.set_title(f"Baseline Spectrum: {participant_id} - {condition_name}")
    return fig


def plot_group_overlay(data_interface, condition_name: str, channels: List[str] = ['Fz']):
    """Plot all participants overlaid for analytical comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))

    participant_ids = data_interface.get_participant_ids()
    spectra = []
    colors = plt.cm.tab10(range(len(participant_ids)))  # Generate distinct colors

    # Plot individual participants with unique colors and labels
    for i, participant_id in enumerate(participant_ids):
        spectrum = data_interface.load_data(participant_id, condition_name, "spectrum")
        if spectrum is not None:
            # Clean participant ID for label
            clean_id = participant_id.replace("_", "")

            # Get channel data manually
            ch_idx = [spectrum.ch_names.index(ch) for ch in channels if ch in spectrum.ch_names]
            if ch_idx:
                data = spectrum.get_data()[ch_idx[0]]  # Get first requested channel
                freqs = spectrum.freqs

                # Plot manually with label
                ax.plot(freqs, data, color=colors[i], alpha=0.7, label=clean_id)
                spectra.append(spectrum)

    # Calculate and plot grand average
    if spectra:
        import numpy as np
        all_data = np.array([s.get_data() for s in spectra])
        avg_data = np.mean(all_data, axis=0)

        # Plot grand average with thick grey line
        ch_idx = [spectra[0].ch_names.index(ch) for ch in channels if ch in spectra[0].ch_names]
        if ch_idx:
            ax.plot(spectra[0].freqs, avg_data[ch_idx[0]],
                    color='grey', linewidth=3, label='Grand Average')

    # Configure compact legend and axis labels
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9,
              ncol=2 if len(participant_ids) > 6 else 1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB)')
    ax.set_title(f"Group Baseline Spectrum: {condition_name} - {channels}")

    return fig

def run_baseline_analysis(config_path: str):
    """Main analysis script for baseline spectrum data"""

    # Get analysis interface from pipeline
    pipeline = EEGPipeline(config_path)
    data_interface = pipeline.get_analysis_interface()

    # Check what data is available
    available_data = data_interface.list_available_data("spectrum")
    print(f"Available data: {available_data}")

    condition_name = data_interface.get_condition_names()[0]  # First condition

    # Create output directory
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    # 1. Generate individual diagnostic plots
    print("Generating individual diagnostic plots...")
    for participant_id in data_interface.get_participant_ids():
        if condition_name in available_data.get(participant_id, []):
            fig = plot_individual_spectrum(data_interface, participant_id, condition_name)
            if fig:
                fig.savefig(output_dir / f"diagnostic_{participant_id}_{condition_name}.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

    # 2. Generate group overlay
    print("Generating group analysis...")
    fig = plot_group_overlay(data_interface, condition_name, channels=['Fz'])
    if fig:
        fig.savefig(output_dir / f"group_overlay_{condition_name}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    return data_interface


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "test_config/RIEEG_baseline_test_processing_params.yml"
    data_interface = run_baseline_analysis(config_path)