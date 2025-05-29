from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.eeg_processor.pipeline import EEGPipeline
from typing import List


def plot_continuous_tfr_single_channel(continuous_tfr, channel_name: str,
                                       freq_range: tuple = (8, 25),
                                       title_suffix: str = ""):
    """Plot continuous TFR for a single channel"""
    if channel_name not in continuous_tfr.ch_names:
        print(f"Channel {channel_name} not found in data")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get channel index and crop to frequency range
    tfr_cropped = continuous_tfr.copy().crop(fmin=freq_range[0], fmax=freq_range[1])

    # Plot spectrogram
    im = tfr_cropped.plot([channel_name], axes=ax, show=False, colorbar=True)

    ax.set_title(f"Continuous TFR: {channel_name} - {title_suffix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    return fig


def plot_continuous_tfr_multi_channel(continuous_tfr, channel_names: List[str],
                                      freq_range: tuple = (8, 25),
                                      title_suffix: str = ""):
    """Plot continuous TFR for multiple channels in subplot grid"""
    # Filter available channels
    available_channels = [ch for ch in channel_names if ch in continuous_tfr.ch_names]

    if not available_channels:
        print(f"None of the requested channels found in data: {channel_names}")
        return None

    n_channels = len(available_channels)
    n_cols = 2
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_channels == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    # Crop to frequency range
    tfr_cropped = continuous_tfr.copy().crop(fmin=freq_range[0], fmax=freq_range[1])

    for i, channel in enumerate(available_channels):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        # Plot individual channel
        tfr_cropped.plot([channel], axes=ax, show=False, colorbar=False)
        ax.set_title(f"{channel}")

        if row == n_rows - 1:  # Bottom row
            ax.set_xlabel("Time (s)")
        if col == 0:  # Left column
            ax.set_ylabel("Frequency (Hz)")

    # Hide unused subplots
    for i in range(n_channels, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    fig.suptitle(f"Continuous TFR: Multiple Channels - {title_suffix}", fontsize=14)
    plt.tight_layout()

    return fig


def plot_tfr_time_course(continuous_tfr, channel_names: List[str],
                         freq_band: tuple = (10, 15), title_suffix: str = ""):
    """Plot power time course in specific frequency band for multiple channels"""
    # Filter available channels
    available_channels = [ch for ch in channel_names if ch in continuous_tfr.ch_names]

    if not available_channels:
        print(f"None of the requested channels found: {channel_names}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Crop to frequency band and average across frequencies
    tfr_band = continuous_tfr.copy().crop(fmin=freq_band[0], fmax=freq_band[1])

    # Plot time course for each channel
    colors = plt.cm.tab10(range(len(available_channels)))

    for i, channel in enumerate(available_channels):
        ch_idx = tfr_band.ch_names.index(channel)
        # Average power across frequency band
        power_timecourse = np.mean(tfr_band.data[ch_idx, :, :], axis=0)
        ax.plot(tfr_band.times, power_timecourse,
                color=colors[i], label=channel, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Power ({freq_band[0]}-{freq_band[1]} Hz)")
    ax.set_title(f"Power Time Course: {freq_band[0]}-{freq_band[1]} Hz - {title_suffix}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_rhythmic_activity(continuous_tfr, channels_of_interest: List[str]):
    """Analyze the rhythmic activity patterns"""
    print("\n=== Rhythmic Activity Analysis ===")

    # Frequency range of interest (based on your observation)
    freq_range = (8, 45)
    rhythm_band = (20, 40)

    # Check data properties
    print(f"Data duration: {continuous_tfr.times[-1]:.1f} seconds")
    print(f"Frequency range: {continuous_tfr.freqs[0]:.1f} - {continuous_tfr.freqs[-1]:.1f} Hz")
    print(f"Available channels of interest: {[ch for ch in channels_of_interest if ch in continuous_tfr.ch_names]}")

    # Calculate power in rhythm band for each channel
    tfr_rhythm = continuous_tfr.copy().crop(fmin=rhythm_band[0], fmax=rhythm_band[1])

    for channel in channels_of_interest:
        if channel in continuous_tfr.ch_names:
            ch_idx = tfr_rhythm.ch_names.index(channel)
            mean_power = np.mean(tfr_rhythm.data[ch_idx, :, :])
            max_power = np.max(tfr_rhythm.data[ch_idx, :, :])
            print(f"{channel}: Mean power = {mean_power:.6f}, Max power = {max_power:.6f}")


def run_continuous_tfr_analysis(config_path: str, participant_id: str = "S_003_F",
                                condition_name: str = "Baseline"):
    """Main analysis script for continuous TFR data"""

    # Get analysis interface from pipeline
    pipeline = EEGPipeline(config_path)
    data_interface = pipeline.get_analysis_interface()

    # Load continuous TFR data
    print(f"Loading continuous TFR data for {participant_id}, {condition_name}...")
    continuous_tfr = data_interface.load_data(participant_id, condition_name, "continuous_tfr")

    if continuous_tfr is None:
        print(f"No continuous TFR data found for {participant_id}, {condition_name}")
        print("Make sure you've run the pipeline with time_frequency_raw stage and output_type='raw_tfr'")
        return None

    # Channels showing rhythmic activity
    channels_of_interest = ['F7', 'F8', 'T7', 'T8', 'FT9', 'FT10']

    # Create output directory
    output_dir = Path("continuous_tfr_analysis")
    output_dir.mkdir(exist_ok=True)

    # Analyze rhythmic patterns
    analyze_rhythmic_activity(continuous_tfr, channels_of_interest)

    # 1. Individual channel spectrograms for key channels
    print("\nGenerating individual channel spectrograms...")
    for channel in ['F7', 'F8', 'Fz']:  # Include Fz for comparison
        if channel in continuous_tfr.ch_names:
            fig = plot_continuous_tfr_single_channel(
                continuous_tfr, channel,
                freq_range=(8, 45),
                title_suffix=f"{participant_id} - {condition_name}"
            )
            if fig:
                fig.savefig(output_dir / f"tfr_spectrogram_{channel}_{participant_id}.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

    # 2. Multi-channel overview of rhythmic channels
    print("Generating multi-channel overview...")
    fig = plot_continuous_tfr_multi_channel(
        continuous_tfr, channels_of_interest,
        freq_range=(8, 45),
        title_suffix=f"{participant_id} - {condition_name}"
    )
    if fig:
        fig.savefig(output_dir / f"tfr_multichannel_{participant_id}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()

    # 3. Power time course in rhythm band
    print("Generating power time course...")
    fig = plot_tfr_time_course(
        continuous_tfr, channels_of_interest,
        freq_band=(20, 40),
        title_suffix=f"{participant_id} - {condition_name}"
    )
    if fig:
        fig.savefig(output_dir / f"tfr_timecourse_{participant_id}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    return continuous_tfr


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "test_config/RIEEG_baseline_singlesubject_test_processing_params.yml"
    continuous_tfr = run_continuous_tfr_analysis(config_path)