import matplotlib.pyplot as plt
import mne
import numpy as np
from utils.erp_tools import find_erp_peaks
from pathlib import Path


def plot_grand_average(participants, conditions, base_path, condition_idx=0, electrode='Fz'):
    """Plot grand average across participants for specified condition and electrode."""

    # Define color palette
    blue_palette = [
        '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5',
        '#084594'
    ]

    # Initialize storage
    all_diff_waves = {cond: {} for cond in conditions}
    all_peak_data = {cond: {} for cond in conditions}

    # Process each participant
    for participant in participants:
        try:
            # Load data
            stan = mne.read_epochs(base_path / f"{participant}_{conditions[condition_idx]}_Standards-epo.fif")
            dev = mne.read_epochs(base_path / f"{participant}_{conditions[condition_idx]}_Deviants-epo.fif")
            stan_evoked = stan.average()

            # Process deviants
            for dev_label in sorted(dev.metadata['event_label'].unique(),
                                    key=lambda x: int(x.replace('ms', ''))):
                dev_evoked = dev[dev_label].average()
                diff_wave = mne.combine_evoked([dev_evoked, stan_evoked], [1, -1])

                # Store for grand average
                if dev_label not in all_diff_waves[conditions[condition_idx]]:
                    all_diff_waves[conditions[condition_idx]][dev_label] = []
                all_diff_waves[conditions[condition_idx]][dev_label].append(diff_wave)

        except FileNotFoundError:
            print(f"Missing data for {participant}")
            continue

    # Compute grand averages
    grand_avg_diff = {}
    grand_peak_data = {}
    for dev_label in all_diff_waves[conditions[condition_idx]]:
        if all_diff_waves[conditions[condition_idx]][dev_label]:
            grand_avg_diff[dev_label] = mne.grand_average(all_diff_waves[conditions[condition_idx]][dev_label])
            grand_peak_data[dev_label] = find_erp_peaks(
                grand_avg_diff[dev_label],
                electrode=electrode,
                mmn_window=(0.175, 0.275),
                p2_window=(0.125, 0.175),
                dynamic_p2_window=True
            )

    # Create figure
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, width_ratios=[1.5, 1.5, 0.9, 0.1],
                          height_ratios=[1.6, 1.6, 0.7, 0.7])

    # Assign subplots
    ax_erp = fig.add_subplot(gs[0:2, :])
    ax_mmn = fig.add_subplot(gs[2:4, 0])
    ax_diff = fig.add_subplot(gs[2:4, 1])
    ax_topomap = fig.add_subplot(gs[2:4, 2])
    ax_topobar = fig.add_subplot(gs[2:4, 3])

    # 1. Plot ERP with markers
    mne.viz.plot_compare_evokeds(
        grand_avg_diff,
        picks=electrode,
        title=f'Grand Average: {conditions[condition_idx]} at {electrode}',
        combine=None,
        colors=blue_palette,
        axes=ax_erp,
        show=False
    )

    # Add markers
    deviant_labels = sorted(grand_peak_data.keys(), key=lambda x: int(x.replace('ms', '')))
    for idx, label in enumerate(deviant_labels):
        peaks = grand_peak_data[label]
        color = blue_palette[idx]

        if peaks['mmn_latency']:
            ax_erp.plot(peaks['mmn_latency'], peaks['mmn_amplitude'],
                        'v', color=color, markersize=9,
                        markeredgecolor='red', markerfacecolor=color, zorder=3)
            ax_erp.text(peaks['mmn_latency'], peaks['mmn_amplitude'] - 0.5,
                        'MMN', ha='center', va='top',
                        color='red', fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

        if peaks['p2_latency']:
            ax_erp.plot(peaks['p2_latency'], peaks['p2_amplitude'],
                        '^', color=color, markersize=9,
                        markeredgecolor='blue', markerfacecolor=color, zorder=3)
            ax_erp.text(peaks['p2_latency'], peaks['p2_amplitude'] + 0.5,
                        'P2', ha='center', va='bottom',
                        color='blue', fontsize=6,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

    # 2. Prepare bar plot data
    durations = [int(label.replace('ms', '')) for label in deviant_labels]
    mmn_amps = [grand_peak_data[label]['mmn_amplitude'] for label in deviant_labels]
    p2_mmn_diffs = []

    for label in deviant_labels:
        if (grand_peak_data[label]['p2_amplitude'] is not None and
                grand_peak_data[label]['mmn_amplitude'] is not None):
            p2_mmn_diffs.append(grand_peak_data[label]['p2_amplitude'] - grand_peak_data[label]['mmn_amplitude'])
        else:
            p2_mmn_diffs.append(None)

    # 3. Plot MMN amplitudes
    bars_mmn = ax_mmn.bar(durations, mmn_amps, color=blue_palette[:len(deviant_labels)])
    ax_mmn.set_title('MMN Amplitude', pad=10)
    ax_mmn.set_ylabel('μV', labelpad=5)
    ax_mmn.set_xticks(durations)
    ax_mmn.set_xticklabels(durations, rotation=45)
    ax_mmn.axhline(0, color='black', linewidth=0.8)

    # 4. Plot P2-MMN differences
    bars_diff = ax_diff.bar(durations, p2_mmn_diffs, color=blue_palette[:len(deviant_labels)])
    ax_diff.set_title('P2-MMN Difference', pad=10)
    ax_diff.set_xlabel('Deviant Duration (ms)', labelpad=10)
    ax_diff.set_ylabel('μV', labelpad=5)
    ax_diff.set_xticks(durations)
    ax_diff.set_xticklabels(durations, rotation=45)
    ax_diff.axhline(0, color='black', linewidth=0.8)

    # 5. Enhanced Topography Plot
    if any(grand_peak_data[label]['mmn_latency'] for label in deviant_labels):
        robust_dev = max(deviant_labels,
                         key=lambda x: abs(grand_peak_data[x]['mmn_amplitude'])
                         if grand_peak_data[x]['mmn_amplitude'] else 0)
        mmn_lat = grand_peak_data[robust_dev]['mmn_latency']

        grand_avg = mne.grand_average(list(grand_avg_diff.values()))

        # Plot with sensor markers
        grand_avg.plot_topomap(
            times=mmn_lat,
            average=0.05,
            axes=[ax_topomap, ax_topobar],
            show=False,
            sensors=True,
            outlines='head',
            contours=6,
            cmap='RdBu_r'
        )
        ax_topomap.set_title(
            f'MMN Scalp Distribution\n({robust_dev} deviant @ {mmn_lat * 1000:.0f}±25ms)',
            pad=20
        )

    # Save the figure
    output_path = Path("C:/Users/sayee/Documents/Research/PythonCode/EEG_Processor/tests/test_results/figures")
    fig.savefig(
        output_path / f"GrandAverage_{conditions[condition_idx]}_{electrode}.png",
        dpi=300, bbox_inches='tight')

    plt.show(block=True)


def plot_conditions_by_deviant(participants, conditions, base_path, electrode='Fz'):
    """Plot conditions overlaid for each deviant, with each deviant as a subplot."""

    # Define color palette (one color per condition)
    condition_colors = {
        "BroadBand": "#1f77b4",
        "NarrowBand Within-Channel": "#ff7f0e",
        "NarrowBand Between-Channel": "#2ca02c"
    }

    # Initialize storage for all conditions
    all_diff_waves = {cond: {} for cond in conditions}

    # Process each participant and condition
    for condition in conditions:
        for participant in participants:
            try:
                # Load data
                stan = mne.read_epochs(base_path / f"{participant}_{condition}_Standards-epo.fif")
                dev = mne.read_epochs(base_path / f"{participant}_{condition}_Deviants-epo.fif")
                stan_evoked = stan.average()

                # Process deviants
                for dev_label in sorted(dev.metadata['event_label'].unique(),
                                        key=lambda x: int(x.replace('ms', ''))):
                    dev_evoked = dev[dev_label].average()
                    diff_wave = mne.combine_evoked([dev_evoked, stan_evoked], [1, -1])

                    # Store for grand average
                    if dev_label not in all_diff_waves[condition]:
                        all_diff_waves[condition][dev_label] = []
                    all_diff_waves[condition][dev_label].append(diff_wave)

            except FileNotFoundError:
                print(f"Missing data for {participant} in {condition}")
                continue

    # Compute grand averages for all conditions
    grand_avg_diff = {cond: {} for cond in conditions}
    for condition in conditions:
        for dev_label in all_diff_waves[condition]:
            if all_diff_waves[condition][dev_label]:
                grand_avg_diff[condition][dev_label] = mne.grand_average(all_diff_waves[condition][dev_label])

    # Get all unique deviant labels across conditions
    all_deviants = set()
    for condition in conditions:
        all_deviants.update(grand_avg_diff[condition].keys())
    sorted_deviants = sorted(all_deviants, key=lambda x: int(x.replace('ms', '')))

    # Create figure with subplots for each deviant
    n_deviants = len(sorted_deviants)
    fig, axes = plt.subplots(n_deviants, 1, figsize=(12, 3 * n_deviants), sharex=True, sharey=True)
    if n_deviants == 1:
        axes = [axes]  # Ensure axes is always a list

    # First pass to determine y-axis limits
    ymin, ymax = 0, 0
    for dev_label in sorted_deviants:
        for condition in conditions:
            if dev_label in grand_avg_diff[condition]:
                data = grand_avg_diff[condition][dev_label].get_data(picks=electrode)
                current_min, current_max = data.min(), data.max()
                ymin = min(ymin, current_min)
                ymax = max(ymax, current_max)

    # Add 10% padding to y-axis limits
    y_padding = (ymax - ymin) * 0.1
    ymin -= y_padding
    ymax += y_padding

    # Create a dictionary to store one line per condition for legend
    legend_lines = {}

    # Plot each deviant in its own subplot with all conditions overlaid
    for ax, dev_label in zip(axes, sorted_deviants):
        # [Previous plotting code remains the same]

        # Format subplot - modified axis handling
        ax.set_title(f'Deviant: {dev_label}', pad=10)
        ax.set_ylabel('Amplitude (μV)', labelpad=5)
        ax.set_ylim(ymin, ymax)

        # Remove bottom spine and all ticks
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Move x-axis to zero line
        ax.axhline(0, color='black', linewidth=0.8, zorder=0)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

        # Add x-ticks at zero line
        ax.xaxis.set_ticks_position('none')  # Hide normal x-ticks
        for spine in ['top', 'bottom']:
            ax.spines[spine].set_visible(False)

        # Create custom x-ticks at zero line
        xticks = ax.get_xticks()
        for xtick in xticks:
            ax.plot([xtick, xtick], [0, -0.05 * (ymax - ymin)],
                    color='black', linewidth=0.8, clip_on=False)
            ax.text(xtick, -0.1 * (ymax - ymin), f'{xtick:.1f}',
                    ha='center', va='top', fontsize=8)

        # Remove x-axis label (we'll add one at the bottom)
        ax.set_xlabel('')

    # Add single x-axis label at bottom
    axes[-1].set_xlabel('Time (s)', labelpad=20)

    # Create a dedicated axes for the legend
    legend_ax = fig.add_axes([0.85, 0.5, 0.1, 0.2], frame_on=False)
    legend_ax.axis('off')  # Hide the axes

    # Create legend in the dedicated axes
    legend = legend_ax.legend(legend_lines.values(), legend_lines.keys(),
                              loc='center',
                              frameon=False)

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Adjust this value as needed

    # Save figure
    output_path = Path("C:/Users/sayee/Documents/Research/PythonCode/EEG_Processor/tests/test_results/figures")
    fig.savefig(
        output_path / f"ConditionsOverlaidByDeviant_{electrode}_clean.png",
        dpi=300, bbox_inches='tight')

    plt.show(block=True)


if __name__ == "__main__":
    conditions = ["BroadBand", "NarrowBand Within-Channel", "NarrowBand Between-Channel"]
    participants = ["GINMMN1", "GINMMN2_good", '03_GINMMN', '04GINMMN', '05_GINMMN', 'GINMMN_6', "GINMMN_07", "GINMMN_08", "GINMMN_11", "GINMMN_12", 'GINMMN_13', 'GINMMN_14', 'GINMMN_15', 'GINMMN_16', 'GINMMN_17']
    base_path = Path("C:/Users/sayee/Documents/Research/PythonCode/EEG_Processor/tests/test_results/interim")

    plot_conditions_by_deviant(participants, conditions, base_path, electrode='Fz')

    for condition_idx, condition in enumerate(conditions):
        for chan in ['Fz']:
            plot_grand_average(participants, conditions, base_path, condition_idx=condition_idx, electrode=chan) # Example usage

