import matplotlib.pyplot as plt
import mne
import numpy as np
from utils.erp_tools import find_erp_peaks

if __name__ == "__main__":
    conditions = ["BroadBand", "NarrowBand Within-Channel", "NarrowBand Between-Channel"]
    participants = ["GINMMN1", "GINMMN2_good", '03_GINMMN', '04GINMMN', '05_GINMMN', 'GINMMN_6', "GINMMN_07.vhdr", "GINMMN_08.vhdr", "GINMMN_11.vhdr", "GINMMN_12.vhdr", 'GINMMN_13.vhdr', 'GINMMN_14.vhdr', 'GINMMN_15.vhdr', 'GINMMN_16.vhdr', 'GINMMN_17.vhdr']

    sub = 0
    cond = 0

    # Load data
    stan = mne.read_epochs(
        f'C://Users//sayee//Documents//Research//PythonCode//EEG_Processor//tests/test_results/interim/{participants[sub]}_{conditions[cond]}_Standards-epo.fif')
    dev = mne.read_epochs(
        f'C://Users//sayee//Documents//Research//PythonCode//EEG_Processor//tests/test_results/interim/{participants[sub]}_{conditions[cond]}_Deviants-epo.fif')
    stan_evoked = stan.average()

    # Process deviants
    diff_waves = {}
    peak_data = {}
    for dev_label in sorted(dev.metadata['event_label'].unique(),
                            key=lambda x: int(x.replace('ms', ''))):
        dev_evoked = dev[dev_label].average()
        diff_waves[dev_label] = mne.combine_evoked([dev_evoked, stan_evoked], [1, -1])
        peak_data[dev_label] = find_erp_peaks(diff_waves[dev_label],mmn_window=(0.175, 0.275),p2_window=(0.125, 0.175), dynamic_p2_window=True)

    # Define the 7-color sequential blue palette
    blue_palette = [
        '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5',
        '#084594'
    ]

    # Create figure with optimized layout
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, width_ratios=[1.5, 1.5, 0.9, 0.1], height_ratios=[1.6, 1.6, 0.7, 0.7])  # Adjust height ratios

    # Assign subplots
    ax_erp = fig.add_subplot(gs[0:2, :])  # ERP takes full width at top
    ax_mmn = fig.add_subplot(gs[2:4, 0])  # MMN bar bottom left
    ax_diff = fig.add_subplot(gs[2:4, 1])  # Difference bar bottom left
    ax_topomap = fig.add_subplot(gs[2:4, 2])  # Topography takes right side
    ax_topobar = fig.add_subplot(gs[2:4, 3])  # Topography takes right side

    # 1. Plot ERP with markers
    mne.viz.plot_compare_evokeds(
        diff_waves,
        picks='Fz',
        title='MMN/P2 Detection at Fz (±25ms mean amplitude)',
        combine=None,
        colors=blue_palette,
        axes=ax_erp,
        show=False
    )

    # Add markers to ERP plot
    deviant_labels = sorted(peak_data.keys(), key=lambda x: int(x.replace('ms', '')))
    for idx, label in enumerate(deviant_labels):
        peaks = peak_data[label]
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

    # 2. Prepare data for bar plots
    durations = [int(label.replace('ms', '')) for label in deviant_labels]
    mmn_amps = [peak_data[label]['mmn_amplitude'] for label in deviant_labels]
    p2_mmn_diffs = []

    for label in deviant_labels:
        if peak_data[label]['p2_amplitude'] is not None and peak_data[label]['mmn_amplitude'] is not None:
            p2_mmn_diffs.append(peak_data[label]['p2_amplitude'] - peak_data[label]['mmn_amplitude'])
        else:
            p2_mmn_diffs.append(None)

    # 3. Plot MMN amplitudes with enhanced formatting
    bars_mmn = ax_mmn.bar(durations, mmn_amps, color=blue_palette[:len(deviant_labels)])  # Fixed width
    ax_mmn.set_title('MMN Amplitude', pad=10)
    ax_mmn.set_ylabel('μV', labelpad=5)
    ax_mmn.set_xticks(durations)
    ax_mmn.set_xticklabels(durations, rotation=45)
    ax_mmn.axhline(0, color='black', linewidth=0.8)

    # 4. Plot P2-MMN differences with matching style
    bars_diff = ax_diff.bar(durations, p2_mmn_diffs, color=blue_palette[:len(deviant_labels)])
    ax_diff.set_title('P2-MMN Difference', pad=10)
    ax_diff.set_xlabel('Deviant Duration (ms)', labelpad=10)
    ax_diff.set_ylabel('μV', labelpad=5)
    ax_diff.set_xticks(durations)
    ax_diff.set_xticklabels(durations, rotation=45)
    ax_diff.axhline(0, color='black', linewidth=0.8)

    # 5. Enhanced Topography Plot
    if any(peak_data[label]['mmn_latency'] for label in deviant_labels):
        # Find most robust MMN (largest amplitude)
        robust_dev = max(deviant_labels,
                         key=lambda x: abs(peak_data[x]['mmn_amplitude']) if peak_data[x]['mmn_amplitude'] else 0)
        mmn_lat = peak_data[robust_dev]['mmn_latency']

        grand_avg = mne.grand_average(list(diff_waves.values()))

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

    plt.show(block=True)
