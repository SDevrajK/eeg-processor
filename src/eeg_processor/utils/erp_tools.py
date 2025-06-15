import mne
import numpy as np


def find_erp_peaks(evoked, electrode='Fz',
                   mmn_window=(0.150, 0.250),
                   p2_window=(0.1, 0.150),
                   dynamic_p2_window=False,
                   min_p2_mmn_separation=0.025):
    """
    Detect MMN and P2 peaks with optional dynamic P2 window adjustment.

    Parameters:
        evoked: MNE Evoked object
        electrode: Channel to analyze
        mmn_window: Time window for MMN detection (s)
        p2_window: Initial time window for P2 detection (s)
        dynamic_p2_window: If True, adjusts P2 window based on MMN latency
        min_p2_mmn_separation: Minimum required time between P2 and MMN (s)
    """
    picks = mne.pick_channels(evoked.ch_names, include=[electrode])
    data = evoked.data[picks[0]]
    times = evoked.times

    # Find MMN (most negative peak in window)
    mmn_mask = (times >= mmn_window[0]) & (times <= mmn_window[1])
    mmn_data = data[mmn_mask]
    mmn_times = times[mmn_mask]
    mmn_idx = np.argmin(mmn_data)
    mmn_latency = mmn_times[mmn_idx]

    # Adjust P2 window dynamically if requested
    if dynamic_p2_window and mmn_latency is not None:
        # Ensure P2 ends at least min_p2_mmn_separation before MMN starts
        new_p2_end = mmn_latency - min_p2_mmn_separation
        if new_p2_end > p2_window[0]:  # Only adjust if valid
            p2_window = (p2_window[0]+(new_p2_end-p2_window[1]), new_p2_end)
            print(f"Adjusted P2 window to: {p2_window}")

    # Find P2 (most positive peak in window)
    p2_mask = (times >= p2_window[0]) & (times <= p2_window[1])
    p2_data = data[p2_mask]
    p2_times = times[p2_mask]

    p2_latency = p2_amp = None
    if len(p2_data) > 0:
        p2_idx = np.argmax(p2_data)
        p2_latency = p2_times[p2_idx]
        # Verify P2 occurs before MMN with minimum separation
        if mmn_latency is not None:
            if p2_latency >= (mmn_latency - min_p2_mmn_separation):
                p2_latency = p2_amp = None
                print("P2 too close to MMN - discarding")

    # Calculate mean amplitude in ±25ms windows
    def get_mean_amp(latency):
        if latency is None:
            return None
        win_mask = (times >= latency - 0.025) & (times <= latency + 0.025)
        return np.mean(data[win_mask])

    return {
        'mmn_latency': mmn_latency,
        'mmn_amplitude': get_mean_amp(mmn_latency),
        'p2_latency': p2_latency,
        'p2_amplitude': get_mean_amp(p2_latency),
        'adjusted_p2_window': p2_window if dynamic_p2_window else None
    }