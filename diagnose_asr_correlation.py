#!/usr/bin/env python
"""Diagnose low ASR channel correlation"""

import numpy as np
import matplotlib.pyplot as plt
from mne import pick_types
from mne.io import read_raw_fif
import sys

def diagnose_asr_correlation(raw_before, raw_after, clean_segment=None):
    """Detailed diagnosis of ASR performance"""
    
    # Get EEG channels only
    picks = pick_types(raw_before.info, meg=False, eeg=True, exclude='bads')
    ch_names = [raw_before.ch_names[i] for i in picks]
    
    # Get data
    data_before = raw_before.get_data(picks=picks)
    data_after = raw_after.get_data(picks=picks)
    
    # 1. Channel-wise correlations
    channel_corrs = []
    for i in range(len(picks)):
        corr = np.corrcoef(data_before[i], data_after[i])[0, 1]
        channel_corrs.append(corr)
    
    # 2. Power spectral density comparison
    from mne.time_frequency import psd_array_welch
    psd_before, freqs = psd_array_welch(data_before, raw_before.info['sfreq'], 
                                        fmin=1, fmax=50, n_fft=2048)
    psd_after, _ = psd_array_welch(data_after, raw_after.info['sfreq'], 
                                   fmin=1, fmax=50, n_fft=2048)
    
    # 3. Variance analysis
    var_before = np.var(data_before, axis=1)
    var_after = np.var(data_after, axis=1)
    var_ratio = var_after / var_before
    
    # 4. Plot diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Channel correlations
    ax = axes[0, 0]
    ax.bar(range(len(channel_corrs)), channel_corrs)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Low correlation threshold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Channel-wise Correlations (mean={np.mean(channel_corrs):.3f})')
    ax.legend()
    
    # Correlation histogram
    ax = axes[0, 1]
    ax.hist(channel_corrs, bins=20, edgecolor='black')
    ax.axvline(x=np.mean(channel_corrs), color='r', linestyle='--', label=f'Mean={np.mean(channel_corrs):.3f}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Channel Correlations')
    ax.legend()
    
    # Worst channels
    ax = axes[0, 2]
    worst_idx = np.argsort(channel_corrs)[:10]
    worst_corrs = [channel_corrs[i] for i in worst_idx]
    worst_names = [ch_names[i] for i in worst_idx]
    ax.barh(range(len(worst_corrs)), worst_corrs)
    ax.set_yticks(range(len(worst_corrs)))
    ax.set_yticklabels(worst_names)
    ax.set_xlabel('Correlation')
    ax.set_title('10 Worst Correlation Channels')
    
    # Power spectrum comparison
    ax = axes[1, 0]
    mean_psd_before = np.mean(psd_before, axis=0)
    mean_psd_after = np.mean(psd_after, axis=0)
    ax.semilogy(freqs, mean_psd_before, label='Before ASR', alpha=0.7)
    ax.semilogy(freqs, mean_psd_after, label='After ASR', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.set_title('Average Power Spectral Density')
    ax.legend()
    
    # Variance ratio
    ax = axes[1, 1]
    ax.bar(range(len(var_ratio)), var_ratio)
    ax.axhline(y=1.0, color='r', linestyle='--')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Variance Ratio (After/Before)')
    ax.set_title(f'Variance Change (mean ratio={np.mean(var_ratio):.3f})')
    
    # Time series comparison for worst channel
    ax = axes[1, 2]
    worst_ch_idx = worst_idx[0]
    time_window = slice(0, int(10 * raw_before.info['sfreq']))  # First 10 seconds
    times = raw_before.times[time_window]
    ax.plot(times, data_before[worst_ch_idx, time_window] * 1e6, 
            label=f'Before ({ch_names[worst_ch_idx]})', alpha=0.7)
    ax.plot(times, data_after[worst_ch_idx, time_window] * 1e6, 
            label='After ASR', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Worst Channel: {ch_names[worst_ch_idx]} (corr={channel_corrs[worst_ch_idx]:.3f})')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n=== ASR Diagnosis Summary ===")
    print(f"Mean channel correlation: {np.mean(channel_corrs):.3f}")
    print(f"Channels with correlation < 0.5: {sum(c < 0.5 for c in channel_corrs)}/{len(channel_corrs)}")
    print(f"Channels with correlation < 0.3: {sum(c < 0.3 for c in channel_corrs)}/{len(channel_corrs)}")
    print(f"Channels with correlation < 0.1: {sum(c < 0.1 for c in channel_corrs)}/{len(channel_corrs)}")
    print(f"\nMean variance ratio: {np.mean(var_ratio):.3f}")
    print(f"Channels with >50% variance reduction: {sum(v < 0.5 for v in var_ratio)}")
    print(f"Channels with >50% variance increase: {sum(v > 1.5 for v in var_ratio)}")
    
    # Analyze clean segment if provided
    if clean_segment is not None:
        print("\n=== Clean Segment Analysis ===")
        clean_picks = pick_types(clean_segment.info, meg=False, eeg=True, exclude='bads')
        clean_data = clean_segment.get_data(picks=clean_picks)
        clean_var = np.var(clean_data, axis=1)
        print(f"Clean segment duration: {clean_segment.times[-1]:.1f}s")
        print(f"Clean segment mean variance: {np.mean(clean_var)*1e12:.2f} µV²")
        print(f"Original data mean variance: {np.mean(var_before)*1e12:.2f} µV²")
        print(f"Variance ratio (original/clean): {np.mean(var_before)/np.mean(clean_var):.2f}")
    
    return channel_corrs, var_ratio

# Usage example
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose_asr_correlation.py <before_asr.fif> <after_asr.fif> [clean_segment.fif]")
        sys.exit(1)
    
    # Load data
    raw_before = read_raw_fif(sys.argv[1], preload=True)
    raw_after = read_raw_fif(sys.argv[2], preload=True)
    
    clean_segment = None
    if len(sys.argv) > 3:
        clean_segment = read_raw_fif(sys.argv[3], preload=True)
    
    # Run diagnosis
    diagnose_asr_correlation(raw_before, raw_after, clean_segment)