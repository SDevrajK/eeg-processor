#!/usr/bin/env python
"""Investigate potential ASR issues and suggest solutions"""

import numpy as np
from mne import pick_types
from mne.io import BaseRaw
import matplotlib.pyplot as plt

def investigate_asr_issues(raw_original, raw_cleaned, clean_segment=None, cutoff=20):
    """Investigate why ASR correlation is low and suggest solutions"""
    
    # Get EEG channels
    picks = pick_types(raw_original.info, meg=False, eeg=True, exclude='bads')
    
    # Get data
    data_orig = raw_original.get_data(picks=picks)
    data_clean = raw_cleaned.get_data(picks=picks)
    
    # Calculate correlations
    correlations = []
    for i in range(len(picks)):
        corr = np.corrcoef(data_orig[i], data_clean[i])[0, 1]
        correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    
    print("=" * 60)
    print("ASR CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"Mean correlation: {mean_corr:.3f}")
    
    # 1. Check if the issue is global or channel-specific
    low_corr_channels = sum(c < 0.3 for c in correlations)
    if low_corr_channels > len(correlations) * 0.5:
        print("\n⚠️  ISSUE: More than 50% of channels have low correlation")
        print("   This suggests global over-correction")
    else:
        print(f"\n✓  Only {low_corr_channels}/{len(correlations)} channels have very low correlation")
        
    # 2. Check variance changes
    var_orig = np.var(data_orig, axis=1)
    var_clean = np.var(data_clean, axis=1)
    var_ratio = var_clean / var_orig
    mean_var_ratio = np.mean(var_ratio)
    
    print(f"\nMean variance ratio (cleaned/original): {mean_var_ratio:.3f}")
    if mean_var_ratio < 0.5:
        print("⚠️  ISSUE: Variance reduced by more than 50%")
        print("   ASR may be removing too much signal")
    
    # 3. Check clean segment quality if provided
    if clean_segment is not None:
        clean_picks = pick_types(clean_segment.info, meg=False, eeg=True, exclude='bads')
        clean_data = clean_segment.get_data(picks=clean_picks)
        clean_var = np.var(clean_data, axis=1)
        
        # Compare clean segment to original data
        orig_subset = data_orig[:, :clean_data.shape[1]]
        clean_orig_ratio = np.mean(clean_var) / np.mean(np.var(orig_subset, axis=1))
        
        print(f"\n📊 CLEAN SEGMENT ANALYSIS:")
        print(f"   Duration: {clean_segment.times[-1]:.1f}s")
        print(f"   Variance ratio (clean/original): {clean_orig_ratio:.3f}")
        
        if clean_orig_ratio > 0.8:
            print("   ⚠️  Clean segment may not be much cleaner than original data")
            print("      Consider selecting a cleaner segment")
    
    # 4. Frequency-specific analysis
    from mne.time_frequency import psd_array_welch
    psd_orig, freqs = psd_array_welch(data_orig, raw_original.info['sfreq'], 
                                      fmin=1, fmax=50, n_fft=2048)
    psd_clean, _ = psd_array_welch(data_clean, raw_cleaned.info['sfreq'], 
                                   fmin=1, fmax=50, n_fft=2048)
    
    # Check power reduction in different bands
    bands = {'Delta (1-4 Hz)': (1, 4),
             'Theta (4-8 Hz)': (4, 8), 
             'Alpha (8-13 Hz)': (8, 13),
             'Beta (13-30 Hz)': (13, 30),
             'Gamma (30-50 Hz)': (30, 50)}
    
    print("\n📊 FREQUENCY BAND ANALYSIS:")
    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        power_orig = np.mean(psd_orig[:, freq_mask])
        power_clean = np.mean(psd_clean[:, freq_mask])
        reduction = (1 - power_clean/power_orig) * 100
        print(f"   {band_name}: {reduction:.1f}% power reduction")
        
    # RECOMMENDATIONS
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    if mean_corr < 0.3:
        recommendations.append("1. Increase cutoff parameter (try 25-30 instead of 20)")
        recommendations.append("2. Check if clean segment is truly clean of artifacts")
        recommendations.append("3. Ensure clean segment is representative of the data")
        
    if low_corr_channels > 10:
        recommendations.append("4. Check for bad channels before ASR")
        recommendations.append("5. Consider running bad channel detection first")
        
    if mean_var_ratio < 0.5:
        recommendations.append("6. Data may have extreme artifacts - inspect visually")
        recommendations.append("7. Consider artifact rejection instead of correction")
        
    for i, rec in enumerate(recommendations, 1):
        print(f"{rec}")
        
    # Generate diagnostic plots
    print("\n📊 Generating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Correlation by channel
    ax = axes[0, 0]
    ax.bar(range(len(correlations)), correlations)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=mean_corr, color='g', linestyle='-', label=f'Mean={mean_corr:.3f}')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Correlation')
    ax.set_title('Channel-wise Correlation (Before vs After ASR)')
    ax.legend()
    
    # 2. Example time series
    ax = axes[0, 1]
    ch_idx = np.argmin(correlations)  # Worst channel
    time_slice = slice(int(10*raw_original.info['sfreq']), int(20*raw_original.info['sfreq']))
    times = np.arange(time_slice.stop - time_slice.start) / raw_original.info['sfreq']
    
    ax.plot(times, data_orig[ch_idx, time_slice] * 1e6, label='Original', alpha=0.7)
    ax.plot(times, data_clean[ch_idx, time_slice] * 1e6, label='After ASR', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'Worst Channel (corr={correlations[ch_idx]:.3f})')
    ax.legend()
    
    # 3. Power spectrum
    ax = axes[1, 0]
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), label='Original', alpha=0.7)
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), label='After ASR', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.set_title('Average Power Spectral Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Variance changes
    ax = axes[1, 1]
    ax.scatter(var_orig * 1e12, var_clean * 1e12, alpha=0.6)
    ax.plot([0, np.max(var_orig)*1e12], [0, np.max(var_orig)*1e12], 'r--', alpha=0.5)
    ax.set_xlabel('Original Variance (µV²)')
    ax.set_ylabel('Cleaned Variance (µV²)')
    ax.set_title('Variance Comparison by Channel')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return correlations, recommendations

# Example usage
if __name__ == "__main__":
    import sys
    from mne.io import read_raw_fif
    
    if len(sys.argv) < 3:
        print("Usage: python investigate_asr_issues.py <before_asr.fif> <after_asr.fif> [clean_segment.fif]")
        sys.exit(1)
    
    raw_before = read_raw_fif(sys.argv[1], preload=True)
    raw_after = read_raw_fif(sys.argv[2], preload=True)
    
    clean_segment = None
    if len(sys.argv) > 3:
        clean_segment = read_raw_fif(sys.argv[3], preload=True)
    
    investigate_asr_issues(raw_before, raw_after, clean_segment)