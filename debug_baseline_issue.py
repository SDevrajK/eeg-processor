"""
Debug script to understand the baseline correction issue
This simulates the exact scenario from the user's data
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for WSL
import matplotlib.pyplot as plt
from src.eeg_processor.processing.time_frequency import apply_single_trial_baseline

# Match user's parameters
n_epochs = 25
n_channels = 1
n_freqs = 49
tmin, tmax = -1.0, 3.0
sfreq = 500  # Assume 500 Hz sampling
decim = 5
effective_sfreq = sfreq / decim

# Create time vector matching user's epochs
n_samples = int((tmax - tmin) * effective_sfreq)
times = np.linspace(tmin, tmax, n_samples)
baseline = (-0.2, 0.0)

print(f"=== USER'S CONFIGURATION ===")
print(f"Epoch window: {tmin} to {tmax} s")
print(f"Baseline window: {baseline}")
print(f"Decimation: {decim}")
print(f"Effective sampling rate: {effective_sfreq} Hz")
print(f"Number of time points: {len(times)}")
print(f"Time resolution: {1/effective_sfreq:.4f} s")

# Check baseline coverage
baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
n_baseline_pts = baseline_mask.sum()
print(f"\nBaseline coverage: {n_baseline_pts} samples from {times[baseline_mask][0]:.3f}s to {times[baseline_mask][-1]:.3f}s")

if n_baseline_pts < 5:
    print(f"⚠️  WARNING: Only {n_baseline_pts} baseline samples! This is too few for reliable statistics.")
    print("With decimation=5 and baseline=[-0.2, 0], you may have insufficient baseline data.")

# Create realistic TFR data with 1/f structure
freqs = np.logspace(np.log10(2), np.log10(50), n_freqs)
power_data = np.zeros((n_epochs, n_channels, n_freqs, len(times)))

# Simulate realistic power with 1/f + event-related changes
for epoch in range(n_epochs):
    for ch in range(n_channels):
        for f_idx, freq in enumerate(freqs):
            # 1/f baseline power
            baseline_power = 100 / freq

            # Add noise
            noise_std = baseline_power * 0.3

            # Create time-varying power
            for t_idx, t in enumerate(times):
                # Baseline period: just 1/f + noise
                if t < 0:
                    power_data[epoch, ch, f_idx, t_idx] = baseline_power + np.random.normal(0, noise_std)
                # Post-stimulus: add event-related increase in alpha/theta
                else:
                    if 4 <= freq <= 8:  # Theta increase
                        power_data[epoch, ch, f_idx, t_idx] = baseline_power * 1.5 + np.random.normal(0, noise_std)
                    elif 8 <= freq <= 13:  # Alpha decrease (ERD)
                        power_data[epoch, ch, f_idx, t_idx] = baseline_power * 0.7 + np.random.normal(0, noise_std)
                    else:
                        power_data[epoch, ch, f_idx, t_idx] = baseline_power + np.random.normal(0, noise_std)

print("\n=== BEFORE BASELINE CORRECTION ===")
baseline_data = power_data[:, 0, :, baseline_mask]
print(f"Baseline period shape: {baseline_data.shape}")
print(f"Low freq (2 Hz) baseline: mean={power_data[:, 0, 0, baseline_mask].mean():.2f}")
print(f"High freq (50 Hz) baseline: mean={power_data[:, 0, -1, baseline_mask].mean():.2f}")
print(f"1/f gradient visible: {power_data[:, 0, 0, baseline_mask].mean() / power_data[:, 0, -1, baseline_mask].mean():.1f}x")

# Apply baseline correction
try:
    corrected_data = apply_single_trial_baseline(power_data, times, baseline)

    # Average across trials
    averaged_corrected = corrected_data.mean(axis=0)

    print("\n=== AFTER BASELINE CORRECTION + AVERAGING ===")
    baseline_corrected = averaged_corrected[0, :, baseline_mask]
    print(f"Baseline period (corrected):")
    print(f"  Low freq (2 Hz): mean={averaged_corrected[0, 0, baseline_mask].mean():.3f}")
    print(f"  High freq (50 Hz): mean={averaged_corrected[0, -1, baseline_mask].mean():.3f}")
    print(f"  Across all freqs: mean={baseline_corrected.mean():.3f}, std={baseline_corrected.std():.3f}")

    # Check if 1/f structure persists
    baseline_freq_profile = averaged_corrected[0, :, baseline_mask].mean(axis=1)
    print(f"\n1/f gradient in baseline (should be flat):")
    print(f"  Range: {baseline_freq_profile.min():.3f} to {baseline_freq_profile.max():.3f}")
    print(f"  Std across frequencies: {baseline_freq_profile.std():.3f}")

    if baseline_freq_profile.std() > 0.3:
        print("\n❌ PROBLEM: 1/f structure still visible in baseline!")
        print("This should not happen if baseline correction is working correctly.")
    else:
        print("\n✓ Baseline is flat across frequencies (1/f removed)")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Before correction
    im0 = axes[0].imshow(power_data.mean(axis=0)[0, :, :],
                         aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs[0], freqs[-1]],
                         cmap='RdBu_r')
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].axvline(baseline[0], color='w', linestyle='--', alpha=0.7, label='Baseline window')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Before Baseline Correction\n(1/f visible)')
    axes[0].legend()
    plt.colorbar(im0, ax=axes[0])

    # After correction
    vmax = max(abs(averaged_corrected[0].min()), abs(averaged_corrected[0].max()))
    im1 = axes[1].imshow(averaged_corrected[0],
                         aspect='auto', origin='lower',
                         extent=[times[0], times[-1], freqs[0], freqs[-1]],
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].axvline(baseline[0], color='w', linestyle='--', alpha=0.7)
    axes[1].axvline(baseline[1], color='w', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('After Baseline Correction\n(should be flat in baseline)')
    plt.colorbar(im1, ax=axes[1])

    # Frequency profile in baseline period
    axes[2].plot(freqs, baseline_freq_profile, 'o-')
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Z-score')
    axes[2].set_title('Baseline Period Frequency Profile\n(should be flat at 0)')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_correction_debug.png', dpi=150)
    print("\n📊 Plot saved to: baseline_correction_debug.png")

except Exception as e:
    print(f"\n❌ ERROR during baseline correction: {e}")
    import traceback
    traceback.print_exc()
