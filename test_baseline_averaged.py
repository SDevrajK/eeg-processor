"""
Test to verify baseline correction behavior when averaged across trials
This simulates what happens in the pipeline
"""
import numpy as np
from src.eeg_processor.processing.time_frequency import apply_single_trial_baseline

# Create synthetic data that mimics real TFR data
n_epochs = 20
n_channels = 1
n_freqs = 10  # Different frequencies (1-30 Hz)
n_times = 100

# Time vector: -0.2 to 3.0s
times = np.linspace(-0.2, 3.0, n_times)
baseline = (-0.2, 0.0)
baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
poststim_mask = times > 0

# Simulate 1/f power structure
freqs = np.logspace(np.log10(1), np.log10(30), n_freqs)
power_data = np.zeros((n_epochs, n_channels, n_freqs, n_times))

print("=== SIMULATING 1/f POWER STRUCTURE ===")
for epoch in range(n_epochs):
    for ch in range(n_channels):
        for f_idx, freq in enumerate(freqs):
            # 1/f structure: lower frequencies have higher baseline power
            baseline_power = 100 / freq  # 1/f

            # Baseline period: 1/f power + small noise
            power_data[epoch, ch, f_idx, baseline_mask] = baseline_power + np.random.normal(0, baseline_power * 0.2, baseline_mask.sum())

            # Post-stimulus: same 1/f power + event-related increase at some frequencies
            if 8 <= freq <= 12:  # Alpha band shows 50% increase
                poststim_power = baseline_power * 1.5
            else:
                poststim_power = baseline_power

            power_data[epoch, ch, f_idx, poststim_mask] = poststim_power + np.random.normal(0, poststim_power * 0.2, poststim_mask.sum())

print("\n=== BEFORE BASELINE CORRECTION ===")
print("Power at different frequencies (baseline period):")
for f_idx, freq in enumerate([freqs[0], freqs[4], freqs[-1]]):
    idx = f_idx if f_idx < 3 else (4 if f_idx == 4 else -1)
    mean_power = power_data[:, 0, idx, baseline_mask].mean()
    print(f"  {freq:.1f} Hz: {mean_power:.2f}")

# Apply single-trial baseline correction
corrected_data = apply_single_trial_baseline(power_data, times, baseline)

# Average across trials (this is what the pipeline does)
averaged_corrected = corrected_data.mean(axis=0)

print("\n=== AFTER BASELINE CORRECTION + AVERAGING ===")
print("Z-score values at different frequencies (baseline period):")
for f_idx, freq in enumerate([freqs[0], freqs[4], freqs[-1]]):
    idx = f_idx if f_idx < 3 else (4 if f_idx == 4 else -1)
    mean_zscore = averaged_corrected[0, idx, baseline_mask].mean()
    std_zscore = averaged_corrected[0, idx, baseline_mask].std()
    print(f"  {freq:.1f} Hz: mean={mean_zscore:.3f}, std={std_zscore:.3f}")

print("\nZ-score values at different frequencies (post-stimulus period):")
for f_idx, freq in enumerate([freqs[0], freqs[4], freqs[-1]]):
    idx = f_idx if f_idx < 3 else (4 if f_idx == 4 else -1)
    mean_zscore = averaged_corrected[0, idx, poststim_mask].mean()
    print(f"  {freq:.1f} Hz: mean={mean_zscore:.3f}")

# Check if 1/f structure is still visible in baseline period
baseline_profile = averaged_corrected[0, :, baseline_mask].mean(axis=1)  # Average across time
print("\n=== FREQUENCY PROFILE IN BASELINE PERIOD ===")
print("Should be flat (all near 0) if baseline correction removes 1/f:")
print(f"Values across frequencies: {baseline_profile}")
print(f"Range: {baseline_profile.min():.3f} to {baseline_profile.max():.3f}")
print(f"Std: {baseline_profile.std():.3f}")

if baseline_profile.std() > 0.5:
    print("\n⚠️  WARNING: 1/f structure still visible in baseline period!")
    print("Expected: all frequencies near 0 in baseline")
    print("This suggests baseline correction is not removing frequency structure")
else:
    print("\n✓ Baseline period is flat across frequencies (1/f removed)")
