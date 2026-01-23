"""
Quick test to verify apply_single_trial_baseline is working correctly
"""
import numpy as np
from src.eeg_processor.processing.time_frequency import apply_single_trial_baseline

# Create synthetic data
n_epochs = 10
n_channels = 2
n_freqs = 5
n_times = 100

# Time vector: -0.2 to 0.8s (baseline: -0.2 to 0)
times = np.linspace(-0.2, 0.8, n_times)
baseline = (-0.2, 0.0)

# Create synthetic power data with known properties
# Baseline: mean=10, std=2
# Post-stimulus: mean=20, std=2 (strong increase)
power_data = np.zeros((n_epochs, n_channels, n_freqs, n_times))

baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
poststim_mask = times > 0

# Fill with synthetic data
for epoch in range(n_epochs):
    for ch in range(n_channels):
        for freq in range(n_freqs):
            # Baseline: mean=10, std=2
            power_data[epoch, ch, freq, baseline_mask] = np.random.normal(10, 2, baseline_mask.sum())
            # Post-stimulus: mean=20, std=2 (2x increase)
            power_data[epoch, ch, freq, poststim_mask] = np.random.normal(20, 2, poststim_mask.sum())

print("=== BEFORE BASELINE CORRECTION ===")
print(f"Baseline period mean: {power_data[..., baseline_mask].mean():.2f} (expected: 10)")
print(f"Baseline period std: {power_data[..., baseline_mask].std():.2f} (expected: 2)")
print(f"Post-stimulus mean: {power_data[..., poststim_mask].mean():.2f} (expected: 20)")
print(f"Post-stimulus std: {power_data[..., poststim_mask].std():.2f} (expected: 2)")

# Apply baseline correction
corrected_data = apply_single_trial_baseline(power_data, times, baseline)

print("\n=== AFTER BASELINE CORRECTION (Z-SCORE) ===")
baseline_corrected = corrected_data[..., baseline_mask]
poststim_corrected = corrected_data[..., poststim_mask]

print(f"Baseline period mean: {baseline_corrected.mean():.2f} (expected: ~0)")
print(f"Baseline period std: {baseline_corrected.std():.2f} (expected: ~1)")
print(f"Post-stimulus mean: {poststim_corrected.mean():.2f} (expected: ~5)")
print(f"Post-stimulus std: {poststim_corrected.std():.2f}")

# Explanation
print("\n=== INTERPRETATION ===")
print("Z-score normalization: (power - baseline_mean) / baseline_std")
print(f"Expected post-stim z-score: (20 - 10) / 2 = 5.0")
print(f"Actual post-stim z-score: {poststim_corrected.mean():.2f}")
print("\nNote: High z-scores (>3) in post-stimulus are EXPECTED and CORRECT")
print("They indicate strong power increase relative to baseline variability")
print("\n✓ If baseline mean ≈ 0 and std ≈ 1: correction is working correctly")
print("✓ If post-stim values are high (5-10): this is NORMAL for strong ERD/ERS")
