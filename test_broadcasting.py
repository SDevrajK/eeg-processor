"""
Test to verify that baseline correction is applied to ALL time points
"""
import numpy as np

# Simulate the exact shapes
n_epochs = 5
n_channels = 2
n_freqs = 3
n_times = 10

# Create power data with known 1/f structure
power_data = np.zeros((n_epochs, n_channels, n_freqs, n_times))

# Fill with simple pattern: low freq has high power everywhere
# Add small noise to avoid zero std
for f in range(n_freqs):
    base_power = 10.0 / (f + 1)  # 1/f pattern: 10, 5, 3.33
    power_data[:, :, f, :] = base_power + np.random.normal(0, base_power * 0.01, (n_epochs, n_channels, n_times))

print("=== BEFORE BASELINE CORRECTION ===")
print(f"Shape: {power_data.shape}")
print(f"Freq 0 (high power): {power_data[0, 0, 0, :3]} (first 3 timepoints)")
print(f"Freq 2 (low power): {power_data[0, 0, 2, :3]} (first 3 timepoints)")

# Simulate baseline correction
times = np.linspace(-0.2, 0.8, n_times)
baseline = (-0.2, 0.0)
baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
baseline_indices = np.where(baseline_mask)[0]

print(f"\nBaseline indices: {baseline_indices}")
print(f"Baseline time points: {times[baseline_indices]}")

# Extract baseline
baseline_power = power_data[..., baseline_indices]
print(f"\nBaseline power shape: {baseline_power.shape}")

# Compute stats
baseline_mean = np.mean(baseline_power, axis=-1, keepdims=True)
baseline_std = np.std(baseline_power, axis=-1, keepdims=True)

print(f"Baseline mean shape: {baseline_mean.shape}")
print(f"Baseline std shape: {baseline_std.shape}")

print(f"\nBaseline mean values (epoch 0, ch 0):")
for f in range(n_freqs):
    print(f"  Freq {f}: mean={baseline_mean[0, 0, f, 0]:.2f}, std={baseline_std[0, 0, f, 0]:.2f}")

# Apply correction
corrected_data = (power_data - baseline_mean) / baseline_std

print("\n=== AFTER BASELINE CORRECTION ===")
print(f"Corrected shape: {corrected_data.shape}")

# Check baseline period (should be ~0 with std ~1)
baseline_corrected = corrected_data[..., baseline_indices]
print(f"\nBaseline period (should be ~0):")
for f in range(n_freqs):
    mean_val = baseline_corrected[:, :, f, :].mean()
    print(f"  Freq {f}: {mean_val:.6f}")

# Check post-baseline period
post_baseline_mask = times > 0
post_indices = np.where(post_baseline_mask)[0]
post_corrected = corrected_data[..., post_indices]

print(f"\nPost-baseline period:")
print(f"  Time points: {times[post_indices]}")
for f in range(n_freqs):
    mean_val = post_corrected[:, :, f, :].mean()
    print(f"  Freq {f}: {mean_val:.6f}")

# The key test: since all time points have the same value (constant power),
# z-score should be 0 everywhere after averaging
print("\n=== KEY TEST ===")
print("Since power is constant across time (no event-related change),")
print("z-score should be ~0 everywhere after baseline correction.")
print("\nActual values (epoch 0, ch 0, all time points):")
for f in range(n_freqs):
    print(f"  Freq {f}: {corrected_data[0, 0, f, :]}")

if np.allclose(corrected_data, 0, atol=1e-10):
    print("\n✓ Baseline correction applied to ALL time points correctly!")
else:
    print(f"\n❌ Problem: corrected data not all zero!")
    print(f"Max abs value: {np.abs(corrected_data).max()}")
