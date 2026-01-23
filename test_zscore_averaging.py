"""
Test to understand why 1/f structure appears in averaged z-score data
"""
import numpy as np
from src.eeg_processor.processing.time_frequency import apply_single_trial_baseline

# Simulate realistic scenario
n_epochs = 25
n_freqs = 10
n_times = 100

times = np.linspace(-0.2, 3.0, n_times)
baseline = (-0.2, 0.0)
baseline_mask = (times >= baseline[0]) & (times <= baseline[1])

# Create data with 1/f structure
freqs = np.logspace(np.log10(2), np.log10(50), n_freqs)
power_data = np.zeros((n_epochs, 1, n_freqs, n_times))

for epoch in range(n_epochs):
    for f_idx, freq in enumerate(freqs):
        # 1/f baseline power
        baseline_power = 100 / freq

        # All time points get baseline power + noise
        power_data[epoch, 0, f_idx, :] = baseline_power + np.random.normal(0, baseline_power * 0.3, n_times)

print("=== BEFORE BASELINE CORRECTION ===")
print("Baseline period (should show 1/f gradient):")
for i in [0, -1]:
    print(f"  {freqs[i]:.1f} Hz: {power_data[:, 0, i, baseline_mask].mean():.2f}")

# Apply z-score baseline correction
corrected = apply_single_trial_baseline(power_data, times, baseline)

# Average across trials
averaged = corrected.mean(axis=0)[0]  # Shape: (n_freqs, n_times)

print("\n=== AFTER Z-SCORE + AVERAGING ===")
print("Baseline period (should be flat at ~0):")
baseline_avg = averaged[:, baseline_mask].mean(axis=1)
for i in [0, -1]:
    print(f"  {freqs[i]:.1f} Hz: {baseline_avg[i]:.3f}")

print(f"\nBaseline frequency profile std: {baseline_avg.std():.3f}")
print(f"Should be < 0.3 for flat profile")

# Check post-baseline period
post_baseline_mask = times > 0.5
post_avg = averaged[:, post_baseline_mask].mean(axis=1)
print(f"\nPost-baseline period:")
for i in [0, -1]:
    print(f"  {freqs[i]:.1f} Hz: {post_avg[i]:.3f}")

print(f"\nPost-baseline frequency profile std: {post_avg.std():.3f}")

if baseline_avg.std() < 0.3:
    print("\n✓ Baseline is flat (1/f removed)")
else:
    print(f"\n❌ Baseline still shows frequency structure! std={baseline_avg.std():.3f}")

if post_avg.std() > 0.3:
    print("⚠️  Post-baseline shows frequency structure")
    print("This is EXPECTED if there's no real event-related change")
    print("With only noise and no true stimulus effect, averaging should give ~0 everywhere")
