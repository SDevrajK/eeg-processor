"""
Script to inspect actual TFR data values to verify baseline correction
"""
import numpy as np
import mne

# You need to provide the path to your actual TFR file
# Example: tfr_path = "/mnt/c/Users/sayee/.../some_tfr_file.h5"

print("=== TFR DATA INSPECTOR ===")
print("\nTo use this script, edit it and set tfr_path to your actual TFR file")
print("Then we can check if baseline correction was applied correctly")
print("\nExpected:")
print("  - Baseline period (-0.2 to 0s) should have mean ≈ 0 across all frequencies")
print("  - No 1/f gradient in baseline period")
print("  - Post-stimulus values can be any z-score (positive or negative)")

# Uncomment and set your file path:
# tfr_path = "/mnt/c/Users/sayee/Documents/Research/HébertLab/.../your_tfr_file.h5"
# tfr = mne.time_frequency.read_tfrs(tfr_path)[0]
#
# baseline_mask = (tfr.times >= -0.2) & (tfr.times <= 0)
#
# print(f"\nTFR Data Shape: {tfr.data.shape}")
# print(f"  Channels: {len(tfr.ch_names)}")
# print(f"  Frequencies: {len(tfr.freqs)} ({tfr.freqs[0]:.1f} - {tfr.freqs[-1]:.1f} Hz)")
# print(f"  Time points: {len(tfr.times)} ({tfr.times[0]:.2f} - {tfr.times[-1]:.2f} s)")
#
# print(f"\nBaseline period ({baseline_mask.sum()} time points):")
# baseline_data = tfr.data[:, :, baseline_mask]
# print(f"  Mean across all channels/freqs/times: {baseline_data.mean():.4f}")
# print(f"  Std: {baseline_data.std():.4f}")
#
# # Check each frequency
# print(f"\nPer-frequency analysis in baseline period:")
# for i in [0, len(tfr.freqs)//2, -1]:
#     freq = tfr.freqs[i]
#     freq_data = tfr.data[:, i, baseline_mask]
#     print(f"  {freq:.1f} Hz: mean={freq_data.mean():.4f}, std={freq_data.std():.4f}")
#
# # Check if 1/f gradient persists
# freq_means = np.mean(tfr.data[:, :, baseline_mask], axis=(0, 2))  # Average across channels and time
# print(f"\nFrequency profile in baseline (should be flat around 0):")
# print(f"  Low freq mean: {freq_means[0]:.4f}")
# print(f"  Mid freq mean: {freq_means[len(freq_means)//2]:.4f}")
# print(f"  High freq mean: {freq_means[-1]:.4f}")
# print(f"  Std across frequencies: {freq_means.std():.4f}")
#
# if freq_means.std() > 0.5:
#     print("\n❌ PROBLEM: 1/f gradient still visible!")
#     print("Baseline correction may not have been applied.")
# elif abs(freq_means.mean()) > 0.5:
#     print("\n❌ PROBLEM: Baseline not centered at 0!")
#     print("Baseline correction may not have been applied.")
# else:
#     print("\n✓ Data appears to be baseline-corrected")
