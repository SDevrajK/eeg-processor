#!/usr/bin/env python
"""Test script for result_saver integration with complex average."""

import numpy as np
import mne
from pathlib import Path
import tempfile
import shutil
from src.eeg_processor.processing.time_frequency import compute_epochs_tfr_average
from src.eeg_processor.state_management.result_saver import ResultSaver
from loguru import logger

def test_result_saver_integration():
    """Test complete pipeline: TFR computation → result saving → loading."""

    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Using temporary directory: {temp_dir}")

    try:
        # Create simulated EEG data
        np.random.seed(42)
        sfreq = 250
        ch_names = ['Fz', 'Cz', 'Pz']
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create epochs
        n_epochs = 30
        n_times = int(1 * sfreq)
        data = np.random.randn(n_epochs, len(ch_names), n_times) * 1e-6
        epochs = mne.EpochsArray(data, info, tmin=-0.2, baseline=None)

        # Test different combinations
        test_cases = [
            {"name": "ITC + Complex Average", "compute_itc": True, "compute_complex_average": True},
            {"name": "Complex Average Only", "compute_itc": False, "compute_complex_average": True},
            {"name": "ITC Only", "compute_itc": True, "compute_complex_average": False},
            {"name": "Neither", "compute_itc": False, "compute_complex_average": False},
        ]

        result_saver = ResultSaver(temp_dir, naming_convention="bids")

        for i, test_case in enumerate(test_cases):
            logger.info(f"\n=== Test Case {i+1}: {test_case['name']} ===")

            # Compute TFR with specified parameters
            tfr = compute_epochs_tfr_average(
                epochs,
                freq_range=[8, 15],
                n_freqs=4,
                method="morlet",
                compute_itc=test_case["compute_itc"],
                compute_complex_average=test_case["compute_complex_average"],
                baseline=(-0.1, 0),
                baseline_mode="logratio"
            )

            # Check what was computed
            has_itc = hasattr(tfr, '_itc_data')
            has_complex_avg = hasattr(tfr, '_complex_average')

            logger.info(f"TFR has ITC: {has_itc}")
            logger.info(f"TFR has Complex Average: {has_complex_avg}")

            # Save using result_saver
            participant_id = f"test{i+1:02d}"
            condition_name = "testcond"
            event_type = test_case['name'].lower().replace(' ', '_')

            result_saver.save_condition(
                tfr=tfr,
                participant_id=participant_id,
                condition_name=condition_name,
                event_type=event_type,
                overwrite=True
            )

            # Verify files were created
            expected_files = []

            # Main TFR file (always created)
            tfr_filename = result_saver._get_filename(participant_id, condition_name, "tfr", "h5", event_type)
            expected_files.append(("TFR", temp_dir / 'processed' / tfr_filename))

            # ITC file (if ITC was computed)
            if test_case["compute_itc"]:
                itc_filename = result_saver._get_filename(participant_id, condition_name, "itc", "h5", event_type)
                expected_files.append(("ITC", temp_dir / 'processed' / itc_filename))

            # Complex average files (if complex average was computed)
            if test_case["compute_complex_average"]:
                complex_avg_filename = result_saver._get_filename(participant_id, condition_name, "complexavg", "npy", event_type)
                expected_files.append(("Complex Average", temp_dir / 'processed' / complex_avg_filename))

                metadata_filename = result_saver._get_filename(participant_id, condition_name, "complexavg-meta", "npy", event_type)
                expected_files.append(("Complex Average Metadata", temp_dir / 'processed' / metadata_filename))

            # Check if files exist
            for file_type, file_path in expected_files:
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    logger.info(f"✅ {file_type} saved: {file_path.name} ({file_size} bytes)")
                else:
                    logger.error(f"❌ {file_type} NOT saved: {file_path.name}")

            # Test loading complex average if it was saved
            if test_case["compute_complex_average"]:
                try:
                    complex_avg_path = temp_dir / 'processed' / complex_avg_filename
                    metadata_path = temp_dir / 'processed' / metadata_filename

                    # Load complex average and metadata
                    loaded_complex_avg = np.load(complex_avg_path)
                    loaded_metadata = np.load(metadata_path, allow_pickle=True).item()

                    logger.info(f"✅ Loaded complex average shape: {loaded_complex_avg.shape}")
                    logger.info(f"✅ Original shape: {tfr.data.shape}")
                    logger.info(f"✅ Shapes match: {loaded_complex_avg.shape == tfr.data.shape}")
                    logger.info(f"✅ Is complex: {np.iscomplexobj(loaded_complex_avg)}")

                    # Test evoked/induced decomposition
                    if np.iscomplexobj(loaded_complex_avg):
                        evoked_power = np.abs(loaded_complex_avg) ** 2
                        total_power = tfr.data  # This should be baseline-corrected power

                        # Note: For log-ratio baseline correction, we can't directly compare
                        # because the baseline correction changes the power values
                        logger.info(f"✅ Can compute evoked power from loaded data")
                        logger.info(f"   Evoked power range: {evoked_power.min():.2e} to {evoked_power.max():.2e}")
                        logger.info(f"   Total power range: {total_power.min():.2f} to {total_power.max():.2f} (dB)")

                except Exception as e:
                    logger.error(f"❌ Failed to load complex average: {str(e)}")

        logger.success("\n✅ All result_saver integration tests completed!")

    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    test_result_saver_integration()