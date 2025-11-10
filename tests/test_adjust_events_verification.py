"""
Verification tests to confirm:
1. Shift properly accounts for sampling rate
2. Annotations are in seconds (not samples)
3. shift_ms: 400 actually shifts forward by 400ms
"""
import numpy as np
import pytest
from mne.io import RawArray
from mne import create_info, Annotations

from src.eeg_processor.utils.raw_data_tools import adjust_event_times


class TestShiftVerification:
    """Verify the actual behavior of time shifts"""

    def test_shift_is_in_seconds_not_samples(self):
        """
        Verify that annotations.onset is in SECONDS, not samples.
        This is critical for understanding how the shift works.
        """
        # Create raw data with known sampling rate
        sfreq = 500  # 500 Hz
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 5000)
        raw = RawArray(data, info)

        # Add annotation at exactly 2.0 seconds
        annotations = Annotations(
            onset=[2.0],
            duration=[0],
            description=['Stimulus/S  1']
        )
        raw.set_annotations(annotations)

        # Verify annotation is stored in seconds
        assert raw.annotations.onset[0] == 2.0, "Annotations should be in seconds"

        # If it were in samples, it would be 1000 (2.0 * 500)
        assert raw.annotations.onset[0] != 1000, "Annotations are NOT in samples"

    def test_400ms_shift_equals_0_4_seconds(self):
        """
        Verify that shift_ms: 400 actually shifts by 0.4 seconds (400ms).
        """
        # Create raw data
        sfreq = 1000  # 1000 Hz for easy verification
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 10000)
        raw = RawArray(data, info)

        # Add annotation at 1.0 seconds
        original_time = 1.0
        annotations = Annotations(
            onset=[original_time],
            duration=[0],
            description=['Stimulus/S  1']
        )
        raw.set_annotations(annotations)

        # Apply 400ms shift
        shift_ms = 400
        result = adjust_event_times(raw, shift_ms=shift_ms, inplace=False)

        # Expected: 1.0 + 0.4 = 1.4 seconds
        expected_time = original_time + (shift_ms / 1000)
        actual_time = result.annotations.onset[0]

        assert np.isclose(actual_time, expected_time), \
            f"Expected {expected_time}s, got {actual_time}s"
        assert np.isclose(actual_time, 1.4), \
            f"400ms shift should result in 1.4s, got {actual_time}s"

    def test_shift_independent_of_sampling_rate(self):
        """
        Verify that the same shift_ms value produces the same time shift
        regardless of sampling rate (because annotations are in seconds).
        """
        shift_ms = 400

        # Test with different sampling rates
        for sfreq in [500, 1000, 2000]:
            info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
            data = np.random.randn(1, int(sfreq * 10))  # 10 seconds of data
            raw = RawArray(data, info)

            original_time = 2.0
            annotations = Annotations(
                onset=[original_time],
                duration=[0],
                description=['Stimulus/S  1']
            )
            raw.set_annotations(annotations)

            result = adjust_event_times(raw, shift_ms=shift_ms, inplace=False)

            # All should shift by exactly 0.4 seconds
            expected_time = original_time + (shift_ms / 1000)
            actual_time = result.annotations.onset[0]

            assert np.isclose(actual_time, expected_time), \
                f"At {sfreq}Hz: Expected {expected_time}s, got {actual_time}s"

    def test_positive_shift_moves_forward(self):
        """
        Verify that positive shift_ms moves events FORWARD in time (later).
        """
        sfreq = 500
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 5000)
        raw = RawArray(data, info)

        original_time = 2.0
        annotations = Annotations(
            onset=[original_time],
            duration=[0],
            description=['Stimulus/S  1']
        )
        raw.set_annotations(annotations)

        # Positive shift
        result = adjust_event_times(raw, shift_ms=400, inplace=False)

        assert result.annotations.onset[0] > original_time, \
            "Positive shift should move event FORWARD (later in time)"
        assert np.isclose(result.annotations.onset[0], 2.4), \
            "400ms forward from 2.0s should be 2.4s"

    def test_negative_shift_moves_backward(self):
        """
        Verify that negative shift_ms moves events BACKWARD in time (earlier).
        """
        sfreq = 500
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 5000)
        raw = RawArray(data, info)

        original_time = 2.0
        annotations = Annotations(
            onset=[original_time],
            duration=[0],
            description=['Stimulus/S  1']
        )
        raw.set_annotations(annotations)

        # Negative shift
        result = adjust_event_times(raw, shift_ms=-400, inplace=False)

        assert result.annotations.onset[0] < original_time, \
            "Negative shift should move event BACKWARD (earlier in time)"
        assert np.isclose(result.annotations.onset[0], 1.6), \
            "-400ms from 2.0s should be 1.6s"

    def test_conversion_accuracy(self):
        """
        Verify that the millisecond-to-second conversion is accurate.
        Test: shift_ms / 1000 = shift_sec
        """
        sfreq = 500
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 5000)
        raw = RawArray(data, info)

        test_cases = [
            (100, 0.1),    # 100ms = 0.1s
            (400, 0.4),    # 400ms = 0.4s
            (1000, 1.0),   # 1000ms = 1.0s
            (50, 0.05),    # 50ms = 0.05s
            (1500, 1.5),   # 1500ms = 1.5s
        ]

        for shift_ms, expected_shift_sec in test_cases:
            original_time = 2.0
            annotations = Annotations(
                onset=[original_time],
                duration=[0],
                description=['Stimulus/S  1']
            )
            raw.set_annotations(annotations)

            result = adjust_event_times(raw, shift_ms=shift_ms, inplace=False)

            expected_time = original_time + expected_shift_sec
            actual_time = result.annotations.onset[0]

            assert np.isclose(actual_time, expected_time, rtol=1e-9), \
                f"{shift_ms}ms: Expected {expected_time}s, got {actual_time}s"

    def test_multiple_events_all_shift_correctly(self):
        """
        Verify that when multiple events are present, they all shift by
        the same amount (in seconds, not samples).
        """
        sfreq = 500
        info = create_info(ch_names=['Cz'], sfreq=sfreq, ch_types=['eeg'])
        data = np.random.randn(1, 10000)
        raw = RawArray(data, info)

        # Events at different times
        original_times = [1.0, 3.5, 7.2, 12.8]
        annotations = Annotations(
            onset=original_times,
            duration=[0] * len(original_times),
            description=['Stimulus/S  1'] * len(original_times)
        )
        raw.set_annotations(annotations)

        shift_ms = 400
        result = adjust_event_times(raw, shift_ms=shift_ms, inplace=False)

        # All should shift by exactly 0.4 seconds
        shift_sec = shift_ms / 1000
        for i, orig_time in enumerate(original_times):
            expected = orig_time + shift_sec
            actual = result.annotations.onset[i]
            assert np.isclose(actual, expected), \
                f"Event {i}: Expected {expected}s, got {actual}s"

    def test_no_sampling_rate_dependency_in_code(self):
        """
        Verify that the code does NOT use raw.info['sfreq'] anywhere
        in the shift calculation (because it shouldn't - annotations are in seconds).
        """
        import inspect
        source = inspect.getsource(adjust_event_times)

        # The shift should only use shift_ms / 1000
        assert "shift_sec = shift_ms / 1000" in source, \
            "Shift calculation should be: shift_ms / 1000"

        # Should NOT use sfreq in the shift calculation
        # (It's fine if sfreq appears elsewhere, but not in shift calculation)
        shift_calc_section = source[source.find("shift_sec"):source.find("Apply changes")]
        assert "sfreq" not in shift_calc_section.lower(), \
            "Shift calculation should NOT depend on sampling rate"


class TestRealWorldScenario:
    """Test with realistic EEG scenarios"""

    def test_realistic_eeg_scenario(self):
        """
        Realistic scenario: 500Hz EEG data, events at various times,
        need to shift by 400ms to account for hardware delay.
        """
        # Realistic EEG setup
        sfreq = 500  # Common EEG sampling rate
        duration = 30  # 30 seconds of recording
        info = create_info(
            ch_names=['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'],
            sfreq=sfreq,
            ch_types=['eeg'] * 6
        )
        data = np.random.randn(6, int(sfreq * duration))
        raw = RawArray(data, info)

        # Realistic event times (every 2-3 seconds)
        event_times = [2.5, 5.1, 7.8, 10.2, 13.6, 16.4, 19.1, 22.5, 25.8, 28.3]
        annotations = Annotations(
            onset=event_times,
            duration=[0] * len(event_times),
            description=['Stimulus/S  1'] * len(event_times)
        )
        raw.set_annotations(annotations)

        # Apply 400ms hardware delay correction
        shift_ms = 400
        result = adjust_event_times(raw, shift_ms=shift_ms, inplace=False)

        # Verify all events shifted by exactly 400ms = 0.4s
        for i, original in enumerate(event_times):
            expected = original + 0.4
            actual = result.annotations.onset[i]
            assert np.isclose(actual, expected, atol=1e-10), \
                f"Event at {original}s: Expected {expected}s, got {actual}s"

        print("\n✓ Real-world test passed:")
        print(f"  Original times: {event_times[:3]}...")
        print(f"  Shifted times:  {[round(t, 1) for t in result.annotations.onset[:3]]}...")
        print(f"  All events shifted by exactly 400ms (0.4s)")
