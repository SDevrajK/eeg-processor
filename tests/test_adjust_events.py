"""
Tests for adjust_events functionality with normalized event code matching
"""
import numpy as np
import pytest
from mne.io import RawArray
from mne import create_info, Annotations

from src.eeg_processor.utils.raw_data_tools import adjust_event_times


@pytest.fixture
def mock_brainvision_raw():
    """Create mock BrainVision-style Raw object with typical event codes"""
    # Create minimal raw data
    info = create_info(ch_names=['Cz', 'Fz'], sfreq=500, ch_types=['eeg', 'eeg'])
    data = np.random.randn(2, 5000)
    raw = RawArray(data, info)

    # Add BrainVision-style annotations with varying spacing
    onset = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    duration = np.zeros(5)
    description = [
        'Stimulus/S  1',   # Single digit - two spaces
        'Stimulus/S  2',   # Single digit - two spaces
        'Stimulus/S 10',   # Double digit - one space
        'Stimulus/S 11',   # Double digit - one space
        'Stimulus/S100'    # Triple digit - no space
    ]

    annotations = Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annotations)

    return raw


@pytest.fixture
def mock_curry_raw():
    """Create mock Curry-style Raw object with numeric event codes"""
    info = create_info(ch_names=['Cz', 'Fz'], sfreq=500, ch_types=['eeg', 'eeg'])
    data = np.random.randn(2, 5000)
    raw = RawArray(data, info)

    # Curry uses plain numbers
    onset = np.array([1.0, 2.0, 3.0])
    duration = np.zeros(3)
    description = ['1', '2', '10']

    annotations = Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annotations)

    return raw


class TestAdjustEventsBasic:
    """Test basic functionality of adjust_event_times"""

    def test_shift_all_events(self, mock_brainvision_raw):
        """Test shifting all events without target specification"""
        shift_ms = 50  # 50 ms shift

        original_onsets = mock_brainvision_raw.annotations.onset.copy()
        result = adjust_event_times(mock_brainvision_raw, shift_ms=shift_ms, inplace=False)

        expected_onsets = original_onsets + (shift_ms / 1000)
        np.testing.assert_allclose(result.annotations.onset, expected_onsets)

    def test_inplace_operation(self, mock_brainvision_raw):
        """Test that inplace=True modifies the original object"""
        shift_ms = 30
        original_onsets = mock_brainvision_raw.annotations.onset.copy()

        result = adjust_event_times(mock_brainvision_raw, shift_ms=shift_ms, inplace=True)

        # Should return the same object
        assert result is mock_brainvision_raw

        # Original should be modified
        expected_onsets = original_onsets + (shift_ms / 1000)
        np.testing.assert_allclose(mock_brainvision_raw.annotations.onset, expected_onsets)

    def test_copy_operation(self, mock_brainvision_raw):
        """Test that inplace=False creates a copy"""
        shift_ms = 30
        original_onsets = mock_brainvision_raw.annotations.onset.copy()

        result = adjust_event_times(mock_brainvision_raw, shift_ms=shift_ms, inplace=False)

        # Should return different object
        assert result is not mock_brainvision_raw

        # Original should be unchanged
        np.testing.assert_allclose(mock_brainvision_raw.annotations.onset, original_onsets)

        # Result should be shifted
        expected_onsets = original_onsets + (shift_ms / 1000)
        np.testing.assert_allclose(result.annotations.onset, expected_onsets)


class TestAdjustEventsNormalization:
    """Test normalized event code matching across formats"""

    def test_target_events_with_integers(self, mock_brainvision_raw):
        """Test that integer event codes work with BrainVision spacing"""
        shift_ms = 100
        target_events = [1, 10]  # Should match 'Stimulus/S  1' and 'Stimulus/S 10'

        result = adjust_event_times(
            mock_brainvision_raw,
            shift_ms=shift_ms,
            target_events=target_events,
            inplace=False
        )

        # Check that only events 1 and 10 were shifted
        original = mock_brainvision_raw.annotations
        result_annot = result.annotations

        for i, desc in enumerate(original.description):
            if desc in ['Stimulus/S  1', 'Stimulus/S 10']:
                # Should be shifted
                expected = original.onset[i] + (shift_ms / 1000)
                assert abs(result_annot.onset[i] - expected) < 1e-6
            else:
                # Should be unchanged
                assert abs(result_annot.onset[i] - original.onset[i]) < 1e-6

    def test_target_events_with_strings(self, mock_brainvision_raw):
        """Test that string event codes are normalized properly"""
        shift_ms = 100
        target_events = ["S1", "S10"]  # Should normalize to match actual events

        result = adjust_event_times(
            mock_brainvision_raw,
            shift_ms=shift_ms,
            target_events=target_events,
            inplace=False
        )

        # Check that events were properly matched and shifted
        original = mock_brainvision_raw.annotations
        result_annot = result.annotations

        for i, desc in enumerate(original.description):
            if desc in ['Stimulus/S  1', 'Stimulus/S 10']:
                expected = original.onset[i] + (shift_ms / 1000)
                assert abs(result_annot.onset[i] - expected) < 1e-6
            else:
                assert abs(result_annot.onset[i] - original.onset[i]) < 1e-6

    def test_protect_events(self, mock_brainvision_raw):
        """Test that protect_events prevents shifting"""
        shift_ms = 100
        protect_events = [1]  # Protect event 1

        result = adjust_event_times(
            mock_brainvision_raw,
            shift_ms=shift_ms,
            protect_events=protect_events,
            inplace=False
        )

        original = mock_brainvision_raw.annotations
        result_annot = result.annotations

        # Event 1 should NOT be shifted
        assert abs(result_annot.onset[0] - original.onset[0]) < 1e-6

        # All other events should be shifted
        for i in range(1, len(original.onset)):
            expected = original.onset[i] + (shift_ms / 1000)
            assert abs(result_annot.onset[i] - expected) < 1e-6

    def test_target_and_protect_interaction(self, mock_brainvision_raw):
        """Test that protect_events overrides target_events"""
        shift_ms = 100
        target_events = [1, 2, 10]
        protect_events = [2]  # Protect event 2 even though it's in target

        result = adjust_event_times(
            mock_brainvision_raw,
            shift_ms=shift_ms,
            target_events=target_events,
            protect_events=protect_events,
            inplace=False
        )

        original = mock_brainvision_raw.annotations
        result_annot = result.annotations

        for i, desc in enumerate(original.description):
            if desc == 'Stimulus/S  2':
                # Protected - should NOT be shifted
                assert abs(result_annot.onset[i] - original.onset[i]) < 1e-6
            elif desc in ['Stimulus/S  1', 'Stimulus/S 10']:
                # Targeted - should be shifted
                expected = original.onset[i] + (shift_ms / 1000)
                assert abs(result_annot.onset[i] - expected) < 1e-6
            else:
                # Not targeted - should NOT be shifted
                assert abs(result_annot.onset[i] - original.onset[i]) < 1e-6


class TestAdjustEventsCurryFormat:
    """Test with Curry format (plain numeric codes)"""

    def test_curry_format_with_integers(self, mock_curry_raw):
        """Test that integer codes work with Curry format"""
        shift_ms = 50
        target_events = [1, 10]

        result = adjust_event_times(
            mock_curry_raw,
            shift_ms=shift_ms,
            target_events=target_events,
            inplace=False
        )

        original = mock_curry_raw.annotations
        result_annot = result.annotations

        for i, desc in enumerate(original.description):
            if desc in ['1', '10']:
                expected = original.onset[i] + (shift_ms / 1000)
                assert abs(result_annot.onset[i] - expected) < 1e-6
            else:
                assert abs(result_annot.onset[i] - original.onset[i]) < 1e-6


class TestAdjustEventsEdgeCases:
    """Test edge cases and error conditions"""

    def test_no_annotations(self):
        """Test handling of raw data with no annotations"""
        info = create_info(ch_names=['Cz'], sfreq=500, ch_types=['eeg'])
        data = np.random.randn(1, 1000)
        raw = RawArray(data, info)

        # Should return unchanged (or copy)
        result = adjust_event_times(raw, shift_ms=50, inplace=False)
        assert result is not raw  # Should be a copy
        assert len(result.annotations) == 0

    def test_negative_shift(self, mock_brainvision_raw):
        """Test that negative shifts work correctly"""
        shift_ms = -50  # Shift backwards

        original_onsets = mock_brainvision_raw.annotations.onset.copy()
        result = adjust_event_times(mock_brainvision_raw, shift_ms=shift_ms, inplace=False)

        expected_onsets = original_onsets + (shift_ms / 1000)
        np.testing.assert_allclose(result.annotations.onset, expected_onsets)

    def test_large_shift(self, mock_brainvision_raw):
        """Test with large shift value"""
        shift_ms = 5000  # 5 second shift

        original_onsets = mock_brainvision_raw.annotations.onset.copy()
        result = adjust_event_times(mock_brainvision_raw, shift_ms=shift_ms, inplace=False)

        expected_onsets = original_onsets + (shift_ms / 1000)
        np.testing.assert_allclose(result.annotations.onset, expected_onsets)

    def test_empty_target_events(self, mock_brainvision_raw):
        """Test behavior with empty target_events list"""
        shift_ms = 50
        target_events = []

        original_onsets = mock_brainvision_raw.annotations.onset.copy()
        result = adjust_event_times(
            mock_brainvision_raw,
            shift_ms=shift_ms,
            target_events=target_events,
            inplace=False
        )

        # With empty target list, no events should match, so none should shift
        np.testing.assert_allclose(result.annotations.onset, original_onsets)


class TestAdjustEventsConsistency:
    """Test that the fix ensures consistent behavior across subjects"""

    def test_consistent_behavior_different_event_ranges(self):
        """
        Test that events shift consistently regardless of whether they're
        single-digit, double-digit, or triple-digit (the original bug)
        """
        # Subject A: Only single-digit events
        info_a = create_info(ch_names=['Cz'], sfreq=500, ch_types=['eeg'])
        data_a = np.random.randn(1, 5000)
        raw_a = RawArray(data_a, info_a)
        raw_a.set_annotations(Annotations(
            onset=[1.0, 2.0],
            duration=[0, 0],
            description=['Stimulus/S  1', 'Stimulus/S  2']
        ))

        # Subject B: Only double-digit events
        info_b = create_info(ch_names=['Cz'], sfreq=500, ch_types=['eeg'])
        data_b = np.random.randn(1, 5000)
        raw_b = RawArray(data_b, info_b)
        raw_b.set_annotations(Annotations(
            onset=[1.0, 2.0],
            duration=[0, 0],
            description=['Stimulus/S 10', 'Stimulus/S 11']
        ))

        shift_ms = 100

        # Shift all events for both subjects using integers
        result_a = adjust_event_times(raw_a, shift_ms=shift_ms, target_events=[1, 2])
        result_b = adjust_event_times(raw_b, shift_ms=shift_ms, target_events=[10, 11])

        # Both should have consistent shifts
        expected_shift_sec = shift_ms / 1000

        for i in range(len(result_a.annotations)):
            actual_shift_a = result_a.annotations.onset[i] - raw_a.annotations.onset[i]
            assert abs(actual_shift_a - expected_shift_sec) < 1e-6

        for i in range(len(result_b.annotations)):
            actual_shift_b = result_b.annotations.onset[i] - raw_b.annotations.onset[i]
            assert abs(actual_shift_b - expected_shift_sec) < 1e-6
