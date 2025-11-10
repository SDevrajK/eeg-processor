"""
Verify that our approach to modifying annotations is valid and follows MNE best practices.
"""
import numpy as np
import pytest
from mne.io import RawArray
from mne import create_info, Annotations

from src.eeg_processor.utils.raw_data_tools import adjust_event_times


class TestMNEAnnotationsAPI:
    """Verify our usage of MNE Annotations API is correct"""

    def test_annotations_onset_is_mutable_numpy_array(self):
        """
        Verify that Annotations.onset is a mutable numpy array.
        This is important because we're creating a copy and modifying it.
        """
        ann = Annotations(onset=[1.0, 2.0], duration=[0, 0], description=['a', 'b'])

        # Should be numpy array
        assert isinstance(ann.onset, np.ndarray)

        # Should be mutable (can modify in place)
        original_value = ann.onset[0]
        ann.onset[0] = 999.0
        assert ann.onset[0] == 999.0
        assert ann.onset[0] != original_value

    def test_creating_new_annotations_is_valid_approach(self):
        """
        Verify that creating a new Annotations object with modified onset
        is a valid approach (as we do in our implementation).
        """
        original = Annotations(
            onset=[1.0, 2.0, 3.0],
            duration=[0.1, 0.2, 0.3],
            description=['a', 'b', 'c'],
            orig_time=None
        )

        # Create new annotations with shifted onset
        new_onset = original.onset.copy()
        new_onset += 0.5

        modified = Annotations(
            onset=new_onset,
            duration=original.duration,
            description=original.description,
            orig_time=original.orig_time
        )

        # Verify it works correctly
        assert len(modified) == len(original)
        assert np.allclose(modified.onset, original.onset + 0.5)
        assert np.array_equal(modified.duration, original.duration)
        assert np.array_equal(modified.description, original.description)

    def test_set_annotations_creates_copy(self):
        """
        Verify that Raw.set_annotations() stores annotations properly.
        This is important for our inplace=False behavior.
        """
        info = create_info(['Cz'], 500, ['eeg'])
        raw = RawArray(np.random.randn(1, 1000), info)

        ann = Annotations([1.0], [0], ['test'])
        original_id = id(ann)

        raw.set_annotations(ann)

        # Raw should have its own copy/reference
        assert hasattr(raw, 'annotations')
        assert len(raw.annotations) == 1

        # Modifying original shouldn't affect raw's annotations
        ann.onset[0] = 999.0
        assert raw.annotations.onset[0] != 999.0

    def test_our_implementation_vs_direct_modification(self):
        """
        Compare our implementation (creating new Annotations) vs
        direct modification of onset array.
        Both should produce the same result.
        """
        # Setup
        info = create_info(['Cz'], 500, ['eeg'])
        raw1 = RawArray(np.random.randn(1, 5000), info)
        raw2 = raw1.copy()

        ann = Annotations([1.0, 2.0, 3.0], [0, 0, 0], ['Stimulus/S  1'] * 3)
        raw1.set_annotations(ann)
        raw2.set_annotations(ann)

        shift_sec = 0.4

        # Method 1: Our implementation (create new Annotations)
        result1 = adjust_event_times(raw1, shift_ms=400, inplace=False)

        # Method 2: Direct modification (alternative approach)
        raw2.annotations.onset[:] += shift_sec

        # Both should produce identical results
        assert np.allclose(result1.annotations.onset, raw2.annotations.onset)

    def test_mne_does_not_provide_shift_method(self):
        """
        Verify that MNE doesn't provide a built-in shift method,
        justifying our custom implementation.
        """
        ann = Annotations([1.0], [0], ['test'])

        # Should NOT have shift/adjust methods
        assert not hasattr(ann, 'shift')
        assert not hasattr(ann, 'adjust')
        assert not hasattr(ann, 'offset')

        # The methods MNE does provide
        expected_methods = ['append', 'copy', 'crop', 'delete', 'rename', 'save', 'set_durations']
        for method in expected_methods:
            assert hasattr(ann, method), f"Expected method '{method}' not found"

    def test_annotations_properties_preserved_after_modification(self):
        """
        Verify that all Annotations properties are preserved when
        creating a new Annotations object (as our implementation does).
        """
        original = Annotations(
            onset=[1.0, 2.0],
            duration=[0.5, 0.7],
            description=['stim1', 'stim2'],
            orig_time=1234567890.0
        )

        # Create new annotations as our implementation does
        new = Annotations(
            onset=original.onset + 0.3,
            duration=original.duration,
            description=original.description,
            orig_time=original.orig_time
        )

        # All properties should be preserved except onset
        assert np.allclose(new.onset, original.onset + 0.3)
        assert np.array_equal(new.duration, original.duration)
        assert np.array_equal(new.description, original.description)
        assert new.orig_time == original.orig_time

    def test_selective_shift_preserves_unshifted_events(self):
        """
        Verify that when shifting only certain events, unshifted events
        remain at their original times.
        """
        info = create_info(['Cz'], 500, ['eeg'])
        raw = RawArray(np.random.randn(1, 5000), info)

        original_times = [1.0, 2.0, 3.0, 4.0]
        ann = Annotations(
            onset=original_times,
            duration=[0] * 4,
            description=['Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  1', 'Stimulus/S  2']
        )
        raw.set_annotations(ann)

        # Shift only event type 1
        result = adjust_event_times(raw, shift_ms=400, target_events=[1], inplace=False)

        # Events with description "Stimulus/S  1" (indices 0, 2) should be shifted
        assert np.isclose(result.annotations.onset[0], 1.4)  # 1.0 + 0.4
        assert np.isclose(result.annotations.onset[2], 3.4)  # 3.0 + 0.4

        # Events with description "Stimulus/S  2" (indices 1, 3) should be unchanged
        assert np.isclose(result.annotations.onset[1], 2.0)
        assert np.isclose(result.annotations.onset[3], 4.0)

    def test_our_approach_handles_empty_annotations(self):
        """
        Verify our implementation handles edge case of no annotations.
        """
        info = create_info(['Cz'], 500, ['eeg'])
        raw = RawArray(np.random.randn(1, 1000), info)

        # No annotations set
        result = adjust_event_times(raw, shift_ms=400, inplace=False)

        # Should return successfully with no modifications
        assert len(result.annotations) == 0

    def test_inplace_vs_copy_behavior(self):
        """
        Verify that inplace=True modifies the original and inplace=False
        creates a copy, as expected.
        """
        info = create_info(['Cz'], 500, ['eeg'])
        raw = RawArray(np.random.randn(1, 5000), info)
        ann = Annotations([1.0, 2.0], [0, 0], ['Stimulus/S  1'] * 2)
        raw.set_annotations(ann)

        original_times = raw.annotations.onset.copy()

        # inplace=False should not modify original
        result_copy = adjust_event_times(raw, shift_ms=400, inplace=False)
        assert result_copy is not raw
        assert np.array_equal(raw.annotations.onset, original_times)
        assert not np.array_equal(result_copy.annotations.onset, original_times)

        # inplace=True should modify original
        result_inplace = adjust_event_times(raw, shift_ms=400, inplace=True)
        assert result_inplace is raw
        assert not np.array_equal(raw.annotations.onset, original_times)


class TestAnnotationsAPIDocumentation:
    """Document what we learned about MNE Annotations API"""

    def test_document_valid_approaches(self):
        """
        Document the valid approaches for modifying annotation times
        based on MNE's API.
        """
        ann = Annotations([1.0, 2.0], [0, 0], ['a', 'b'])

        # Approach 1: Direct modification (works but modifies in place)
        ann_direct = ann.copy()
        ann_direct.onset[:] += 0.5
        assert np.allclose(ann_direct.onset, [1.5, 2.5])

        # Approach 2: Create new Annotations (our approach - safer)
        ann_new = Annotations(
            onset=ann.onset + 0.5,
            duration=ann.duration,
            description=ann.description,
            orig_time=ann.orig_time
        )
        assert np.allclose(ann_new.onset, [1.5, 2.5])

        # Both approaches are valid according to MNE's API
        print("\n✓ Both approaches produce identical results")
        print("  Our approach (creating new Annotations) is safer for:")
        print("  - Preserving original data when inplace=False")
        print("  - Selective modification with masks")
        print("  - Clear separation of concerns")
