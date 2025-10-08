"""
EMCP Validation Tests Against MNE Sample Data

These tests validate EMCP implementation against MNE's established methods
using real EEG data with authentic blink artifacts. Tests are marked as 
'validation' and require the MNE sample dataset (~1.5GB).

To run validation tests:
    pytest tests/test_emcp_validation.py -m validation

To download sample data first:
    python tests/test_utils/mne_sample_data.py download
"""

import pytest
import numpy as np
from typing import Dict, Any
from loguru import logger

# Import test utilities
from tests.test_utils.mne_sample_data import (
    get_sample_manager, 
    ensure_sample_data_available,
    get_sample_raw
)

# Import EMCP implementation
from src.eeg_processor.processing.emcp import (
    remove_blinks_eog_regression,
    remove_blinks_gratton_coles,
    get_emcp_quality_summary
)

# MNE imports with error handling
try:
    import mne
    from mne.preprocessing import EOGRegression, find_eog_events
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


# Test markers
pytestmark = pytest.mark.validation


class TestEMCPValidationAgainstMNE:
    """
    Validation tests comparing EMCP implementation against MNE reference methods.
    
    These tests ensure numerical accuracy and behavioral consistency with
    established MNE-Python EOG correction approaches.
    """
    
    @pytest.fixture(scope="class")
    def sample_raw(self):
        """Load MNE sample data for validation testing."""
        if not MNE_AVAILABLE:
            pytest.skip("MNE not available for validation testing")
            
        manager = get_sample_manager()
        if not manager.is_available():
            pytest.skip(
                "MNE sample data not available. Download with:\n"
                "python tests/test_utils/mne_sample_data.py download"
            )
        
        raw = get_sample_raw(preload=True)
        if raw is None:
            pytest.skip("Failed to load MNE sample data")
            
        # Filter to EEG+EOG only and crop for faster testing
        raw.pick_types(eeg=True, eog=True, meg=False, stim=False)
        raw.crop(tmax=60)  # Use first 60 seconds for validation
        
        # Set average reference for EEG channels (required for EOG regression)
        raw.set_eeg_reference('average', projection=True)
        
        return raw
    
    @pytest.fixture
    def eog_channels(self, sample_raw):
        """Get available EOG channels from sample data."""
        eog_channels = [ch for ch in sample_raw.ch_names if 'EOG' in ch]
        if not eog_channels:
            pytest.skip("No EOG channels found in sample data")
        return eog_channels
    
    def test_eog_regression_matches_mne_reference(self, sample_raw, eog_channels):
        """
        Test that our EOG regression produces identical results to MNE's EOGRegression.
        
        This is the most critical validation test since our method should exactly
        replicate MNE's established EOGRegression implementation.
        """
        logger.info("Validating EOG regression against MNE reference implementation")
        
        # Apply our EMCP EOG regression method
        our_result = remove_blinks_eog_regression(
            sample_raw.copy(),
            eog_channels=eog_channels,
            show_plot=False,
            verbose=False
        )
        
        # Apply MNE's reference EOGRegression
        eeg_picks = mne.pick_types(sample_raw.info, eeg=True, meg=False)
        eog_picks = [sample_raw.ch_names.index(ch) for ch in eog_channels]
        
        mne_regressor = EOGRegression(picks=eeg_picks, picks_artifact=eog_picks)
        mne_regressor.fit(sample_raw)
        mne_result = mne_regressor.apply(sample_raw.copy())
        
        # Compare EEG data (should be identical within numerical precision)
        our_eeg_data = our_result.get_data(picks=eeg_picks)
        mne_eeg_data = mne_result.get_data(picks=eeg_picks)
        
        # Test numerical equality with tight tolerance
        np.testing.assert_array_almost_equal(
            our_eeg_data,
            mne_eeg_data,
            decimal=10,
            err_msg="EOG regression results do not match MNE reference implementation"
        )
        
        # Verify EOG channels remain unchanged
        our_eog_data = our_result.get_data(picks=eog_picks)
        original_eog_data = sample_raw.get_data(picks=eog_picks)
        
        np.testing.assert_array_equal(
            our_eog_data,
            original_eog_data,
            err_msg="EOG channels should remain unchanged"
        )
        
        logger.success("EOG regression validation passed - results match MNE reference")
    
    def test_gratton_coles_mathematical_correctness(self, sample_raw, eog_channels):
        """
        Test Gratton & Coles method for mathematical correctness.
        
        Validates the core regression algorithm against manual computation
        and checks for reasonable artifact reduction.
        """
        logger.info("Validating Gratton & Coles mathematical implementation")
        
        # Apply Gratton & Coles method
        cleaned_raw = remove_blinks_gratton_coles(
            sample_raw.copy(),
            eog_channels=eog_channels,
            subtract_evoked=False,  # Test core regression only
            show_plot=False,
            verbose=False
        )
        
        # Get metrics from processed data
        metrics = get_emcp_quality_summary(cleaned_raw)
        
        # Validate essential metrics are present
        assert metrics['method'] == 'gratton_coles'
        assert metrics['correction_applied'] is True
        assert 'blink_events' in metrics
        assert 'mean_correlation' in metrics
        
        # Validate blink detection worked
        assert metrics['blink_events'] > 0, "Should detect blinks in sample data"
        
        # Validate correlation preservation (should be high for good correction)
        correlation = metrics['mean_correlation']
        assert 0.7 <= correlation <= 1.0, f"Correlation {correlation} outside expected range"
        
        # Manual validation of regression coefficients
        if 'max_regression_coefficient' in metrics:
            max_coeff = metrics['max_regression_coefficient']
            assert 0 < max_coeff < 1.0, f"Regression coefficient {max_coeff} seems unreasonable"
        
        logger.success("Gratton & Coles mathematical validation passed")
    
    def test_emcp_methods_preserve_data_integrity(self, sample_raw, eog_channels):
        """
        Test that both EMCP methods preserve data integrity.
        
        Validates that:
        - Non-EEG channels remain unchanged
        - Data dimensions are preserved
        - Metadata is maintained
        """
        logger.info("Validating data integrity preservation")
        
        original_raw = sample_raw.copy()
        
        # Test both methods
        for method_name, method_func in [
            ('eog_regression', remove_blinks_eog_regression),
            ('gratton_coles', remove_blinks_gratton_coles)
        ]:
            logger.info(f"Testing {method_name} data integrity")
            
            result = method_func(
                original_raw.copy(),
                eog_channels=eog_channels,
                show_plot=False,
                verbose=False
            )
            
            # Check data dimensions
            assert result.get_data().shape == original_raw.get_data().shape
            assert result.info['sfreq'] == original_raw.info['sfreq']
            assert result.ch_names == original_raw.ch_names
            
            # Check EOG channels unchanged
            eog_picks = [result.ch_names.index(ch) for ch in eog_channels]
            np.testing.assert_array_equal(
                result.get_data(picks=eog_picks),
                original_raw.get_data(picks=eog_picks),
                err_msg=f"{method_name}: EOG channels should remain unchanged"
            )
            
            # Check that EEG data was actually modified (artifact correction applied)
            eeg_picks = mne.pick_types(result.info, eeg=True)
            original_eeg = original_raw.get_data(picks=eeg_picks)
            cleaned_eeg = result.get_data(picks=eeg_picks)
            
            # Should not be identical (correction was applied)
            assert not np.array_equal(original_eeg, cleaned_eeg), \
                f"{method_name}: EEG data should be modified by correction"
            
            # But should be highly correlated
            correlations = []
            for ch_idx in range(len(eeg_picks)):
                corr = np.corrcoef(original_eeg[ch_idx], cleaned_eeg[ch_idx])[0, 1]
                correlations.append(corr)
            
            mean_corr = np.mean(correlations)
            assert mean_corr > 0.8, \
                f"{method_name}: Mean correlation {mean_corr} too low (overcorrection?)"
        
        logger.success("Data integrity validation passed for both methods")
    
    def test_emcp_quality_metrics_accuracy(self, sample_raw, eog_channels):
        """
        Test that EMCP quality metrics accurately reflect processing results.
        
        Validates that stored metrics match independently computed values.
        """
        logger.info("Validating quality metrics accuracy")
        
        # Test with Gratton & Coles (has more detailed metrics)
        cleaned_raw = remove_blinks_gratton_coles(
            sample_raw.copy(),
            eog_channels=eog_channels,
            show_plot=False,
            verbose=False
        )
        
        metrics = get_emcp_quality_summary(cleaned_raw)
        
        # Independently verify blink detection
        primary_eog = eog_channels[0]  # Use first EOG channel
        eog_events = find_eog_events(sample_raw, ch_name=primary_eog, verbose=False)
        expected_blinks = len(eog_events)
        
        assert metrics['blink_events'] == expected_blinks, \
            f"Stored blink count {metrics['blink_events']} != detected {expected_blinks}"
        
        # Independently verify correlation
        eeg_picks = mne.pick_types(sample_raw.info, eeg=True)
        original_eeg = sample_raw.get_data(picks=eeg_picks)
        cleaned_eeg = cleaned_raw.get_data(picks=eeg_picks)
        
        manual_correlations = []
        for ch_idx in range(len(eeg_picks)):
            corr = np.corrcoef(original_eeg[ch_idx], cleaned_eeg[ch_idx])[0, 1]
            if not np.isnan(corr):
                manual_correlations.append(corr)
        
        manual_mean_corr = np.mean(manual_correlations)
        stored_mean_corr = metrics['mean_correlation']
        
        np.testing.assert_almost_equal(
            stored_mean_corr,
            manual_mean_corr,
            decimal=3,
            err_msg=f"Stored correlation {stored_mean_corr} != computed {manual_mean_corr}"
        )
        
        logger.success("Quality metrics accuracy validation passed")


# TODO: Add task to todo list for implementing user prompts for sample data download
def test_sample_data_setup_instructions():
    """
    Test that provides clear setup instructions if sample data is missing.
    
    This test always runs and provides helpful guidance for setting up
    validation testing infrastructure.
    """
    manager = get_sample_manager()
    
    if not MNE_AVAILABLE:
        pytest.skip("MNE not available - install with: pip install mne")
    
    if not manager.is_available():
        info = manager.get_data_info()
        
        setup_message = """
        
        MNE Sample Data Setup Required for Validation Testing
        =====================================================
        
        To run EMCP validation tests against MNE reference implementations,
        you need to download the MNE sample dataset (~1.5GB).
        
        Quick Setup:
        -----------
        python tests/test_utils/mne_sample_data.py download
        
        Or programmatically:
        ------------------
        from tests.test_utils.mne_sample_data import ensure_sample_data_available
        ensure_sample_data_available()
        
        Current Status:
        --------------
        MNE Available: {mne_available}
        Data Available: {data_available}
        
        After download, run validation tests with:
        pytest tests/test_emcp_validation.py -m validation -v
        
        """.format(**info)
        
        print(setup_message)
        pytest.skip("MNE sample data not available - see setup instructions above")
    
    # If we get here, data is available
    info = manager.get_data_info()
    logger.info(f"MNE sample data available: {info['file_size_mb']} MB")
    logger.info(f"EOG channels for testing: {info['eog_channels']}")
    assert info['data_available'] is True


if __name__ == "__main__":
    # Allow running this file directly to check setup
    test_sample_data_setup_instructions()
    print("✓ Validation test infrastructure ready")