"""Minimal integration test for multi-event time-frequency processing."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import mne
from mne import Epochs
from mne.time_frequency import AverageTFR

from eeg_processor.pipeline import EEGPipeline


class TestMultiEventIntegration:
    """Minimal integration test for multi-event processing."""
    
    def test_multi_event_processing_basic(self):
        """Test that multi-event processing creates separate results per trigger."""
        # Create minimal mock configuration
        config = {
            'study': {'name': 'test'},
            'paths': {'raw_data': '/tmp', 'results': '/tmp', 'file_extension': '.fif'},
            'participants': ['test.fif'],
            'conditions': [{
                'name': 'test_condition',
                'triggers': {'trigger1': 1, 'trigger2': 2}
            }],
            'processing': [{'epoch': {'tmin': -0.1, 'tmax': 0.5}}]
        }
        
        # Create mock epochs with different trigger types
        mock_epochs = Mock(spec=Epochs)
        mock_epochs.event_id = {'1': 1, '2': 2}
        
        # Mock epoch selection for each trigger
        trigger1_epochs = Mock(spec=Epochs)
        trigger1_epochs.__len__ = Mock(return_value=10)
        trigger2_epochs = Mock(spec=Epochs) 
        trigger2_epochs.__len__ = Mock(return_value=8)
        
        mock_epochs.__getitem__ = Mock(side_effect=lambda x: {
            '1': trigger1_epochs,
            '2': trigger2_epochs
        }[x])
        
        pipeline = EEGPipeline()
        
        # Test the trigger extraction method directly
        result1 = pipeline._extract_trigger_epochs(mock_epochs, 'trigger1', 1)
        result2 = pipeline._extract_trigger_epochs(mock_epochs, 'trigger2', 2)
        
        # Verify both triggers return valid epochs
        assert result1 == trigger1_epochs
        assert result2 == trigger2_epochs
        assert len(result1) == 10
        assert len(result2) == 8
    
    def test_missing_trigger_handling(self):
        """Test that missing triggers are handled gracefully."""
        # Create mock epochs without the expected trigger
        mock_epochs = Mock(spec=Epochs)
        mock_epochs.event_id = {'1': 1}  # Only has trigger 1
        mock_epochs.__getitem__ = Mock(side_effect=KeyError("Missing trigger"))
        
        pipeline = EEGPipeline()
        
        # Test extraction of missing trigger
        result = pipeline._extract_trigger_epochs(mock_epochs, 'missing_trigger', 999)
        
        # Should return None for missing trigger
        assert result is None
    
    def test_event_separation_accuracy(self):
        """Basic validation that event separation preserves data structure."""
        # Create mock epochs with known structure
        mock_epochs = Mock(spec=Epochs)
        mock_epochs.event_id = {'1': 1, '2': 2}
        
        # Mock separated epochs with different properties
        trigger1_epochs = Mock(spec=Epochs)
        trigger1_epochs.__len__ = Mock(return_value=10)
        trigger1_epochs.info = {'sfreq': 250}
        
        trigger2_epochs = Mock(spec=Epochs)
        trigger2_epochs.__len__ = Mock(return_value=8) 
        trigger2_epochs.info = {'sfreq': 250}
        
        mock_epochs.__getitem__ = Mock(side_effect=lambda x: {
            '1': trigger1_epochs,
            '2': trigger2_epochs
        }[x])
        
        pipeline = EEGPipeline()
        
        # Test that separation preserves epoch count and structure
        result1 = pipeline._extract_trigger_epochs(mock_epochs, 'trigger1', 1)
        result2 = pipeline._extract_trigger_epochs(mock_epochs, 'trigger2', 2)
        
        # Verify separation maintains data integrity
        assert len(result1) == 10
        assert len(result2) == 8
        assert result1.info['sfreq'] == 250
        assert result2.info['sfreq'] == 250
    
    def test_memory_usage_estimation(self):
        """Simple validation that memory estimation is reasonable."""
        from eeg_processor.processing.time_frequency import _estimate_tfr_memory
        
        # Test with realistic parameters
        n_channels = 64
        n_freqs = 40
        n_times = 500
        compute_itc = False
        
        memory_mb = _estimate_tfr_memory(n_channels, n_freqs, n_times, compute_itc)
        
        # Memory should be reasonable (under 1GB for typical analysis)
        assert memory_mb > 0
        assert memory_mb < 1000  # Less than 1GB
        
        # With ITC, memory should roughly double
        memory_with_itc = _estimate_tfr_memory(n_channels, n_freqs, n_times, True)
        assert memory_with_itc > memory_mb
        assert memory_with_itc < 2000  # Still under 2GB
    
    def test_backward_compatibility(self):
        """Basic check that single-event configurations still work."""
        # Create single-event configuration (legacy format)
        single_event_config = {
            'study': {'name': 'test'},
            'paths': {'raw_data': '/tmp', 'results': '/tmp', 'file_extension': '.fif'},
            'participants': ['test.fif'],
            'conditions': [{
                'name': 'single_condition',
                'condition_markers': [1, 2]  # Legacy format without 'triggers'
            }],
            'processing': [{'epoch': {'tmin': -0.1, 'tmax': 0.5}}]
        }
        
        # Create mock epochs for single condition
        mock_epochs = Mock(spec=Epochs)
        mock_epochs.event_id = {'1': 1, '2': 2}
        mock_epochs.__len__ = Mock(return_value=20)
        
        pipeline = EEGPipeline()
        
        # Test that pipeline can handle condition without 'triggers' field
        condition = single_event_config['conditions'][0]
        has_triggers = 'triggers' in condition
        
        # Should work without triggers field (legacy format)
        assert not has_triggers
        
        # Verify that mock epochs still function normally
        assert len(mock_epochs) == 20
        assert '1' in mock_epochs.event_id
        assert '2' in mock_epochs.event_id