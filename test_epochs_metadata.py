"""
Test epochs metadata handling specifically.
"""

import mne
import numpy as np
import pandas as pd
from datetime import date
import tempfile
import os

def test_epochs_metadata():
    """Test epochs metadata approach thoroughly."""
    
    print("Testing epochs metadata approach...")
    
    # Create raw data
    sfreq = 1000
    n_channels = 4
    n_samples = 2000  # Longer for proper epochs
    data = np.random.randn(n_channels, n_samples)
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info)
    
    # Create events
    events = mne.make_fixed_length_events(raw, duration=0.5, start=0.1)[:5]
    print(f"Created {len(events)} events")
    
    # Create epochs with proper parameters
    epochs = mne.Epochs(
        raw, 
        events, 
        tmin=-0.1, 
        tmax=0.4, 
        baseline=(-0.1, 0),
        preload=True,  # Important!
        verbose=False
    )
    
    print(f"Created epochs: {len(epochs)} epochs")
    
    # Create metadata with participant info
    participant_metadata = pd.DataFrame({
        'participant_id': ['S_002_F'] * len(epochs),
        'responder': ['yes'] * len(epochs),
        'age': [25] * len(epochs),
        'group': ['control'] * len(epochs),
        'trial_number': range(1, len(epochs) + 1)
    })
    
    # Add metadata to epochs
    epochs.metadata = participant_metadata
    print("✓ Added metadata to epochs")
    print(f"Metadata shape: {epochs.metadata.shape}")
    print("Sample metadata:")
    print(epochs.metadata.head(2))
    
    # Test saving and loading
    with tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        epochs.save(temp_file, overwrite=True, verbose=False)
        print("✓ Saved epochs with metadata")
        
        # Load back
        loaded_epochs = mne.read_epochs(temp_file, verbose=False)
        print("✓ Loaded epochs")
        
        # Check metadata persistence
        if loaded_epochs.metadata is not None:
            print("✓ Metadata survived save/load cycle!")
            print(f"Loaded metadata shape: {loaded_epochs.metadata.shape}")
            print("Sample loaded metadata:")
            print(loaded_epochs.metadata.head(2))
            
            # Verify specific fields
            responder_values = loaded_epochs.metadata['responder'].unique()
            print(f"Responder values preserved: {responder_values}")
            
        else:
            print("✗ Metadata was lost during save/load")
            
    except Exception as e:
        print(f"✗ Error during epochs metadata test: {e}")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Test with other MNE objects created from epochs
    print("\nTesting metadata propagation to derived objects:")
    
    try:
        # Test evoked (averaged epochs)
        evoked = epochs.average()
        print("✓ Created evoked from epochs")
        print(f"Evoked has metadata attr: {hasattr(evoked, 'metadata')}")
        
        # Save/load evoked
        with tempfile.NamedTemporaryFile(suffix='-ave.fif', delete=False) as tmp:
            evoked_file = tmp.name
            
        evoked.save(evoked_file, overwrite=True, verbose=False)
        loaded_evoked = mne.read_evoked(evoked_file, verbose=False)
        print(f"Loaded evoked has metadata: {hasattr(loaded_evoked, 'metadata')}")
        
        os.unlink(evoked_file)
        
    except Exception as e:
        print(f"✗ Error testing evoked: {e}")

if __name__ == "__main__":
    test_epochs_metadata()