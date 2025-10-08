"""
Test script to investigate MNE's restrictions on adding metadata to info dictionaries.
"""

import mne
import numpy as np

def test_mne_metadata_restrictions():
    """Test different approaches to adding participant metadata to MNE objects."""
    
    print("Testing MNE metadata restrictions...")
    
    # Create a simple raw object for testing
    sfreq = 1000
    n_channels = 4
    n_samples = 1000
    data = np.random.randn(n_channels, n_samples)
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    ch_types = ['eeg'] * n_channels
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Test participant metadata
    participant_metadata = {
        'participant_id': 'S_002_F',
        'responder': 'yes',
        'age': 25,
        'group': 'control'
    }
    
    print("\n1. Testing direct info dictionary modification:")
    try:
        raw.info['subject_info'] = participant_metadata
        print("✓ Successfully added subject_info directly")
        print(f"  Added: {raw.info['subject_info']}")
    except Exception as e:
        print(f"✗ Failed to add subject_info directly: {e}")
    
    print("\n2. Testing MNE's official subject_info format:")
    try:
        # MNE expects specific fields in subject_info
        official_subject_info = {
            'id': 1,  # Integer ID required
            'his_id': 'S_002_F',
            'last_name': 'Control',
            'first_name': 'Subject',
            'birthday': (1990, 1, 1),  # (year, month, day)
            'sex': 1,  # 1=male, 2=female, 0=unknown
            'hand': 1,  # 1=right, 2=left, 3=ambidextrous
        }
        raw.info['subject_info'] = official_subject_info
        print("✓ Successfully added official subject_info format")
        print(f"  Added: {raw.info['subject_info']}")
    except Exception as e:
        print(f"✗ Failed to add official subject_info: {e}")
    
    print("\n3. Testing custom fields in info:")
    try:
        # Try adding custom fields directly to info
        raw.info['participant_metadata'] = participant_metadata
        print("✓ Successfully added participant_metadata field")
    except Exception as e:
        print(f"✗ Failed to add participant_metadata field: {e}")
    
    print("\n4. Testing metadata as object attribute:")
    try:
        # Add as direct attribute to the raw object
        raw.participant_metadata = participant_metadata
        print("✓ Successfully added as object attribute")
        print(f"  Added: {raw.participant_metadata}")
    except Exception as e:
        print(f"✗ Failed to add as object attribute: {e}")
    
    print("\n5. Testing saving and loading with custom metadata:")
    import tempfile
    import os
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.fif', delete=False) as tmp:
            temp_file = tmp.name
        
        # Save with metadata
        raw.save(temp_file, overwrite=True, verbose=False)
        print("✓ Successfully saved raw with metadata")
        
        # Load and check if metadata persists
        loaded_raw = mne.io.read_raw_fif(temp_file, verbose=False)
        
        # Check different metadata locations
        print("  Checking loaded raw:")
        print(f"    subject_info: {loaded_raw.info.get('subject_info', 'Not found')}")
        print(f"    participant_metadata in info: {loaded_raw.info.get('participant_metadata', 'Not found')}")
        print(f"    participant_metadata as attribute: {getattr(loaded_raw, 'participant_metadata', 'Not found')}")
        
        # Clean up
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error during save/load test: {e}")
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("\n6. Testing epochs metadata attribute:")
    try:
        # Create epochs to test the metadata attribute
        events = mne.make_fixed_length_events(raw, duration=0.5)[:5]  # 5 events
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.4, verbose=False)
        
        # Test MNE's built-in metadata system for epochs
        import pandas as pd
        metadata_df = pd.DataFrame({
            'participant_id': ['S_002_F'] * len(epochs),
            'responder': ['yes'] * len(epochs),
            'trial_id': range(len(epochs))
        })
        
        epochs.metadata = metadata_df
        print("✓ Successfully added pandas DataFrame as epochs.metadata")
        print(f"  Shape: {epochs.metadata.shape}")
        print(f"  Columns: {list(epochs.metadata.columns)}")
        
        # Test saving epochs with metadata
        with tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp:
            temp_epochs_file = tmp.name
            
        epochs.save(temp_epochs_file, overwrite=True, verbose=False)
        loaded_epochs = mne.read_epochs(temp_epochs_file, verbose=False)
        
        print("  After loading:")
        if loaded_epochs.metadata is not None:
            print(f"    Metadata preserved: {loaded_epochs.metadata.shape}")
            print(f"    Sample data: {loaded_epochs.metadata.iloc[0].to_dict()}")
        else:
            print("    Metadata lost during save/load")
            
        os.unlink(temp_epochs_file)
        
    except Exception as e:
        print(f"✗ Error testing epochs metadata: {e}")
        if 'temp_epochs_file' in locals() and os.path.exists(temp_epochs_file):
            os.unlink(temp_epochs_file)

if __name__ == "__main__":
    test_mne_metadata_restrictions()