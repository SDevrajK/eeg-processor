"""
Test script to find proper MNE-compatible ways to store participant metadata.
"""

import mne
import numpy as np
import pandas as pd
from datetime import date

def test_mne_metadata_proper():
    """Test proper MNE approaches for participant metadata."""
    
    print("Testing proper MNE metadata approaches...")
    
    # Create a simple raw object
    sfreq = 1000
    n_channels = 4
    n_samples = 1000
    data = np.random.randn(n_channels, n_samples)
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    ch_types = ['eeg'] * n_channels
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    print("\n1. Testing proper subject_info format:")
    try:
        from datetime import date
        # Use proper MNE subject_info format
        subject_info = {
            'id': 2,  # Integer ID
            'his_id': 'S_002_F',  # String identifier
            'last_name': 'F',
            'first_name': 'S_002',
            'birthday': date(1990, 1, 1),  # Proper date object
            'sex': 2,  # 2=female (from 'S_002_F')
        }
        raw.info['subject_info'] = subject_info
        print("✓ Successfully added proper subject_info")
        print(f"  Added: {raw.info['subject_info']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n2. Testing temp field for custom metadata:")
    try:
        participant_metadata = {
            'responder': 'yes',
            'age': 25,
            'group': 'control'
        }
        raw.info['temp'] = participant_metadata
        print("✓ Successfully added to temp field")
        print(f"  Added: {raw.info['temp']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n3. Testing save/load with temp field:")
    import tempfile
    import os
    
    try:
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file = tmp.name
        
        raw.save(temp_file, overwrite=True, verbose=False)
        loaded_raw = mne.io.read_raw_fif(temp_file, verbose=False)
        
        print("  After loading:")
        print(f"    subject_info: {loaded_raw.info.get('subject_info')}")
        print(f"    temp field: {loaded_raw.info.get('temp', 'Not found')}")
        
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n4. Testing epochs metadata (proper MNE approach):")
    try:
        # Create events and epochs
        events = mne.make_fixed_length_events(raw, duration=0.5)[:5]
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.4, baseline=(0, 0), verbose=False)
        
        # Create proper metadata DataFrame
        metadata_df = pd.DataFrame({
            'participant_id': ['S_002_F'] * len(epochs),
            'responder': ['yes'] * len(epochs),
            'age': [25] * len(epochs),
            'group': ['control'] * len(epochs),
            'trial_id': range(len(epochs))
        })
        
        epochs.metadata = metadata_df
        print("✓ Successfully added epochs metadata")
        print(f"  Shape: {epochs.metadata.shape}")
        print(f"  Columns: {list(epochs.metadata.columns)}")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp:
            temp_epochs_file = tmp.name
            
        epochs.save(temp_epochs_file, overwrite=True, verbose=False)
        loaded_epochs = mne.read_epochs(temp_epochs_file, verbose=False)
        
        print("  After loading epochs:")
        if loaded_epochs.metadata is not None:
            print(f"    Metadata preserved: {loaded_epochs.metadata.shape}")
            print(f"    Sample: {loaded_epochs.metadata.iloc[0].to_dict()}")
        else:
            print("    Metadata lost")
            
        os.unlink(temp_epochs_file)
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n5. Testing custom attribute preservation approaches:")
    
    # Test storing in description field
    try:
        raw.info['description'] = f"participant_id=S_002_F;responder=yes;age=25;group=control"
        print("✓ Added metadata to description field")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file2 = tmp.name
        raw.save(temp_file2, overwrite=True, verbose=False)
        loaded_raw2 = mne.io.read_raw_fif(temp_file2, verbose=False)
        print(f"  Description after load: {loaded_raw2.info.get('description', 'Not found')}")
        os.unlink(temp_file2)
        
    except Exception as e:
        print(f"✗ Description approach failed: {e}")
    
    print("\n6. Testing annotations approach:")
    try:
        # Store metadata as annotations (creative approach)
        from mne import Annotations
        
        # Create an annotation with metadata
        metadata_str = "responder=yes,age=25,group=control"
        annotations = Annotations(
            onset=[0], 
            duration=[0.001], 
            description=[f"PARTICIPANT_METADATA: {metadata_str}"]
        )
        raw.set_annotations(annotations)
        print("✓ Added metadata as annotations")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file3 = tmp.name
        raw.save(temp_file3, overwrite=True, verbose=False)
        loaded_raw3 = mne.io.read_raw_fif(temp_file3, verbose=False)
        
        if loaded_raw3.annotations is not None and len(loaded_raw3.annotations) > 0:
            print(f"  Annotations preserved: {loaded_raw3.annotations.description}")
        else:
            print("  Annotations lost")
            
        os.unlink(temp_file3)
        
    except Exception as e:
        print(f"✗ Annotations approach failed: {e}")
        
    print("\nSUMMARY:")
    print("- Direct info modification: ✗ (MNE blocks custom keys)")
    print("- temp field: ✗ (explicitly warned as non-persistent)")  
    print("- subject_info: ✓ (but limited to MNE's predefined fields)")
    print("- epochs.metadata: ✓ (best for epochs, but only works for epochs)")
    print("- description field: ? (need to verify)")
    print("- annotations: ? (creative but may be preserved)")

if __name__ == "__main__":
    test_mne_metadata_proper()