"""
Test extending MNE's subject_info field with custom participant metadata.
"""

import mne
import numpy as np
from datetime import date
import tempfile
import os

def test_subject_info_extension():
    """Test adding custom fields to MNE's subject_info."""
    
    print("Testing subject_info extension with custom metadata...")
    
    # Create raw data
    sfreq = 1000
    data = np.random.randn(4, 1000)
    info = mne.create_info(['EEG1', 'EEG2', 'EEG3', 'EEG4'], sfreq, ch_types=['eeg'] * 4)
    raw = mne.io.RawArray(data, info)
    
    # Test 1: Create proper subject_info first
    print("\n1. Creating proper subject_info:")
    try:
        subject_info = {
            'id': 2,
            'his_id': 'S_002_F',
            'last_name': 'F', 
            'first_name': 'S_002',
            'birthday': date(1990, 1, 1),
            'sex': 2,  # female
        }
        raw.info['subject_info'] = subject_info
        print("✓ Basic subject_info created successfully")
        print(f"  Current subject_info: {raw.info['subject_info']}")
        
    except Exception as e:
        print(f"✗ Failed to create basic subject_info: {e}")
        return
    
    # Test 2: Try to add custom fields to existing subject_info
    print("\n2. Adding custom fields to subject_info:")
    try:
        # Get the current subject_info object
        current_subject_info = raw.info['subject_info']
        print(f"  Subject_info type: {type(current_subject_info)}")
        print(f"  Subject_info dir: {[attr for attr in dir(current_subject_info) if not attr.startswith('_')]}")
        
        # Try different approaches to add custom data
        
        # Approach 2a: Direct attribute setting
        try:
            current_subject_info.responder = 'yes'
            print("✓ Added 'responder' as direct attribute")
        except Exception as e:
            print(f"✗ Direct attribute failed: {e}")
        
        # Approach 2b: Check if it has a dict-like interface
        try:
            if hasattr(current_subject_info, '__setitem__'):
                current_subject_info['responder'] = 'yes'
                print("✓ Added 'responder' as dict item")
            else:
                print("✗ Subject_info doesn't support dict-like access")
        except Exception as e:
            print(f"✗ Dict-like access failed: {e}")
            
        # Approach 2c: Check if we can modify the underlying dict
        try:
            if hasattr(current_subject_info, '__dict__'):
                current_subject_info.__dict__['responder'] = 'yes'
                current_subject_info.__dict__['age'] = 25
                current_subject_info.__dict__['group'] = 'control'
                print("✓ Added custom fields to __dict__")
                print(f"  Updated subject_info: {current_subject_info}")
            else:
                print("✗ No __dict__ attribute")
        except Exception as e:
            print(f"✗ __dict__ modification failed: {e}")
        
    except Exception as e:
        print(f"✗ Failed to access subject_info: {e}")
    
    # Test 3: Try creating a custom subject_info-like object
    print("\n3. Testing custom subject_info object:")
    try:
        # Try creating a dict with both standard and custom fields
        extended_subject_info = {
            'id': 2,
            'his_id': 'S_002_F',
            'last_name': 'F',
            'first_name': 'S_002', 
            'birthday': date(1990, 1, 1),
            'sex': 2,
            # Custom fields
            'responder': 'yes',
            'age': 25,
            'group': 'control'
        }
        
        raw.info['subject_info'] = extended_subject_info
        print("✗ This should fail - testing if MNE accepts extra fields in dict")
        
    except Exception as e:
        print(f"✓ Expected failure: {e}")
        
    # Test 4: Save and load to see what persists
    print("\n4. Testing save/load persistence:")
    try:
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file = tmp.name
            
        raw.save(temp_file, overwrite=True, verbose=False)
        loaded_raw = mne.io.read_raw_fif(temp_file, verbose=False)
        
        print("✓ Save/load successful")
        print(f"  Loaded subject_info: {loaded_raw.info['subject_info']}")
        
        # Check if custom attributes survived
        loaded_subject_info = loaded_raw.info['subject_info']
        if hasattr(loaded_subject_info, 'responder'):
            print(f"  Custom 'responder' field: {loaded_subject_info.responder}")
        else:
            print("  Custom 'responder' field: Not found")
            
        if hasattr(loaded_subject_info, '__dict__'):
            custom_fields = {k: v for k, v in loaded_subject_info.__dict__.items() 
                           if k not in ['id', 'his_id', 'last_name', 'first_name', 'birthday', 'sex', 'hand']}
            print(f"  Custom fields in __dict__: {custom_fields}")
        
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Test 5: Alternative - use his_id field creatively
    print("\n5. Testing creative use of existing fields:")
    try:
        # Store metadata in his_id field with structured format
        metadata_string = f"S_002_F|responder=yes|age=25|group=control"
        
        new_raw = mne.io.RawArray(data, info)  # Fresh raw object
        subject_info_with_metadata = {
            'id': 2,
            'his_id': metadata_string,  # Pack metadata here
            'last_name': 'F',
            'first_name': 'S_002',
            'birthday': date(1990, 1, 1),
            'sex': 2,
        }
        
        new_raw.info['subject_info'] = subject_info_with_metadata
        print("✓ Packed metadata into his_id field")
        print(f"  his_id content: {new_raw.info['subject_info']['his_id']}")
        
        # Test save/load
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file2 = tmp.name
            
        new_raw.save(temp_file2, overwrite=True, verbose=False)
        loaded_raw2 = mne.io.read_raw_fif(temp_file2, verbose=False)
        
        loaded_his_id = loaded_raw2.info['subject_info']['his_id']
        print(f"  Loaded his_id: {loaded_his_id}")
        
        # Parse metadata back out
        if '|' in loaded_his_id:
            parts = loaded_his_id.split('|')
            participant_id = parts[0]
            metadata = {}
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    metadata[key] = value
            
            print(f"  Parsed participant_id: {participant_id}")
            print(f"  Parsed metadata: {metadata}")
        
        os.unlink(temp_file2)
        
    except Exception as e:
        print(f"✗ his_id approach failed: {e}")

if __name__ == "__main__":
    test_subject_info_extension()