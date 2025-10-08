"""
Direct test of the metadata functionality without file dependencies.
"""

import sys
sys.path.append('/home/sdevrajk/projects/eeg-processor/src')

import numpy as np
import mne
import tempfile
import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class MockParticipant:
    id: str
    metadata: Dict

def test_metadata_function():
    """Test the metadata function directly."""
    
    print("Testing metadata function directly...")
    
    # Import the pipeline to get access to the method
    from eeg_processor.pipeline import EEGPipeline
    
    # Create pipeline object (without config)
    pipeline = EEGPipeline()
    
    # Create mock participant
    participant = MockParticipant(
        id='S_002_F',
        metadata={
            'responder': 'yes',
            'age': 25,
            'group': 'control'
        }
    )
    
    print(f"Mock participant: {participant.id}")
    print(f"Mock metadata: {participant.metadata}")
    
    # Create mock raw data
    print("\nCreating mock raw data...")
    sfreq = 1000
    n_channels = 4
    n_samples = 2000
    data = np.random.randn(n_channels, n_samples)
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info)
    print("✓ Raw data created")
    
    # Test the metadata function
    print("\nTesting _add_participant_metadata_to_raw function...")
    try:
        pipeline._add_participant_metadata_to_raw(raw, participant)
        print("✓ Function executed without errors")
        
        # Check results
        subject_info = raw.info.get('subject_info')
        if subject_info:
            print(f"✓ subject_info created: {subject_info}")
            
            his_id = subject_info['his_id']
            print(f"✓ his_id content: '{his_id}'")
            
            # Parse metadata
            if '|' in his_id:
                parts = his_id.split('|')
                participant_id = parts[0]
                metadata = {}
                
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        metadata[key] = value
                
                print(f"✓ Parsed participant_id: '{participant_id}'")
                print(f"✓ Parsed metadata: {metadata}")
                
                # Validate
                expected_metadata = {
                    'responder': 'yes',
                    'age': '25',  # Note: converted to string
                    'group': 'control'
                }
                
                if participant_id == 'S_002_F' and metadata == expected_metadata:
                    print("✅ Metadata correctly encoded!")
                else:
                    print(f"❌ Metadata mismatch!")
                    print(f"   Expected ID: 'S_002_F', Got: '{participant_id}'")
                    print(f"   Expected metadata: {expected_metadata}")
                    print(f"   Got metadata: {metadata}")
            else:
                print("❌ his_id doesn't contain metadata separator '|'")
        else:
            print("❌ No subject_info found")
            
    except Exception as e:
        print(f"❌ Error in metadata function: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test persistence
    print("\nTesting save/load persistence...")
    with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # Save
        raw.save(temp_file, overwrite=True, verbose=False)
        print("✓ Raw data saved")
        
        # Load
        loaded_raw = mne.io.read_raw_fif(temp_file, verbose=False)
        print("✓ Raw data loaded")
        
        # Check persistence
        loaded_subject_info = loaded_raw.info.get('subject_info')
        if loaded_subject_info:
            loaded_his_id = loaded_subject_info['his_id']
            print(f"✓ Loaded his_id: '{loaded_his_id}'")
            
            # Parse loaded metadata
            if '|' in loaded_his_id:
                parts = loaded_his_id.split('|')
                loaded_participant_id = parts[0]
                loaded_metadata = {}
                
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        loaded_metadata[key] = value
                
                expected_metadata = {
                    'responder': 'yes',
                    'age': '25',
                    'group': 'control'
                }
                
                if loaded_participant_id == 'S_002_F' and loaded_metadata == expected_metadata:
                    print("✅ Metadata persistence verified!")
                else:
                    print("❌ Persistence failed!")
                    print(f"   Expected ID: 'S_002_F', Got: '{loaded_participant_id}'")
                    print(f"   Expected: {expected_metadata}")
                    print(f"   Got: {loaded_metadata}")
            else:
                print("❌ Loaded his_id missing metadata")
        else:
            print("❌ subject_info lost during save/load")
    
    except Exception as e:
        print(f"❌ Error during save/load test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_metadata_function()