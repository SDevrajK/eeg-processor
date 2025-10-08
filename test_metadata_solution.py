"""
Test the implemented metadata solution.
"""

import tempfile
import os
import sys
sys.path.append('/home/sdevrajk/projects/eeg-processor/src')

from eeg_processor.pipeline import EEGPipeline

def test_metadata_solution():
    """Test that participant metadata gets saved to MNE objects."""
    
    print("Testing implemented metadata solution...")
    
    # Create a minimal test configuration
    test_config = {
        'study': {
            'name': 'test_metadata',
            'dataset': 'test'
        },
        'paths': {
            'raw_data': '/tmp',
            'results': '/tmp/test_results',
            'file_extension': '.fif'
        },
        'participants': {
            'TEST_001_F': {
                'file': 'test_001.fif',
                'responder': 'yes',
                'age': 25,
                'group': 'control'
            },
            'TEST_002_M': {
                'file': 'test_002.fif',
                'responder': 'no',
                'age': 30,
                'group': 'treatment'
            }
        },
        'processing': [
            {'filter': {'l_freq': 1, 'h_freq': 40}}
        ],
        'conditions': [
            {
                'name': 'test_condition',
                'triggers': {'test': 1}
            }
        ]
    }
    
    # Test 1: Configuration loading and participant metadata extraction
    print("\n1. Testing configuration loading:")
    try:
        pipeline = EEGPipeline(test_config)
        participants = pipeline.participant_handler.participants
        
        print(f"✓ Loaded {len(participants)} participants")
        for p in participants:
            print(f"  {p.id}: {p.metadata}")
            
        # Test our new metadata function
        print("\n2. Testing metadata extraction function:")
        import numpy as np
        import mne
        from mne.io import BaseRaw
        
        # Create mock raw data
        sfreq = 1000
        data = np.random.randn(4, 2000)
        info = mne.create_info(['EEG1', 'EEG2', 'EEG3', 'EEG4'], sfreq, ch_types=['eeg']*4)
        raw = mne.io.RawArray(data, info)
        
        # Test the metadata function
        participant = participants[0]  # TEST_001_F
        pipeline._add_participant_metadata_to_raw(raw, participant)
        
        print("✓ Metadata function executed without errors")
        
        # Check if subject_info was created
        subject_info = raw.info.get('subject_info')
        if subject_info:
            print(f"  ✓ subject_info created: {subject_info}")
            his_id = subject_info['his_id']
            print(f"  ✓ his_id content: {his_id}")
            
            # Parse the metadata back
            if '|' in his_id:
                parts = his_id.split('|')
                parsed_id = parts[0]
                metadata = {}
                for part in parts[1:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        metadata[key] = value
                
                print(f"  ✓ Parsed participant_id: {parsed_id}")
                print(f"  ✓ Parsed metadata: {metadata}")
                
                # Verify our test data
                expected = {'responder': 'yes', 'age': '25', 'group': 'control'}
                if metadata == expected:
                    print("  ✅ Metadata correctly stored and retrieved!")
                else:
                    print(f"  ❌ Metadata mismatch. Expected: {expected}, Got: {metadata}")
            
        else:
            print("  ❌ No subject_info found")
        
        print("\n3. Testing save/load persistence:")
        with tempfile.NamedTemporaryFile(suffix='_raw.fif', delete=False) as tmp:
            temp_file = tmp.name
            
        try:
            raw.save(temp_file, overwrite=True, verbose=False)
            loaded_raw = mne.io.read_raw_fif(temp_file, verbose=False)
            
            loaded_subject_info = loaded_raw.info.get('subject_info')
            if loaded_subject_info:
                loaded_his_id = loaded_subject_info['his_id']
                print(f"  ✓ Metadata survived save/load: {loaded_his_id}")
                
                # Parse again
                if '|' in loaded_his_id:
                    parts = loaded_his_id.split('|')
                    loaded_metadata = {}
                    for part in parts[1:]:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            loaded_metadata[key] = value
                    
                    expected = {'responder': 'yes', 'age': '25', 'group': 'control'}
                    if loaded_metadata == expected:
                        print("  ✅ Metadata persistence verified!")
                    else:
                        print(f"  ❌ Persistence failed. Expected: {expected}, Got: {loaded_metadata}")
            else:
                print("  ❌ subject_info lost during save/load")
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_solution()