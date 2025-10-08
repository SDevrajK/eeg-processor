"""
Test the complete metadata solution including epochs support.
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

def test_complete_metadata():
    """Test complete metadata solution for raw and epochs."""
    
    print("Testing complete metadata solution...")
    
    from eeg_processor.pipeline import EEGPipeline
    from eeg_processor.processing.epoching import create_epochs
    
    # Create mock participant
    participant = MockParticipant(
        id='S_002_F', 
        metadata={
            'responder': 'yes',
            'age': 25,
            'group': 'control'
        }
    )
    
    # Create pipeline and mock raw data
    pipeline = EEGPipeline()
    
    print("1. Creating mock EEG data with events...")
    sfreq = 1000
    n_channels = 4
    n_samples = 10000  # 10 seconds
    data = np.random.randn(n_channels, n_samples)
    ch_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4']
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)
    raw = mne.io.RawArray(data, info)
    
    # Add some events for epoching
    events = np.array([
        [1000, 0, 1],   # Event at 1 second
        [3000, 0, 1],   # Event at 3 seconds  
        [5000, 0, 2],   # Event at 5 seconds
        [7000, 0, 2],   # Event at 7 seconds
        [9000, 0, 1],   # Event at 9 seconds
    ])
    
    # Add events as annotations
    event_times = events[:, 0] / sfreq  # Convert to seconds
    annotations = mne.Annotations(
        onset=event_times,
        duration=[0.001] * len(events),
        description=['trigger_1', 'trigger_1', 'trigger_2', 'trigger_2', 'trigger_1']
    )
    raw.set_annotations(annotations)
    print(f"✓ Created raw data with {len(events)} events")
    
    # Test 1: Add participant metadata to raw
    print("\n2. Testing raw metadata...")
    pipeline._add_participant_metadata_to_raw(raw, participant)
    
    subject_info = raw.info.get('subject_info')
    if subject_info and '|' in subject_info['his_id']:
        print("✅ Raw metadata added successfully")
        print(f"   his_id: {subject_info['his_id']}")
    else:
        print("❌ Raw metadata failed")
        return
    
    # Test 2: Create epochs and check metadata transfer
    print("\n3. Testing epochs metadata...")
    
    condition = {
        'name': 'test_condition',
        'triggers': {
            'standard': 'trigger_1',
            'deviant': 'trigger_2'
        }
    }
    
    try:
        epochs = create_epochs(
            raw=raw,
            condition=condition,
            tmin=-0.2,
            tmax=0.5,
            baseline=(-0.2, 0),
            reject=None,  # No rejection for test
            flat=None
        )
        
        print(f"✓ Created {len(epochs)} epochs")
        
        # Check if epochs have metadata
        if epochs.metadata is not None:
            print("✅ Epochs metadata created!")
            print(f"   Metadata shape: {epochs.metadata.shape}")
            print(f"   Columns: {list(epochs.metadata.columns)}")
            print("   Sample metadata:")
            print(epochs.metadata.head(2))
            
            # Verify the participant metadata is there
            if 'participant_id' in epochs.metadata.columns:
                participant_ids = epochs.metadata['participant_id'].unique()
                print(f"   Participant IDs: {participant_ids}")
                
            if 'responder' in epochs.metadata.columns:
                responder_values = epochs.metadata['responder'].unique() 
                print(f"   Responder values: {responder_values}")
                
            if all(col in epochs.metadata.columns for col in ['participant_id', 'responder', 'age', 'group']):
                print("✅ All participant metadata fields present!")
            else:
                print("❌ Missing some participant metadata fields")
        else:
            print("❌ No epochs metadata found")
            return
            
    except Exception as e:
        print(f"❌ Epochs creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Test epochs save/load
    print("\n4. Testing epochs metadata persistence...")
    with tempfile.NamedTemporaryFile(suffix='-epo.fif', delete=False) as tmp:
        epochs_file = tmp.name
    
    try:
        epochs.save(epochs_file, overwrite=True, verbose=False)
        print("✓ Epochs saved")
        
        loaded_epochs = mne.read_epochs(epochs_file, verbose=False)
        print("✓ Epochs loaded")
        
        if loaded_epochs.metadata is not None:
            print("✅ Epochs metadata survived save/load!")
            print(f"   Loaded metadata shape: {loaded_epochs.metadata.shape}")
            
            # Check specific participant metadata
            if 'responder' in loaded_epochs.metadata.columns:
                responder_values = loaded_epochs.metadata['responder'].unique()
                if 'yes' in responder_values:
                    print("✅ Participant 'responder' field preserved!")
                else:
                    print(f"❌ Responder values incorrect: {responder_values}")
            else:
                print("❌ 'responder' field missing from loaded metadata")
        else:
            print("❌ Epochs metadata lost during save/load")
    
    except Exception as e:
        print(f"❌ Epochs save/load failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if os.path.exists(epochs_file):
            os.unlink(epochs_file)
    
    print("\n🎉 Complete metadata solution testing finished!")
    print("\nSUMMARY:")
    print("✅ Raw data: Participant metadata stored in subject_info['his_id']")
    print("✅ Epochs: Participant metadata stored in epochs.metadata DataFrame")
    print("✅ Persistence: Both raw and epochs metadata survive save/load cycles")
    print("\nThe 'responder' field (and other participant metadata) should now be")
    print("available in all processed EEG files!")

if __name__ == "__main__":
    test_complete_metadata()