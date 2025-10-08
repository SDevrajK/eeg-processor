"""
Utilities for adding participant metadata to all MNE data types.

This module provides functions to extract and add participant metadata
to all supported MNE data types while respecting MNE's restrictions.
"""

import pandas as pd
from typing import Dict, Any, Union, Optional
from loguru import logger

# MNE imports
from mne.io import BaseRaw  
from mne import Epochs, Evoked
from mne.time_frequency import AverageTFR, EpochsTFR, RawTFR, Spectrum

# Try to import additional spectrum types
try:
    from mne.time_frequency.spectrum import EpochsSpectrum
except ImportError:
    EpochsSpectrum = None

def extract_participant_metadata_from_raw(raw: BaseRaw) -> Dict[str, Any]:
    """
    Extract participant metadata from raw data subject_info.
    
    Args:
        raw: MNE Raw object containing participant metadata
        
    Returns:
        Dict containing participant metadata, or empty dict if none found
    """
    try:
        subject_info = raw.info.get('subject_info')
        if not subject_info:
            return {}
            
        his_id = subject_info.get('his_id', '')
        if not his_id or '|' not in his_id:
            return {}
            
        # Parse metadata from his_id
        parts = his_id.split('|')
        participant_id = parts[0]
        metadata = {'participant_id': participant_id}
        
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                metadata[key] = value
                
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to extract participant metadata from raw: {e}")
        return {}


def add_participant_metadata_to_epochs(epochs: Epochs, participant_metadata: Dict[str, Any]) -> None:
    """
    Add participant metadata to epochs using MNE's metadata DataFrame.
    
    Args:
        epochs: MNE Epochs object
        participant_metadata: Dict of participant metadata
    """
    try:
        if not participant_metadata:
            return
            
        n_epochs = len(epochs)
        if n_epochs == 0:
            # Still create metadata for empty epochs
            epochs_metadata = pd.DataFrame({
                **{key: [value] * max(1, n_epochs) for key, value in participant_metadata.items()},
                'epoch_number': list(range(n_epochs)) if n_epochs > 0 else [0]
            })
            if n_epochs == 0:
                epochs_metadata = epochs_metadata.iloc[:0]  # Keep structure but empty
        else:
            epochs_metadata = pd.DataFrame({
                **{key: [value] * n_epochs for key, value in participant_metadata.items()},
                'epoch_number': range(n_epochs)
            })
        
        epochs.metadata = epochs_metadata
        logger.debug(f"Added participant metadata to {n_epochs} epochs")
        
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to epochs: {e}")


def add_participant_metadata_to_evoked(evoked: Evoked, participant_metadata: Dict[str, Any]) -> None:
    """
    Add participant metadata to evoked object using comment field.
    
    Args:
        evoked: MNE Evoked object
        participant_metadata: Dict of participant metadata
    """
    try:
        if not participant_metadata:
            return
            
        # Store metadata in comment field as structured string
        metadata_str = '|'.join([f"{k}={v}" for k, v in participant_metadata.items()])
        
        if evoked.comment:
            evoked.comment = f"{evoked.comment} | PARTICIPANT_METADATA: {metadata_str}"
        else:
            evoked.comment = f"PARTICIPANT_METADATA: {metadata_str}"
            
        logger.debug(f"Added participant metadata to evoked: {list(participant_metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to evoked: {e}")


def add_participant_metadata_to_tfr(tfr: Union[AverageTFR, EpochsTFR, RawTFR], 
                                   participant_metadata: Dict[str, Any]) -> None:
    """
    Add participant metadata to time-frequency objects using comment field.
    
    Args:
        tfr: MNE TFR object (AverageTFR, EpochsTFR, or RawTFR)
        participant_metadata: Dict of participant metadata
    """
    try:
        if not participant_metadata:
            return
            
        # Store metadata in comment field as structured string
        metadata_str = '|'.join([f"{k}={v}" for k, v in participant_metadata.items()])
        
        if hasattr(tfr, 'comment') and tfr.comment:
            tfr.comment = f"{tfr.comment} | PARTICIPANT_METADATA: {metadata_str}"
        else:
            tfr.comment = f"PARTICIPANT_METADATA: {metadata_str}"
            
        logger.debug(f"Added participant metadata to TFR ({type(tfr).__name__}): {list(participant_metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to TFR: {e}")


def add_participant_metadata_to_spectrum(spectrum: Spectrum, 
                                        participant_metadata: Dict[str, Any]) -> None:
    """
    Add participant metadata to spectrum object.
    
    Args:
        spectrum: MNE Spectrum object
        participant_metadata: Dict of participant metadata
    """
    try:
        if not participant_metadata:
            return
            
        # For Spectrum objects, we'll add to the info dict if possible
        # Otherwise use a custom attribute
        try:
            # Try to add to info (may not work due to MNE restrictions)
            metadata_str = '|'.join([f"{k}={v}" for k, v in participant_metadata.items()])
            spectrum.info['description'] = f"PARTICIPANT_METADATA: {metadata_str}"
            logger.debug(f"Added participant metadata to spectrum info")
        except:
            # Fallback: add as custom attribute (won't persist through save/load)
            spectrum._participant_metadata = participant_metadata
            logger.debug(f"Added participant metadata as spectrum attribute: {list(participant_metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to spectrum: {e}")


def add_participant_metadata_to_mne_object(mne_object: Any, 
                                          participant_metadata: Optional[Dict[str, Any]] = None,
                                          source_raw: Optional[BaseRaw] = None) -> None:
    """
    Universal function to add participant metadata to any MNE object.
    
    Args:
        mne_object: Any MNE data object
        participant_metadata: Optional dict of metadata to add
        source_raw: Optional raw object to extract metadata from
        
    Note:
        Either participant_metadata or source_raw must be provided.
        If source_raw is provided, metadata will be extracted from it.
    """
    try:
        # Extract metadata from source_raw if provided
        if participant_metadata is None:
            if source_raw is None:
                logger.warning("Either participant_metadata or source_raw must be provided")
                return
            participant_metadata = extract_participant_metadata_from_raw(source_raw)
        
        if not participant_metadata:
            return
        
        # Route to appropriate function based on object type
        if isinstance(mne_object, Epochs):
            add_participant_metadata_to_epochs(mne_object, participant_metadata)
        elif isinstance(mne_object, Evoked):
            add_participant_metadata_to_evoked(mne_object, participant_metadata)
        elif isinstance(mne_object, (AverageTFR, EpochsTFR, RawTFR)):
            add_participant_metadata_to_tfr(mne_object, participant_metadata)
        elif isinstance(mne_object, Spectrum) or (EpochsSpectrum and isinstance(mne_object, EpochsSpectrum)):
            add_participant_metadata_to_spectrum(mne_object, participant_metadata)
        elif isinstance(mne_object, BaseRaw):
            logger.debug("Raw objects should use pipeline._add_participant_metadata_to_raw")
        else:
            logger.warning(f"Unsupported MNE object type for metadata: {type(mne_object)}")
            
    except Exception as e:
        logger.warning(f"Failed to add participant metadata to MNE object: {e}")


def extract_participant_metadata_from_mne_object(mne_object: Any) -> Dict[str, Any]:
    """
    Extract participant metadata from any MNE object.
    
    Args:
        mne_object: Any MNE data object
        
    Returns:
        Dict containing extracted participant metadata
    """
    try:
        if isinstance(mne_object, BaseRaw):
            return extract_participant_metadata_from_raw(mne_object)
            
        elif isinstance(mne_object, Epochs):
            if mne_object.metadata is not None and 'participant_id' in mne_object.metadata.columns:
                # Extract from first row (all rows should have same participant data)
                row = mne_object.metadata.iloc[0] if len(mne_object.metadata) > 0 else {}
                return {col: row[col] for col in mne_object.metadata.columns 
                       if col not in ['epoch_number']}
        
        elif isinstance(mne_object, Evoked):
            if hasattr(mne_object, 'comment') and mne_object.comment:
                return _parse_metadata_from_comment(mne_object.comment)
                
        elif isinstance(mne_object, (AverageTFR, EpochsTFR, RawTFR)):
            if hasattr(mne_object, 'comment') and mne_object.comment:
                return _parse_metadata_from_comment(mne_object.comment)
                
        elif isinstance(mne_object, Spectrum):
            if hasattr(mne_object, '_participant_metadata'):
                return mne_object._participant_metadata
            elif 'description' in mne_object.info and 'PARTICIPANT_METADATA' in str(mne_object.info['description']):
                return _parse_metadata_from_comment(mne_object.info['description'])
        
        return {}
        
    except Exception as e:
        logger.warning(f"Failed to extract participant metadata from MNE object: {e}")
        return {}


def _parse_metadata_from_comment(comment: str) -> Dict[str, Any]:
    """Parse participant metadata from comment field."""
    try:
        if 'PARTICIPANT_METADATA:' not in comment:
            return {}
            
        # Extract metadata portion
        metadata_part = comment.split('PARTICIPANT_METADATA:')[1].strip()
        if '|' in metadata_part:
            metadata_part = metadata_part.split('|')[0].strip()
        
        # Parse key=value pairs
        metadata = {}
        for part in metadata_part.split('|'):
            if '=' in part:
                key, value = part.split('=', 1)
                metadata[key.strip()] = value.strip()
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Failed to parse metadata from comment: {e}")
        return {}


# Convenience function for pipeline integration
def propagate_participant_metadata(source_object: Any, target_object: Any) -> None:
    """
    Propagate participant metadata from one MNE object to another.
    
    Args:
        source_object: MNE object containing metadata
        target_object: MNE object to receive metadata
    """
    try:
        metadata = extract_participant_metadata_from_mne_object(source_object)
        if metadata:
            add_participant_metadata_to_mne_object(target_object, metadata)
            logger.debug(f"Propagated metadata from {type(source_object).__name__} to {type(target_object).__name__}")
    except Exception as e:
        logger.warning(f"Failed to propagate metadata: {e}")