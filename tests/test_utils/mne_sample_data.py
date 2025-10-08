"""
MNE Sample Dataset Management for Validation Testing

This module handles downloading and managing MNE's sample dataset for validation
testing against established MNE examples. The sample dataset is used as ground
truth for validating EMCP and other processing methods.

The dataset is approximately 1.5GB and will be downloaded to:
    tests/test_data/mne_sample/

This directory is excluded from git via .gitignore to avoid repository bloat.
"""

import os
from pathlib import Path
from typing import Optional, Union
import tempfile
import shutil
from loguru import logger

# Import MNE with error handling for optional dependency
try:
    import mne
    from mne.datasets import sample
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    logger.warning("MNE not available - sample data download disabled")


class MNESampleDataManager:
    """
    Manages MNE sample dataset download and access for validation testing.
    
    The sample dataset contains:
    - Raw EEG/MEG data with realistic artifacts
    - EOG channels for blink artifact validation
    - Established ground truth from MNE tutorials
    """
    
    def __init__(self, test_data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize sample data manager.
        
        Args:
            test_data_dir: Base test data directory. If None, uses tests/test_data/
        """
        if test_data_dir is None:
            # Default to tests/test_data/ relative to this file
            self.test_data_dir = Path(__file__).parent.parent / "test_data"
        else:
            self.test_data_dir = Path(test_data_dir)
            
        self.mne_sample_dir = self.test_data_dir / "mne_sample"
        self.mne_sample_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"MNE sample data directory: {self.mne_sample_dir}")
        
    def is_available(self) -> bool:
        """Check if MNE sample data is available locally."""
        if not MNE_AVAILABLE:
            return False
            
        sample_file = self.get_sample_raw_file()
        return sample_file is not None and sample_file.exists()
    
    def get_sample_raw_file(self) -> Optional[Path]:
        """
        Get path to sample raw file.
        
        Returns:
            Path to sample_audvis_raw.fif if it exists, None otherwise
        """
        if not MNE_AVAILABLE:
            return None
            
        # Check if we have it in our test directory
        local_file = self.mne_sample_dir / "sample_audvis_raw.fif"
        if local_file.exists():
            return local_file
            
        # Check MNE's default data path
        try:
            data_path = sample.data_path(download=False)
            mne_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
            if mne_file.exists():
                return mne_file
        except Exception:
            pass
            
        return None
    
    def download_sample_data(self, force: bool = False) -> bool:
        """
        Download MNE sample dataset if not already available.
        
        Args:
            force: If True, download even if data already exists
            
        Returns:
            True if data is available after download, False otherwise
        """
        if not MNE_AVAILABLE:
            logger.error("MNE not available - cannot download sample data")
            return False
            
        if not force and self.is_available():
            logger.info("MNE sample data already available")
            return True
            
        logger.info("Downloading MNE sample dataset (~1.5GB)...")
        logger.info("This is a one-time download for validation testing")
        
        try:
            # Download to MNE's default location
            data_path = sample.data_path(download=True)
            source_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
            
            if source_file.exists():
                # Copy to our test directory for easier access
                dest_file = self.mne_sample_dir / "sample_audvis_raw.fif"
                logger.info(f"Copying sample data to {dest_file}")
                shutil.copy2(source_file, dest_file)
                
                logger.success("MNE sample data downloaded successfully")
                return True
            else:
                logger.error("Sample data download failed - file not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to download MNE sample data: {e}")
            return False
    
    def get_sample_raw(self, preload: bool = True):
        """
        Load MNE sample raw data for testing.
        
        Args:
            preload: Whether to preload data into memory
            
        Returns:
            MNE Raw object or None if not available
        """
        if not MNE_AVAILABLE:
            logger.error("MNE not available")
            return None
            
        sample_file = self.get_sample_raw_file()
        if sample_file is None:
            logger.error("MNE sample data not available - run download_sample_data() first")
            return None
            
        try:
            logger.info(f"Loading MNE sample data from {sample_file}")
            raw = mne.io.read_raw_fif(sample_file, preload=preload, verbose=False)
            
            # Ensure we have EOG channels for EMCP testing
            eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch]
            if not eog_channels:
                logger.warning("No EOG channels found in sample data")
            else:
                logger.info(f"Found EOG channels: {eog_channels}")
                
            return raw
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return None
    
    def get_data_info(self) -> dict:
        """
        Get information about available sample data.
        
        Returns:
            Dictionary with data availability and characteristics
        """
        info = {
            'mne_available': MNE_AVAILABLE,
            'data_available': self.is_available(),
            'sample_file': None,
            'file_size_mb': None,
            'eog_channels': [],
            'eeg_channels': [],
            'sampling_rate': None
        }
        
        sample_file = self.get_sample_raw_file()
        if sample_file and sample_file.exists():
            info['sample_file'] = str(sample_file)
            info['file_size_mb'] = round(sample_file.stat().st_size / (1024 * 1024), 1)
            
            # Get channel information if MNE is available
            if MNE_AVAILABLE:
                try:
                    raw = mne.io.read_raw_fif(sample_file, preload=False, verbose=False)
                    info['eog_channels'] = [ch for ch in raw.ch_names if 'EOG' in ch]
                    info['eeg_channels'] = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == 'eeg']
                    info['sampling_rate'] = raw.info['sfreq']
                except Exception as e:
                    logger.warning(f"Could not read sample file info: {e}")
        
        return info


# Global instance for easy access
_sample_manager = None

def get_sample_manager() -> MNESampleDataManager:
    """Get global MNE sample data manager instance."""
    global _sample_manager
    if _sample_manager is None:
        _sample_manager = MNESampleDataManager()
    return _sample_manager

def ensure_sample_data_available() -> bool:
    """
    Ensure MNE sample data is available for testing.
    
    Returns:
        True if data is available, False otherwise
    """
    manager = get_sample_manager()
    
    if manager.is_available():
        return True
        
    logger.info("MNE sample data not found - attempting download...")
    return manager.download_sample_data()

def get_sample_raw(preload: bool = True):
    """
    Get MNE sample raw data for validation testing.
    
    Args:
        preload: Whether to preload data into memory
        
    Returns:
        MNE Raw object or None if not available
    """
    manager = get_sample_manager()
    return manager.get_sample_raw(preload=preload)

def print_sample_data_info():
    """Print information about MNE sample data availability."""
    manager = get_sample_manager()
    info = manager.get_data_info()
    
    print("\n=== MNE Sample Data Info ===")
    print(f"MNE Available: {info['mne_available']}")
    print(f"Data Available: {info['data_available']}")
    
    if info['sample_file']:
        print(f"Sample File: {info['sample_file']}")
        print(f"File Size: {info['file_size_mb']} MB")
        print(f"Sampling Rate: {info['sampling_rate']} Hz")
        print(f"EOG Channels: {info['eog_channels']}")
        print(f"EEG Channels: {len(info['eeg_channels'])} channels")
    else:
        print("Sample file not found")
        print("\nTo download sample data:")
        print("  python -c 'from tests.test_utils.mne_sample_data import ensure_sample_data_available; ensure_sample_data_available()'")
    
    print("=" * 30)


if __name__ == "__main__":
    # CLI interface for downloading sample data
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        manager = get_sample_manager()
        success = manager.download_sample_data(force="--force" in sys.argv)
        sys.exit(0 if success else 1)
    else:
        print_sample_data_info()