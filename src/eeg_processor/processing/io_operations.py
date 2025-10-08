"""
I/O operations for the EEG processing pipeline.

This module provides stages for saving and loading processed data.
"""

from pathlib import Path
from typing import Optional, Union
from loguru import logger
from mne.io import BaseRaw
from mne import Epochs


def save_raw(data: Union[BaseRaw, Epochs], 
             suffix: str = "", 
             format: str = "fif",
             output_dir: Optional[str] = None,
             **kwargs) -> Union[BaseRaw, Epochs]:
    """
    Save raw or epochs data to disk with optional suffix.
    
    This stage is useful for:
    - Saving intermediate processing results
    - Creating reference segments for later use
    - Exporting data in different formats
    
    Args:
        data: Raw or Epochs data to save
        suffix: String to append to filename before extension
        format: Output format ('fif', 'edf', 'bdf', 'gdf')
        output_dir: Custom output directory (uses results_dir if not specified)
        **kwargs: Additional parameters passed to save functions
        
    Returns:
        Input data unchanged
        
    Example config:
        - save_raw:
            suffix: "_clean_segment"
            format: "fif"
    """
    # Determine output path
    if hasattr(data, 'filenames') and data.filenames:
        original_path = Path(data.filenames[0])
    else:
        # Fallback for data without filenames
        original_path = Path("unknown_file.fif")
        logger.warning("No filename found in data, using default name")
    
    # Use custom output dir or default to parent + suffix
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Default to same directory as original file
        output_path = original_path.parent
    
    # Create output directory if needed
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Build output filename
    stem = original_path.stem.replace('.fif', '')  # Remove any existing .fif
    output_file = output_path / f"{stem}{suffix}.{format}"
    
    # Save based on data type and format
    logger.info(f"Saving data to {output_file}")
    
    try:
        if isinstance(data, BaseRaw):
            if format == "fif":
                data.save(output_file, overwrite=True, **kwargs)
            elif format in ["edf", "bdf", "gdf"]:
                # These formats are supported via export
                data.export(output_file, overwrite=True, **kwargs)
            else:
                raise ValueError(f"Unsupported format for Raw data: {format}")
                
        elif isinstance(data, Epochs):
            if format == "fif":
                data.save(output_file, overwrite=True, **kwargs)
            else:
                raise ValueError(f"Epochs can only be saved in FIF format, got: {format}")
        else:
            raise TypeError(f"Cannot save data of type {type(data)}")
            
        logger.success(f"Successfully saved {type(data).__name__} to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise
    
    # Return data unchanged
    return data


def crop_participant_segment(raw: BaseRaw, 
                            use_metadata: bool = True,
                            tmin: Optional[float] = None,
                            tmax: Optional[float] = None,
                            participant_info: Optional[dict] = None,
                            **kwargs) -> BaseRaw:
    """
    Crop raw data using participant-specific time ranges from metadata.
    
    This stage reads time ranges from participant metadata to extract
    specific segments like clean baselines for each participant.
    
    Args:
        raw: Raw EEG data
        use_metadata: Whether to read times from participant metadata
        tmin: Default start time if not in metadata
        tmax: Default end time if not in metadata
        participant_info: Participant metadata (provided by pipeline)
        **kwargs: Additional parameters
        
    Returns:
        Cropped raw data
        
    Example config:
        - crop_participant_segment:
            use_metadata: true
            
    With participant metadata:
        P_001:
          file: "subject01.vhdr"
          clean_segment:
            tmin: 30
            tmax: 120
    """
    if use_metadata and participant_info:
        # Extract clean segment times from participant metadata
        if 'clean_segment' in participant_info:
            segment_info = participant_info['clean_segment']
            tmin = segment_info.get('tmin', tmin)
            tmax = segment_info.get('tmax', tmax)
            logger.info(f"Using participant-specific segment: {tmin}s to {tmax}s")
        else:
            logger.warning("No clean_segment info in participant metadata, using defaults")
    
    if tmin is None or tmax is None:
        raise ValueError("tmin and tmax must be provided either in metadata or as parameters")
    
    # Validate times
    data_duration = raw.times[-1]
    if tmin < 0:
        logger.warning(f"tmin ({tmin}) is negative, setting to 0")
        tmin = 0
    if tmax > data_duration:
        logger.warning(f"tmax ({tmax}) exceeds data duration ({data_duration:.1f}), adjusting")
        tmax = data_duration
    if tmin >= tmax:
        raise ValueError(f"Invalid time range: tmin ({tmin}) >= tmax ({tmax})")
    
    # Perform the crop
    logger.info(f"Cropping data from {tmin}s to {tmax}s (duration: {tmax-tmin}s)")
    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    return cropped