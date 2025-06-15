# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import yaml

from mne import create_info
from mne.io import BaseRaw, RawArray

def compute_eog_channels(
        raw: BaseRaw,
        heog_pair: Optional[tuple] = None,
        veog_pair: Optional[tuple] = None,
        ch_names: dict = {'heog': 'HEOG', 'veog': 'VEOG'},
        inplace: bool = False,
        overwrite: bool = False
) -> BaseRaw:
    """
    Compute HEOG/VEOG from electrode pairs.

    Args:
        raw: Raw data containing EOG electrodes
        heog_pair: Tuple of (left, right) electrodes for HEOG (e.g. ('E1', 'E2'))
        veog_pair: Tuple of (upper, lower) electrodes for VEOG (e.g. ('Fp1', 'Fp2'))
        ch_names: Dictionary for output channel names
        inplace: Whether to modify input object or create copy
        overwrite: Replace existing channels if they exist

    Returns:
        Raw object with added HEOG/VEOG channels
    """
    if inplace:
        current_raw = raw
    else:
        current_raw = raw.copy()

    # Load data if not already loaded (required for adding channels)
    data_was_loaded = current_raw.preload
    if not data_was_loaded:
        logger.debug("Loading raw data for EOG computation")
        current_raw.load_data()

    existing_chs = set(current_raw.ch_names)

    # Validate electrodes exist
    def _validate_pair(pair, name):
        if pair is None:
            return None
        missing = [ch for ch in pair if ch not in existing_chs]
        if missing:
            raise ValueError(f"Missing {name} electrodes: {missing}")
        return pair

    heog_pair = _validate_pair(heog_pair, 'HEOG')
    veog_pair = _validate_pair(veog_pair, 'VEOG')

    # Compute derivatives
    if heog_pair:
        heog_data = current_raw.get_data(picks=[heog_pair[0]]) - current_raw.get_data(picks=[heog_pair[1]])
        _add_channel(current_raw, heog_data, ch_names['heog'], 'eog', overwrite)

    if veog_pair:
        veog_data = current_raw.get_data(picks=[veog_pair[0]]) - current_raw.get_data(picks=[veog_pair[1]])
        _add_channel(current_raw, veog_data, ch_names['veog'], 'eog', overwrite)

    return current_raw


def _add_channel(raw, data, ch_name, ch_type, overwrite):
    """Helper to safely add channels"""
    if ch_name in raw.ch_names and not overwrite:
        raise ValueError(f"Channel {ch_name} already exists. Set overwrite=True.")

    info = create_info([ch_name], raw.info['sfreq'], [ch_type])
    new_raw = RawArray(data, info)
    raw.add_channels([new_raw], force_update_info=True)
