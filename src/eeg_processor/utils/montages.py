# Create new file: src/eeg_processor/utils/montages.py
from mne import channels
from mne.io import BaseRaw


def add_standard_montage(raw: BaseRaw, montage_name: str = 'standard_1005') -> BaseRaw:
    """
    Apply standard electrode positions

    Args:
        raw: Loaded Raw object
        montage_name: One of MNE's standard montages

    Returns:
        Raw object with montage set
    """
    montage = channels.make_standard_montage(montage_name)
    return raw.set_montage(montage)
