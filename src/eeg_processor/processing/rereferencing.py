# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import yaml
import os

import mne
from mne.io import BaseRaw
from typing import List, Optional, Union
import numpy as np

def set_reference(
        raw: BaseRaw,
        method: Union[str, List[str]] = "average",
        exclude: Optional[List[str]] = None,
        inplace: bool = False,
        interpolate_bads: bool = True,
        projection: bool = False,
        verbose: bool = False
) -> BaseRaw:
    """
    Apply rereferencing to raw EEG data with robust exclude handling.
    Preserves existing projections (e.g., for blink correction) while adding new reference.

    Args:
        raw: MNE Raw object.
        method: Reference type. Options:
            - "average": Average reference (default)
            - ["M1", "M2"]: Bipolar mastoid reference
            - "REST": Reference electrode standardization technique
        exclude: Channels to exclude from reference computation.
        interpolate_bads: Interpolate bad channels before rereferencing.
        projection: If True, add reference as projector (preserves original data).
        verbose: Print logging messages.

    Returns:
        Rereferenced Raw object.
    """
    if inplace:
        raw_ref = raw
    else:
        raw_ref = raw.copy()

    # Handle bad channels
    if interpolate_bads and raw_ref.info['bads']:
        raw_ref.interpolate_bads()
        if verbose:
            logger.info(f"Interpolated bad channels: {raw_ref.info['bads']}")

    # Handle excluded channels
    if exclude:
        picks = mne.pick_types(raw_ref.info, meg=False, eeg=True, exclude=exclude)
        ch_names = [raw_ref.ch_names[i] for i in picks]
    else:
        ch_names = None

    # Save existing projections to reapply later
    existing_projs = raw_ref.info['projs']

    # Apply reference schemes
    if method == "average":
        if ch_names:
            raw_ref.set_eeg_reference(ref_channels=ch_names, projection=projection)
        else:
            raw_ref.set_eeg_reference(ref_channels="average", projection=projection)
        if verbose:
            logger.info("Applied average reference.")

    elif isinstance(method, list):
        if not all(ch in raw_ref.ch_names for ch in method):
            raise ValueError("One or more reference channels not found in data.")
        if ch_names and any(ch not in ch_names for ch in method):
            raise ValueError("One or more reference channels are excluded!")
        raw_ref.set_eeg_reference(ref_channels=method, projection=projection)
        if verbose:
            logger.info(f"Applied custom reference: {method}")

    elif method == "REST":
        try:
            # Set up montage if not already present
            if raw_ref.info['dig'] is None:
                raise ValueError(
                    "REST reference requires electrode positions (montage). No digitization found in raw data.")

            # Set up FSAverage data
            mne.datasets.fetch_fsaverage(verbose=False)

            # Create template source space and BEM model
            logger.info("Setting up forward model for REST reference...")
            src = mne.setup_source_space('fsaverage', spacing='oct5', add_dist=False, verbose=False)
            bem = mne.make_bem_model('fsaverage', ico=3, conductivity=[0.3, 0.006, 0.3], verbose=False)
            bem_sol = mne.make_bem_solution(bem, verbose=False)

            # Create forward solution
            forward = mne.make_forward_solution(
                raw_ref.info,
                trans='fsaverage',  # Use template transformation
                src=src,
                bem=bem_sol,
                meg=False,
                eeg=True,
                mindist=5.0,
                verbose=False
            )

            # Apply REST reference with forward model
            raw_ref.set_eeg_reference(ref_channels='REST', forward=forward, projection=projection)
            logger.info("Applied REST reference with template forward model.")

        except ImportError:
            raise ImportError("REST requires mne>=1.0 and numpy>=1.20")

    else:
        raise ValueError(f"Unsupported reference method: {method}")

    # Reapply existing projections (e.g., blink regression projection)
    if existing_projs:
        raw_ref.add_proj(existing_projs, remove_duplicates=True)
        if verbose:
            logger.info("Preserved existing projections.")

    if verbose:
        ref_info = f"Reference: {method} | Excluded: {exclude}"
        logger.info(ref_info)

    return raw_ref


def validate_reference(raw: BaseRaw) -> None:
    """
    Check if referencing was applied correctly.
    Prints the current reference channels and projectors.
    """
    if raw.info['custom_ref_applied']:
        print(f"✓ Custom reference applied: {raw.info['description']}")
    else:
        print("✗ No reference change detected.")

    if raw.info['projs']:
        print(f"Active projectors: {[p['desc'] for p in raw.info['projs']]}")


def plot_reference_effects(
        raw: BaseRaw,
        raw_ref: BaseRaw,
        channels: List[str] = ["Fz", "Cz", "Pz"],
        duration: float = 2.0
) -> None:
    """
    Plot before/after comparison of rereferencing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 6))
    if len(channels) == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        # Original
        raw.plot(
            start=0,
            duration=duration,
            picks=[ch],
            show=False,
            color='b',
            title=f"{ch} (Before=Blue, After=Red)",
            axes=ax
        )
        # Rereferenced
        raw_ref.plot(
            start=0,
            duration=duration,
            picks=[ch],
            show=False,
            color='r',
            axes=ax
        )

    plt.tight_layout()
    plt.show()