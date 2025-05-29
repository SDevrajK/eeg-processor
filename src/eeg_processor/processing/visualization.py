from typing import Optional
from mne.io import BaseRaw
from mne import Epochs
import matplotlib.pyplot as plt


def plot_stage(data, stage_name: str, **kwargs):
    """Unified plotting interface"""
    fig = None

    if isinstance(data, BaseRaw):
        fig = _plot_raw(data, title=f"Raw: {stage_name}", **kwargs)
    elif isinstance(data, Epochs):
        fig = _plot_epochs(data, title=f"Epochs: {stage_name}", **kwargs)

    if fig:
        plt.show(block=False)
    return fig


def _plot_raw(raw: BaseRaw, **kwargs):
    defaults = dict(
        n_channels=4,
        duration=5.0,
        scalings="auto",
        clipping=None,
        block=True
    )
    return raw.plot(**{**defaults, **kwargs})


def _plot_epochs(epochs: Epochs, **kwargs):
    defaults = dict(
        n_epochs=5,
        n_channels=8,
        scalings="auto",
        block=True
    )
    return epochs.plot(**{**defaults, **kwargs})