# Standard library
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Union
import yaml

from mne.io import BaseRaw
import matplotlib.pyplot as plt


def filter_data(raw: BaseRaw,
                l_freq: Optional[float] = None,
                h_freq: Optional[float] = None,
                notch: Union[bool, float, List[float], None] = None,
                inplace: bool = False,
                **filter_kwargs) -> BaseRaw:
    """Apply filtering with optional in-place operation"""

    if hasattr(raw, 'preload') and not raw.preload:
        logger.info("Data not preloaded - loading into memory for filtering...")
        if inplace:
            current_data = raw.load_data()
        else:
            current_data = raw.copy().load_data()
    else:
        if inplace:
            current_data = raw
        else:
            current_data = raw.copy()

    # Apply notch filter FIRST
    if notch is not None:
        notch_freqs = _parse_notch_freqs(notch)
        if notch_freqs:  # Only apply if we have frequencies to notch
            logger.info(f"Applying notch filter at frequencies: {notch_freqs}")
            current_data.notch_filter(
                freqs=notch_freqs,
                method='fir',
                phase='zero-double',
                fir_window='hamming',
                verbose=False
            )

    # Apply bandpass filter SECOND
    if l_freq is not None or h_freq is not None:
        current_data.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method='fir',
            phase='zero-double',
            fir_window='hamming',
            verbose=False,
            **filter_kwargs
        )

    return current_data


def _parse_notch_freqs(notch: Union[bool, float, List[float]]) -> List[float]:
    """Convert notch argument to list of frequencies."""
    if isinstance(notch, bool):
        return [50.0] if notch else []
    if isinstance(notch, (float, int)):
        return [float(notch)]
    if isinstance(notch, list):
        return [float(f) for f in notch]
    raise ValueError("notch must be bool, float, or list")


def _plot_filter_effects(
        raw: BaseRaw,
        raw_filt: BaseRaw,
        plot_psd: bool = True,
        plot_raw: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        show: bool = None,
        psd_kwargs: Optional[dict] = None,
        raw_kwargs: Optional[dict] = None
) -> None:
    """
    Generate comparison plots of filtering effects using modern MNE methods.

    Args:
        raw: Unfiltered Raw object
        raw_filt: Filtered Raw object
        plot_psd: Whether to plot PSD comparison
        plot_raw: Whether to plot raw traces comparison
        save_dir: Directory to save plots (None to disable saving)
        show: Whether to display plots (default: True if save_dir is None)
        psd_kwargs: Additional arguments for PSD computation/plotting
        raw_kwargs: Additional arguments for raw plots
    """
    if not (plot_psd or plot_raw):
        return

    if show is None:
        show = save_dir is None

    # Set default kwargs if not provided
    psd_kwargs = psd_kwargs or {'fmax': 100}
    raw_kwargs = raw_kwargs or {
        'start': 0,
        'duration': 10,
        'n_channels': 5,
        'show_scrollbars': False,
        'show_scalebars': False,
    }

    try:
        if plot_psd:
            # Compute PSDs using modern method
            psd_unfiltered = raw.compute_psd(**psd_kwargs)
            psd_filtered = raw_filt.compute_psd(**psd_kwargs)

            # Create figure manually for better control
            fig_psd, ax = plt.subplots(figsize=(10, 5))
            psd_unfiltered.plot(axes=ax, alpha=0.6, show=False)
            psd_filtered.plot(axes=ax, alpha=0.6, show=False)

            ax.set_title('Power Spectral Density\n', fontsize=12)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density (dB)')
            ax.legend(['Before Filtering', 'After Filtering'])

            _save_or_show(fig_psd, save_dir, "filter_psd_comparison.png", show)

        if plot_raw:
            # Create raw traces comparison plot
            # fig_raw, axes = plt.subplots(figsize=(12, 6))
            # raw.plot(
            #     title='Before Filtering',
            #     block=True,
            #     **raw_kwargs
            # )
            #
            # plt.tight_layout()
            # _save_or_show(fig_raw, save_dir, "before_filter.png", show)

            fig_filt, axes = plt.subplots(figsize=(12, 6))
            #raw_filt.set_annotations(None)
            raw_filt.plot(
                title='After Filtering',
                block=True,
                **raw_kwargs
            )
            # plt.tight_layout()
            # _save_or_show(fig_filt, save_dir, "after_filter.png", show)


    except Exception as e:
        import warnings
        warnings.warn(f"Failed to generate plots: {str(e)}")
        if 'fig_psd' in locals():
            plt.close(fig_psd)
        if 'fig_raw' in locals():
            plt.close(fig_raw)
        if 'fig_filt' in locals():
            plt.close(fig_filt)

def _save_or_show(
        fig,
        save_dir: Optional[Union[str, Path]],
        filename: str,
        show: bool = True
) -> None:
    """Handle plot saving/showing with error checking."""
    try:
        if save_dir is not None:
            save_path = Path(save_dir) / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        elif show:
            plt.show(block=True)
        else:
            plt.close(fig)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to save/show plot: {str(e)}")
        plt.close(fig)