from pathlib import Path
from mne.io import BaseRaw
from .brainvision import BrainVisionLoader
from .curry import CurryLoader
from .neuroscan import NeuroscanLoader
from .fif import FifLoader
from .bdf import BDFLoader
from .eeglab import EEGLABLoader
from .edf import EDFLoader
from .ant import ANTLoader
from collections import defaultdict
import time
#from ..utils.montages import add_standard_montage

# Tracking variables
_format_counts = defaultdict(int)
_last_loaded_file = None


def get_loader_stats() -> dict:
    """Track which formats are being used"""
    return {
        'load_counts': dict(_format_counts),
        'last_loaded': _last_loaded_file,
        'timestamp': time.time()
    }


def load_raw(file_path: str | Path, **kwargs) -> BaseRaw:
    global _last_loaded_file
    path = Path(file_path).expanduser().resolve()
    _last_loaded_file = str(path)

    # Order matters - more specific loaders first
    loaders = [
        BrainVisionLoader,  # .vhdr
        FifLoader,          # .fif, .fiff
        BDFLoader,          # .bdf (Biosemi)
        EEGLABLoader,       # .set (EEGLAB)
        EDFLoader,          # .edf (European Data Format)
        ANTLoader,          # .cnt (ANT Neuro - check first for .cnt)
        NeuroscanLoader,    # .cnt (Neuroscan - fallback for .cnt)
        CurryLoader,        # .dap, .dat, .ceo
    ]

    for loader in loaders:
        if loader.supports_format(path):
            _format_counts[loader.__name__] += 1
            return loader.load(path, **kwargs)

    _format_counts['unsupported'] += 1
    raise ValueError(f"Unsupported format: {path.suffix}",
                    f"Supported formats:\n"
                    f"- BrainVision: .vhdr (requires .eeg/.vmrk)\n"
                    f"- FIFF: .fif, .fiff\n"
                    f"- Biosemi: .bdf\n"
                    f"- EEGLAB: .set (with .fdt)\n"
                    f"- EDF/EDF+: .edf\n"
                    f"- ANT Neuro: .cnt\n"
                    f"- Neuroscan: .cnt\n"
                    f"- Curry: .dap, .dat, .ceo"
                )