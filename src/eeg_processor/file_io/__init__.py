from pathlib import Path
from mne.io import BaseRaw
from .brainvision import BrainVisionLoader
from .curry import CurryLoader
from .neuroscan import NeuroscanLoader
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

    for loader in [BrainVisionLoader, CurryLoader, NeuroscanLoader]:
        if loader.supports_format(path):
            _format_counts[loader.__name__] += 1
            return loader.load(path, **kwargs)

    _format_counts['unsupported'] += 1
    raise ValueError(f"Unsupported format: {path.suffix}",
                    f"Supported formats:\n"
                    f"- BrainVision: .vhdr (requires .eeg/.vmrk)\n"
                    f"- Curry 7: .dap, .dat, or .ceo\n"
                    f"- Neuroscan: .cnt"
                )