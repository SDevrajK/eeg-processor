"""Example of using the heartbeat decorator for long-running EEG processing functions."""

from eeg_processor.utils.performance import with_heartbeat
import mne


# Example 1: Decorating your own functions
@with_heartbeat(interval=5, message="Computing ICA components")
def compute_ica_with_progress(raw, n_components=20):
    """Compute ICA with heartbeat progress indicator."""
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
    ica.fit(raw)
    return ica


# Example 2: Wrapping external MNE functions
# Wrap MNE's slow functions that don't provide progress feedback
wrapped_find_bad_channels = with_heartbeat(
    interval=3, 
    message="Detecting bad channels"
)(mne.preprocessing.find_bad_channels_lof)

wrapped_compute_psd = with_heartbeat(
    interval=2,
    message="Computing power spectral density"
)(mne.time_frequency.psd_array_welch)


# Example 3: Using in processing pipelines
class EEGProcessorWithHeartbeat:
    """Example of integrating heartbeat monitoring in processing stages."""
    
    @with_heartbeat(interval=5, message="Filtering raw data")
    def filter_data(self, raw, l_freq=0.1, h_freq=40):
        """Apply filtering with progress indicator."""
        return raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    @with_heartbeat(interval=10, message="Running ASR artifact removal")
    def run_asr(self, raw, cutoff=20):
        """Run ASR with progress indicator."""
        # This would call your ASR implementation
        # For example:
        # from asrpy import ASR
        # asr = ASR(cutoff=cutoff)
        # return asr.fit_transform(raw)
        pass
    
    @with_heartbeat(interval=3, message="Epoching data")
    def create_epochs(self, raw, events, tmin=-0.2, tmax=0.8):
        """Create epochs with progress indicator."""
        return mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None)


# Example 4: Dynamic wrapping for external functions
def process_with_heartbeat(func, message="Processing", interval=5):
    """Helper to wrap any function with heartbeat monitoring."""
    return with_heartbeat(interval=interval, message=message)(func)


# Usage examples:
if __name__ == "__main__":
    # Load some sample data
    # raw = mne.io.read_raw_fif('sample_data.fif', preload=True)
    
    # Example of wrapping MNE's computationally expensive functions
    # ica = process_with_heartbeat(
    #     mne.preprocessing.ICA(n_components=20).fit,
    #     message="Fitting ICA",
    #     interval=5
    # )(raw)
    
    # Or use the pre-wrapped functions
    # bad_channels = wrapped_find_bad_channels(raw)
    
    print("See the examples above for using the heartbeat decorator with EEG processing functions.")
    print("\nKey benefits:")
    print("- Visual feedback that processing is still active")
    print("- Clean single-line progress with elapsed time")
    print("- Non-intrusive (runs in separate thread)")
    print("- Works with any function (your own or external packages)")