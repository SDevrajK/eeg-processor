"""Performance optimization utilities for EEG Processor."""

import time
import psutil
import gc
import functools
import cProfile
import pstats
import threading
import sys
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
from loguru import logger


class PerformanceMonitor:
    """Monitor and optimize performance during EEG processing."""
    
    def __init__(self, memory_limit: float = 0.8, enable_profiling: bool = False):
        """Initialize performance monitor.
        
        Args:
            memory_limit: Maximum memory usage as fraction of total (0.8 = 80%)
            enable_profiling: Whether to enable CPU profiling
        """
        self.memory_limit = memory_limit
        self.enable_profiling = enable_profiling
        self.profiler = None
        self.start_time = None
        self.memory_snapshots = []
        self.performance_stats = {}
        
        if enable_profiling:
            self.profiler = cProfile.Profile()
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        if self.profiler:
            self.profiler.enable()
        
        # Initial memory snapshot
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available,
            'stage': 'start'
        })
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring and return results."""
        if self.profiler:
            self.profiler.disable()
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        # Final memory snapshot
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available,
            'stage': 'end'
        })
        
        self.performance_stats = {
            'total_time': total_time,
            'peak_memory': max(s['memory_percent'] for s in self.memory_snapshots),
            'memory_snapshots': self.memory_snapshots,
            'profiling_enabled': self.profiler is not None
        }
        
        logger.info(f"Performance monitoring completed: {total_time:.2f}s")
        return self.performance_stats
    
    def checkpoint(self, stage_name: str):
        """Create a performance checkpoint."""
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available,
            'stage': stage_name
        })
        
        # Check memory pressure
        current_memory = psutil.virtual_memory().percent / 100
        if current_memory > self.memory_limit:
            logger.warning(f"Memory usage high: {current_memory:.1%} at stage '{stage_name}'")
            self.force_garbage_collection()
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        logger.debug("Forcing garbage collection")
        gc.collect()
    
    def get_profile_stats(self, output_path: Optional[Path] = None) -> Optional[pstats.Stats]:
        """Get profiling statistics."""
        if not self.profiler:
            return None
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        if output_path:
            stats.dump_stats(str(output_path))
        
        return stats
    
    def estimate_processing_time(self, data_size: int, participant_count: int) -> float:
        """Estimate processing time based on data characteristics."""
        # Simple heuristic based on data size
        base_time_per_mb = 2.0  # seconds per MB
        size_mb = data_size / (1024 * 1024)
        estimated_time = base_time_per_mb * size_mb * participant_count
        
        return estimated_time
    
    def suggest_optimizations(self) -> Dict[str, str]:
        """Suggest performance optimizations based on monitoring data."""
        suggestions = {}
        
        if not self.memory_snapshots:
            return suggestions
        
        peak_memory = max(s['memory_percent'] for s in self.memory_snapshots)
        
        if peak_memory > 90:
            suggestions['memory'] = "Consider processing fewer participants simultaneously or reducing epoch length"
        elif peak_memory > 80:
            suggestions['memory'] = "Memory usage is high, consider enabling chunk processing"
        
        if self.performance_stats.get('total_time', 0) > 3600:  # 1 hour
            suggestions['time'] = "Processing is slow, consider parallel processing or data reduction"
        
        return suggestions


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def estimate_memory_usage(raw_data) -> float:
        """Estimate memory usage for raw data object."""
        try:
            # Get data shape and dtype
            if hasattr(raw_data, 'get_data'):
                # Estimate without loading all data
                n_channels = len(raw_data.ch_names)
                n_times = len(raw_data.times)
                dtype_size = 8  # float64 bytes
            else:
                return 0.0
            
            # Estimate memory in GB
            memory_bytes = n_channels * n_times * dtype_size
            memory_gb = memory_bytes / (1024**3)
            
            return memory_gb
            
        except Exception:
            return 0.0
    
    @staticmethod
    def check_available_memory() -> Dict[str, float]:
        """Check available system memory."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent,
            'percent_available': 100 - memory.percent
        }
    
    @staticmethod
    def suggest_chunk_size(data_length: int, available_memory_gb: float) -> int:
        """Suggest optimal chunk size based on available memory."""
        # Conservative estimate: use 50% of available memory
        target_memory_gb = available_memory_gb * 0.5
        
        # Estimate bytes per sample (rough approximation)
        bytes_per_sample = 64 * 8  # 64 channels * 8 bytes per float64
        samples_per_gb = (1024**3) / bytes_per_sample
        
        optimal_chunk_samples = int(target_memory_gb * samples_per_gb)
        
        # Ensure chunk size is reasonable
        chunk_size = min(optimal_chunk_samples, data_length)
        chunk_size = max(chunk_size, 1000)  # Minimum 1000 samples
        
        return chunk_size


def performance_profile(output_dir: Optional[Path] = None):
    """Decorator to profile function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            
            try:
                profiler.enable()
                result = func(*args, **kwargs)
                profiler.disable()
                
                # Save profile if output directory specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    profile_file = output_dir / f"{func.__name__}_profile.prof"
                    profiler.dump_stats(str(profile_file))
                    
                    # Also save text report
                    stats = pstats.Stats(profiler)
                    stats.sort_stats('cumulative')
                    text_file = output_dir / f"{func.__name__}_profile.txt"
                    with open(text_file, 'w') as f:
                        stats.print_stats(file=f)
                
                return result
                
            except Exception as e:
                profiler.disable()
                raise e
        
        return wrapper
    return decorator


def memory_monitor(memory_limit: float = 0.9):
    """Decorator to monitor memory usage during function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before execution
            initial_memory = psutil.virtual_memory().percent / 100
            
            if initial_memory > memory_limit:
                logger.warning(f"Memory usage already high ({initial_memory:.1%}) before executing {func.__name__}")
                gc.collect()  # Try to free some memory
            
            try:
                result = func(*args, **kwargs)
                
                # Check memory after execution
                final_memory = psutil.virtual_memory().percent / 100
                memory_increase = final_memory - initial_memory
                
                if memory_increase > 0.1:  # 10% increase
                    logger.info(f"Function {func.__name__} increased memory usage by {memory_increase:.1%}")
                
                if final_memory > memory_limit:
                    logger.warning(f"Memory usage high ({final_memory:.1%}) after {func.__name__}")
                    gc.collect()
                
                return result
                
            except MemoryError:
                logger.error(f"MemoryError in {func.__name__} - forcing garbage collection")
                gc.collect()
                raise
        
        return wrapper
    return decorator


def with_heartbeat(interval: float = 5.0, message: str = "Processing"):
    """Decorator that shows a spinning progress indicator for long-running functions.
    
    Displays a single-line progress indicator with elapsed time that updates at the
    specified interval. Runs in a separate daemon thread to avoid interfering with
    the main computation.
    
    Args:
        interval: Update interval in seconds (default: 5.0)
        message: Message to display (default: "Processing")
        
    Usage:
        @with_heartbeat(interval=5, message="Computing ICA")
        def compute_ica(data):
            # Long computation
            pass
            
        # Or wrap external functions:
        wrapped_func = with_heartbeat()(external_package.slow_function)
        result = wrapped_func(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Event to signal the heartbeat thread to stop
            stop_event = threading.Event()
            
            # Spinner characters for visual feedback
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spinner_idx = 0
            
            # Track start time
            start_time = time.time()
            
            def heartbeat_thread():
                """Thread function that displays the heartbeat."""
                nonlocal spinner_idx
                
                # Wait a moment to let the function print its initial messages
                time.sleep(0.1)
                
                # Add newline before starting heartbeat to avoid line conflicts
                sys.stdout.write('\n')
                sys.stdout.flush()
                
                while not stop_event.is_set():
                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                    
                    # Create progress message
                    spinner = spinner_chars[spinner_idx % len(spinner_chars)]
                    progress_msg = f"\r{spinner} {message}... [{time_str}]"
                    
                    # Write to stdout and flush
                    sys.stdout.write(progress_msg)
                    sys.stdout.flush()
                    
                    spinner_idx += 1
                    
                    # Wait for interval or until stop event
                    if stop_event.wait(interval):
                        break
                
                # Clear the line and move to new line when done
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
            
            # Start heartbeat thread
            heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
            heartbeat.start()
            
            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
                return result
            finally:
                # Stop heartbeat thread
                stop_event.set()
                heartbeat.join(timeout=1.0)  # Wait max 1 second for thread to finish
                
                # Log completion
                elapsed = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
        
        return wrapper
    return decorator


class ChunkProcessor:
    """Process large datasets in memory-efficient chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, overlap: int = 0):
        """Initialize chunk processor.
        
        Args:
            chunk_size: Size of each chunk in samples (auto if None)
            overlap: Overlap between chunks in samples
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_in_chunks(self, data, processing_func: Callable, **kwargs):
        """Process data in chunks."""
        if self.chunk_size is None:
            # Auto-determine chunk size based on available memory
            memory_info = MemoryOptimizer.check_available_memory()
            self.chunk_size = MemoryOptimizer.suggest_chunk_size(
                data.n_times, memory_info['available_gb']
            )
        
        n_samples = data.n_times
        results = []
        
        for start_idx in range(0, n_samples, self.chunk_size - self.overlap):
            end_idx = min(start_idx + self.chunk_size, n_samples)
            
            # Extract chunk
            chunk = data.copy().crop(
                tmin=data.times[start_idx],
                tmax=data.times[end_idx - 1]
            )
            
            # Process chunk
            try:
                chunk_result = processing_func(chunk, **kwargs)
                results.append(chunk_result)
                
                # Clean up
                del chunk
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")
                continue
        
        return self._combine_chunk_results(results)
    
    def _combine_chunk_results(self, chunk_results):
        """Combine results from multiple chunks."""
        if not chunk_results:
            return None
        
        # Simple concatenation for now
        # TODO: Implement proper combination logic based on data type
        return chunk_results


class ParallelProcessor:
    """Utilities for parallel processing optimization."""
    
    @staticmethod
    def get_optimal_n_jobs() -> int:
        """Get optimal number of parallel jobs based on system resources."""
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative estimate: 1 job per 4GB RAM or 1 job per 2 cores
        memory_limit = max(1, int(memory_gb / 4))
        cpu_limit = max(1, int(cpu_count / 2))
        
        optimal_jobs = min(memory_limit, cpu_limit, cpu_count)
        
        logger.info(f"Optimal parallel jobs: {optimal_jobs} (based on {cpu_count} cores, {memory_gb:.1f}GB RAM)")
        return optimal_jobs
    
    @staticmethod
    def estimate_memory_per_job(total_memory_gb: float, n_jobs: int) -> float:
        """Estimate memory available per parallel job."""
        # Reserve 20% for system
        available_memory = total_memory_gb * 0.8
        memory_per_job = available_memory / n_jobs
        
        return memory_per_job


# Performance testing utilities
def benchmark_processing_stage(stage_func: Callable, test_data, n_runs: int = 3) -> Dict[str, float]:
    """Benchmark a processing stage performance."""
    times = []
    memory_usage = []
    
    for run in range(n_runs):
        # Measure memory before
        initial_memory = psutil.virtual_memory().percent
        
        # Time the execution
        start_time = time.time()
        try:
            result = stage_func(test_data.copy())
            end_time = time.time()
            
            # Measure memory after
            final_memory = psutil.virtual_memory().percent
            
            times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)
            
            # Clean up
            del result
            gc.collect()
            
        except Exception as e:
            logger.error(f"Benchmark run {run} failed: {e}")
            continue
    
    if not times:
        return {"error": "All benchmark runs failed"}
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "mean_memory_increase": np.mean(memory_usage),
        "n_successful_runs": len(times)
    }