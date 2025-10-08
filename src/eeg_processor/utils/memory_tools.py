import psutil
import os
import gc
import sys
import tracemalloc
import functools
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger


def get_process_memory_detailed() -> Dict[str, Any]:
    """Get detailed process memory breakdown"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_full_info = process.memory_full_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size (physical memory)
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'uss_mb': getattr(memory_full_info, 'uss', 0) / (1024 * 1024),  # Unique Set Size (memory unique to process)
            'pss_mb': getattr(memory_full_info, 'pss', 0) / (1024 * 1024),  # Proportional Set Size
            'shared_mb': getattr(memory_full_info, 'shared', 0) / (1024 * 1024),  # Shared memory
            'text_mb': getattr(memory_full_info, 'text', 0) / (1024 * 1024),  # Text/code memory
            'data_mb': getattr(memory_full_info, 'data', 0) / (1024 * 1024),  # Data memory
            'lib_mb': getattr(memory_full_info, 'lib', 0) / (1024 * 1024),  # Library memory
            'dirty_mb': getattr(memory_full_info, 'dirty', 0) / (1024 * 1024),  # Dirty pages
            'memory_percent': process.memory_percent(),
            'num_fds': int(process.num_fds()) if hasattr(process, 'num_fds') else 0,  # File descriptors
            'num_threads': int(process.num_threads()),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as e:
        logger.warning(f"Could not get detailed memory info: {e}")
        # Fallback to basic memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'uss_mb': 0,
            'pss_mb': 0,
            'shared_mb': 0,
            'text_mb': 0,
            'data_mb': 0,
            'lib_mb': 0,
            'dirty_mb': 0,
            'memory_percent': process.memory_percent(),
            'num_fds': 0,
            'num_threads': int(process.num_threads()),
        }

def get_mne_object_memory(obj) -> Dict[str, Any]:
    """Calculate memory usage of MNE objects (Raw, Epochs, ICA, etc.)"""
    try:
        memory_info = {
            'object_type': type(obj).__name__,
            'data_mb': 0,
            'metadata_mb': 0,
            'total_mb': 0,
            'data_shape': None,
            'data_dtype': None,
            'estimated_mb': 0
        }
        
        # Calculate data array memory
        if hasattr(obj, 'get_data'):
            try:
                data = obj.get_data()
                if isinstance(data, np.ndarray):
                    memory_info['data_mb'] = data.nbytes / (1024 * 1024)
                    memory_info['data_shape'] = data.shape
                    memory_info['data_dtype'] = str(data.dtype)
            except Exception as e:
                logger.debug(f"Could not get data from {type(obj).__name__}: {e}")
        
        # Calculate additional memory for different MNE object types
        if hasattr(obj, '_data') and obj._data is not None:
            memory_info['data_mb'] = obj._data.nbytes / (1024 * 1024)
            memory_info['data_shape'] = obj._data.shape
            memory_info['data_dtype'] = str(obj._data.dtype)
        
        # Estimate metadata memory (info, events, etc.)
        metadata_size = 0
        if hasattr(obj, 'info'):
            metadata_size += sys.getsizeof(obj.info)
        if hasattr(obj, 'events'):
            if obj.events is not None:
                metadata_size += obj.events.nbytes if hasattr(obj.events, 'nbytes') else sys.getsizeof(obj.events)
        if hasattr(obj, 'times'):
            if obj.times is not None:
                metadata_size += obj.times.nbytes if hasattr(obj.times, 'nbytes') else sys.getsizeof(obj.times)
        
        memory_info['metadata_mb'] = metadata_size / (1024 * 1024)
        memory_info['total_mb'] = memory_info['data_mb'] + memory_info['metadata_mb']
        
        # Create estimated memory usage
        if hasattr(obj, 'info') and 'sfreq' in obj.info:
            n_channels = len(obj.info['ch_names']) if 'ch_names' in obj.info else 0
            if hasattr(obj, 'times') and obj.times is not None:
                n_samples = len(obj.times)
                memory_info['estimated_mb'] = (n_channels * n_samples * 8) / (1024 * 1024)  # Assume float64
        
        return memory_info
        
    except Exception as e:
        logger.warning(f"Error calculating memory for {type(obj).__name__}: {e}")
        return {
            'object_type': type(obj).__name__,
            'data_mb': 0,
            'metadata_mb': 0,
            'total_mb': 0,
            'data_shape': None,
            'data_dtype': None,
            'estimated_mb': 0,
            'error': str(e)
        }

def memory_profile(func):
    """Decorator to profile memory usage of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory state
        initial_memory = get_process_memory_detailed()
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            tracemalloc_started = True
        else:
            tracemalloc_started = False
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Get final memory state
            final_memory = get_process_memory_detailed()
            
            # Calculate memory delta
            memory_delta = {
                'rss_delta_mb': final_memory['rss_mb'] - initial_memory['rss_mb'],
                'vms_delta_mb': final_memory['vms_mb'] - initial_memory['vms_mb'],
                'uss_delta_mb': final_memory['uss_mb'] - initial_memory['uss_mb'],
            }
            
            # Note: Advanced leak detection removed - use debug_memory.py for detailed analysis
            leak_analysis = {'potential_leaks': {}}
            
            # Get tracemalloc stats
            tracemalloc_stats = None
            if tracemalloc.is_tracing():
                try:
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')[:10]
                    tracemalloc_stats = [
                        {
                            'filename': stat.traceback.format()[0],
                            'size_mb': stat.size / (1024 * 1024),
                            'count': stat.count
                        }
                        for stat in top_stats
                    ]
                except Exception as e:
                    logger.debug(f"Error getting tracemalloc stats: {e}")
            
            # Log memory profile
            logger.info(f"Memory profile for {func.__name__}:")
            logger.info(f"  RSS delta: {memory_delta['rss_delta_mb']:.2f} MB")
            logger.info(f"  VMS delta: {memory_delta['vms_delta_mb']:.2f} MB")
            logger.info(f"  USS delta: {memory_delta['uss_delta_mb']:.2f} MB")
            
            # Store profiling info on result if possible
            if hasattr(result, '__dict__'):
                result._memory_profile = {
                    'function_name': func.__name__,
                    'initial_memory': initial_memory,
                    'final_memory': final_memory,
                    'memory_delta': memory_delta,
                    'leak_analysis': leak_analysis,
                    'tracemalloc_stats': tracemalloc_stats
                }
            
            return result
            
        finally:
            if tracemalloc_started:
                tracemalloc.stop()
    
    return wrapper


def monitor_gc_effectiveness() -> Dict[str, Any]:
    """Monitor garbage collection effectiveness"""
    # Get GC stats before
    gc_stats_before = gc.get_stats()
    collected_before = gc.get_count()
    
    # Force garbage collection
    collected_objects = gc.collect()
    
    # Get GC stats after
    gc_stats_after = gc.get_stats()
    collected_after = gc.get_count()
    
    return {
        'collected_objects': collected_objects,
        'gc_count_before': collected_before,
        'gc_count_after': collected_after,
        'gc_stats_before': gc_stats_before,
        'gc_stats_after': gc_stats_after,
        'effectiveness_score': collected_objects / max(sum(collected_before), 1)  # Avoid division by zero
    }


class MemoryTracker:
    """Context manager for tracking memory usage during operations"""
    
    def __init__(self, operation_name: str, baseline_memory: Optional[float] = None, 
                 warn_threshold_mb: float = 2000, critical_threshold_mb: float = 8000):
        self.operation_name = operation_name
        self.baseline_memory = baseline_memory
        self.warn_threshold_mb = warn_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.memory_before: Optional[Dict[str, Any]] = None
        self.memory_after: Optional[Dict[str, Any]] = None
        
    def __enter__(self):
        self.memory_before = get_process_memory_detailed()
        if self.baseline_memory is None:
            self.baseline_memory = self.memory_before['rss_mb']
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_after = get_process_memory_detailed()
        self.log_memory_change()
        
    def log_memory_change(self):
        """Log memory change with appropriate warning levels"""
        if not self.memory_before or not self.memory_after:
            return
            
        delta = self.memory_after['rss_mb'] - self.memory_before['rss_mb']
        current = self.memory_after['rss_mb']
        
        # Choose appropriate log level
        if delta > self.warn_threshold_mb:
            logger.warning(f"Large memory increase in {self.operation_name}: {delta:.0f} MB. Consider processing fewer participants at once.")
        elif delta > 100:  # Log significant increases
            logger.info(f"After {self.operation_name}: Memory={current:.0f}MB (+{delta:.0f}MB)")
        else:
            logger.debug(f"After {self.operation_name}: Memory={current:.0f}MB (+{delta:.0f}MB)")
            
        # Critical memory warning
        if current > self.critical_threshold_mb:
            logger.warning(f"High memory usage: {current/1024:.1f} GB. The system may become slow.")
            
    @property
    def memory_delta_mb(self) -> float:
        """Get memory change in MB"""
        if self.memory_before and self.memory_after:
            return self.memory_after['rss_mb'] - self.memory_before['rss_mb']
        return 0.0
        
    @property
    def current_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        if self.memory_after:
            return self.memory_after['rss_mb']
        return 0.0


def cleanup_and_monitor_gc(obj_to_delete, operation_name: str = "cleanup") -> Dict[str, Any]:
    """Clean up object and monitor GC effectiveness"""
    if obj_to_delete is not None:
        del obj_to_delete
    
    gc_result = monitor_gc_effectiveness()
    
    if gc_result['collected_objects'] > 0:
        logger.info(f"{operation_name.capitalize()} freed memory, GC collected {gc_result['collected_objects']} objects")
    else:
        logger.debug(f"{operation_name.capitalize()} completed, no objects collected by GC")
        
    return gc_result



def log_memory_with_context(context: str, baseline_memory: Optional[float] = None, 
                          duration_seconds: Optional[float] = None, participant_id: Optional[str] = None):
    """Log memory usage with contextual information"""
    current_memory = get_process_memory_detailed()
    current_mb = current_memory['rss_mb']
    
    log_parts = [f"Memory: {current_mb:.0f} MB"]
    
    if baseline_memory is not None:
        delta = current_mb - baseline_memory
        log_parts.append(f"(+{delta:.0f} MB from baseline)")
        
    if duration_seconds is not None:
        log_parts.append(f"after {duration_seconds:.1f}s")
        
    log_message = f"{context} - {' '.join(log_parts)}"
    
    if participant_id:
        log_message = f"[{participant_id}] {log_message}"
        
    logger.info(log_message)
    return current_memory


