import psutil

def get_memory_pressure():
    """Get current memory usage as percentage of available system memory"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_percent': memory.percent,
        'pressure_level': 'critical' if memory.percent > 85 else
                         'warning' if memory.percent > 70 else 'normal'
    }

def get_memory_metrics(memory_before, memory_after):
    # After getting memory_before and memory_after
    memory_delta_mb = (memory_after['used_percent'] - memory_before['used_percent']) * memory_after['total_gb'] * 1024 / 100

    # Add memory data to the stage metrics you're already tracking
    return {
        'memory_before': memory_before,
        'memory_after': memory_after,
        'memory_delta_mb': memory_delta_mb,
        'pressure_level': memory_after['pressure_level']
    }

