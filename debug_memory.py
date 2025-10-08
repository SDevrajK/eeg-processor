#!/usr/bin/env python3
"""
Memory debugging script for EEG processing pipeline.
Run this to identify memory bottlenecks in your specific workflow.
"""

import psutil
import os
import gc
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Physical memory
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
        'percent': process.memory_percent()
    }

def get_system_memory():
    """Get system memory info."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_percent': memory.percent,
        'pressure_level': 'critical' if memory.percent > 85 else
                         'warning' if memory.percent > 70 else 'normal'
    }

def memory_checkpoint(label, detailed=False):
    """Print memory usage at a checkpoint with enhanced analysis."""
    proc_mem = get_memory_usage()
    sys_mem = get_system_memory()
    
    # Get enhanced memory details
    try:
        from src.eeg_processor.utils.memory_tools import get_process_memory_detailed
        detailed_mem = get_process_memory_detailed()
        # Use local object counting for debugging
        import gc
        from collections import defaultdict
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            object_counts[type(obj).__name__] += 1
        object_counts = dict(object_counts)
        
        print(f"\n📊 ENHANCED MEMORY CHECKPOINT: {label}")
        print(f"   Process Memory: {proc_mem['rss_mb']:.1f} MB ({proc_mem['percent']:.1f}%)")
        print(f"   System Memory:  {sys_mem['used_percent']:.1f}% used, {sys_mem['available_gb']:.1f}GB available")
        print(f"   Pressure Level: {sys_mem['pressure_level'].upper()}")
        
        if detailed:
            print(f"   Virtual Memory: {proc_mem['vms_mb']:.1f} MB")
            print(f"   USS (Unique):   {detailed_mem['uss_mb']:.1f} MB")
            print(f"   PSS (Proportional): {detailed_mem['pss_mb']:.1f} MB")
            print(f"   Shared Memory:  {detailed_mem['shared_mb']:.1f} MB")
            print(f"   Total System:   {sys_mem['total_gb']:.1f} GB")
            print(f"   File Descriptors: {detailed_mem['num_fds']}")
            print(f"   Threads:        {detailed_mem['num_threads']}")
            print(f"   Total Objects:  {sum(object_counts.values()):,}")
        
        return {
            'basic': {'proc_mem': proc_mem, 'sys_mem': sys_mem},
            'detailed': detailed_mem,
            'objects': object_counts
        }
        
    except ImportError:
        print(f"\n📊 BASIC MEMORY CHECKPOINT: {label}")
        print(f"   Process Memory: {proc_mem['rss_mb']:.1f} MB ({proc_mem['percent']:.1f}%)")
        print(f"   System Memory:  {sys_mem['used_percent']:.1f}% used, {sys_mem['available_gb']:.1f}GB available")
        print(f"   Pressure Level: {sys_mem['pressure_level'].upper()}")
        
        if detailed:
            print(f"   Virtual Memory: {proc_mem['vms_mb']:.1f} MB")
            print(f"   Total System:   {sys_mem['total_gb']:.1f} GB")
        
        return proc_mem, sys_mem

def monitor_file_loading(file_path):
    """Monitor memory usage during file loading."""
    print(f"🔍 MONITORING FILE LOADING: {file_path}")
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")
    
    # Before loading
    memory_checkpoint("BEFORE loading file")
    
    # Import here to measure loading overhead
    import mne
    from src.eeg_processor.file_io.brainvision import BrainVisionLoader
    
    memory_checkpoint("AFTER imports")
    
    # Load file
    print("   Loading file...")
    if file_path.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(file_path, preload=False, verbose=False)
    else:
        raise ValueError("Add support for your file format")
    
    memory_checkpoint("AFTER loading (preload=False)")
    
    # Preload data
    print("   Preloading data...")
    raw.load_data()
    
    proc_mem, sys_mem = memory_checkpoint("AFTER preloading data", detailed=True)
    
    # Calculate memory multiplication
    memory_multiplier = proc_mem['rss_mb'] / file_size_mb
    print(f"\n🎯 MEMORY ANALYSIS:")
    print(f"   File size:        {file_size_mb:.1f} MB")
    print(f"   Process memory:   {proc_mem['rss_mb']:.1f} MB")
    print(f"   Memory multiplier: {memory_multiplier:.1f}x")
    
    if memory_multiplier > 3:
        print("   ⚠️  HIGH MEMORY USAGE - Consider optimizations")
    elif memory_multiplier > 2:
        print("   ⚡ MODERATE MEMORY USAGE - Normal for preloaded data")
    else:
        print("   ✅ EFFICIENT MEMORY USAGE")
    
    return raw

def test_epoching_memory(raw, events, event_id):
    """Test memory usage during epoching."""
    print(f"\n🔍 MONITORING EPOCHING PROCESS")
    
    memory_checkpoint("BEFORE epoching")
    
    # Test without preloading first
    print("   Creating epochs (preload=False)...")
    epochs_lazy = mne.Epochs(
        raw.copy(),
        events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        preload=False,  # Don't load into memory
        verbose=False
    )
    
    memory_checkpoint("AFTER epochs creation (preload=False)")
    
    # Now preload
    print("   Preloading epochs...")
    epochs_lazy.load_data()
    
    memory_checkpoint("AFTER epochs preloading", detailed=True)
    
    print(f"   Number of epochs: {len(epochs_lazy)}")
    print(f"   Epochs shape: {epochs_lazy.get_data().shape}")
    
    return epochs_lazy

def cleanup_test():
    """Test memory cleanup."""
    print(f"\n🧹 TESTING MEMORY CLEANUP")
    
    memory_checkpoint("BEFORE cleanup")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"   Garbage collected: {collected} objects")
    
    memory_checkpoint("AFTER garbage collection")

if __name__ == "__main__":
    print("🚀 EEG PROCESSOR MEMORY DEBUGGER")
    print("=" * 50)
    
    # Initial memory state
    memory_checkpoint("SCRIPT START", detailed=True)
    
    # Test with your file
    file_path = input("Enter path to your EEG file (.vhdr): ").strip()
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        exit(1)
    
    try:
        # Monitor file loading
        raw = monitor_file_loading(file_path)
        
        # Find events for epoching test
        print("\n🔍 Finding events...")
        events = mne.find_events(raw, verbose=False)
        print(f"   Found {len(events)} events")
        
        if len(events) > 0:
            event_id = {'stimulus': events[0, 2]}  # Use first event type
            
            # Test epoching
            epochs = test_epoching_memory(raw, events, event_id)
            
            # Cleanup test
            del raw, epochs, events
            cleanup_test()
        else:
            print("   No events found - skipping epoching test")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Memory debugging complete!")

def detailed_memory_analysis(raw_data, stage_name):
    """Perform detailed memory analysis of processing stage"""
    try:
        from src.eeg_processor.utils.memory_tools import (get_mne_object_memory, 
                                                           get_process_memory_detailed,
                                                           monitor_gc_effectiveness)
        # Local implementations for debugging
        import gc
        from collections import defaultdict
        
        def get_object_counts():
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                object_counts[type(obj).__name__] += 1
            return dict(object_counts)
        
        def detect_memory_leaks(before_objects, after_objects):
            potential_leaks = {}
            for obj_type, after_count in after_objects.items():
                before_count = before_objects.get(obj_type, 0)
                increase = after_count - before_count
                if increase > 10000:
                    potential_leaks[obj_type] = {
                        'before': before_count,
                        'after': after_count,
                        'increase': increase
                    }
            return {
                'potential_leaks': potential_leaks,
                'total_objects_before': sum(before_objects.values()),
                'total_objects_after': sum(after_objects.values()),
                'net_object_increase': sum(after_objects.values()) - sum(before_objects.values())
            }
        
        print(f"\n🔬 DETAILED MEMORY ANALYSIS: {stage_name}")
        print("=" * 60)
        
        # Analyze MNE object memory
        object_memory = get_mne_object_memory(raw_data)
        print(f"📦 MNE Object Analysis:")
        print(f"   Object Type: {object_memory['object_type']}")
        print(f"   Data Memory: {object_memory['data_mb']:.1f} MB")
        print(f"   Metadata Memory: {object_memory['metadata_mb']:.1f} MB")
        print(f"   Total Memory: {object_memory['total_mb']:.1f} MB")
        print(f"   Data Shape: {object_memory['data_shape']}")
        print(f"   Data Type: {object_memory['data_dtype']}")
        
        # Process memory breakdown
        process_memory = get_process_memory_detailed()
        print(f"\n🏗️  Process Memory Breakdown:")
        print(f"   RSS (Physical): {process_memory['rss_mb']:.1f} MB")
        print(f"   USS (Unique): {process_memory['uss_mb']:.1f} MB") 
        print(f"   PSS (Proportional): {process_memory['pss_mb']:.1f} MB")
        print(f"   VMS (Virtual): {process_memory['vms_mb']:.1f} MB")
        print(f"   Shared Memory: {process_memory['shared_mb']:.1f} MB")
        
        # Object count analysis
        object_counts = get_object_counts()
        print(f"\n📊 Object Count Analysis:")
        print(f"   Total Objects: {sum(object_counts.values()):,}")
        
        # Show top object types
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for obj_type, count in sorted_objects:
            print(f"   {obj_type}: {count:,}")
            
        # Memory efficiency analysis
        if object_memory['total_mb'] > 0:
            efficiency = object_memory['total_mb'] / process_memory['rss_mb']
            print(f"\n📈 Memory Efficiency:")
            print(f"   Object/Process ratio: {efficiency:.3f}")
            print(f"   Memory overhead: {(1-efficiency)*100:.1f}%")
            
            if efficiency < 0.5:
                print("   ⚠️  HIGH OVERHEAD - Consider memory optimization")
            elif efficiency > 0.8:
                print("   ✅ EFFICIENT MEMORY USAGE")
        
        return {
            'object_memory': object_memory,
            'process_memory': process_memory,
            'object_counts': object_counts
        }
        
    except ImportError as e:
        print(f"❌ Enhanced memory tools not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in detailed analysis: {e}")
        return None

def profile_processing_stages(raw_data, config=None):
    """Profile memory usage across all processing stages"""
    try:
        # Use local implementations for debugging
        import gc
        from collections import defaultdict
        
        def get_object_counts():
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                object_counts[type(obj).__name__] += 1
            return dict(object_counts)
        
        def detect_memory_leaks(before_objects, after_objects):
            potential_leaks = {}
            for obj_type, after_count in after_objects.items():
                before_count = before_objects.get(obj_type, 0)
                increase = after_count - before_count
                if increase > 10000:
                    potential_leaks[obj_type] = {
                        'before': before_count,
                        'after': after_count,
                        'increase': increase
                    }
            return {
                'potential_leaks': potential_leaks,
                'total_objects_before': sum(before_objects.values()),
                'total_objects_after': sum(after_objects.values()),
                'net_object_increase': sum(after_objects.values()) - sum(before_objects.values())
            }
        
        print(f"\n🚀 PROCESSING STAGES MEMORY PROFILER")
        print("=" * 60)
        
        # Define typical EEG processing stages
        stages = [
            "Initial Load",
            "Filtering", 
            "Bad Channel Detection",
            "Re-referencing",
            "Artifact Removal (ICA)",
            "Epoching",
            "Time-Frequency Analysis"
        ]
        
        stage_results = []
        initial_objects = get_object_counts()
        previous_objects = initial_objects.copy()
        
        for i, stage in enumerate(stages):
            print(f"\n📍 Stage {i+1}: {stage}")
            
            # Simulate stage processing (in real use, this would be actual processing)
            current_checkpoint = memory_checkpoint(f"Stage {i+1}: {stage}", detailed=True)
            current_objects = get_object_counts()
            
            # Detect memory leaks since previous stage
            leak_analysis = detect_memory_leaks(previous_objects, current_objects)
            
            stage_result = {
                'stage_name': stage,
                'memory_info': current_checkpoint,
                'leak_analysis': leak_analysis
            }
            stage_results.append(stage_result)
            
            # Report significant findings
            if leak_analysis['potential_leaks']:
                print(f"   ⚠️  Potential leaks detected:")
                for obj_type, leak_info in leak_analysis['potential_leaks'].items():
                    print(f"      {obj_type}: +{leak_info['increase']} objects")
            
            if leak_analysis['net_object_increase'] > 1000:
                print(f"   📈 Large object increase: +{leak_analysis['net_object_increase']} objects")
            
            previous_objects = current_objects.copy()
        
        # Overall analysis
        print(f"\n📊 OVERALL PROFILING SUMMARY")
        print("=" * 40)
        
        total_leaks = sum(len(result['leak_analysis']['potential_leaks']) for result in stage_results)
        total_object_increase = stage_results[-1]['leak_analysis']['total_objects_after'] - initial_objects.get('total', sum(initial_objects.values()))
        
        print(f"   Total stages analyzed: {len(stages)}")
        print(f"   Stages with leaks: {sum(1 for r in stage_results if r['leak_analysis']['potential_leaks'])}")
        print(f"   Total object increase: {total_object_increase:,}")
        
        if total_leaks > 0:
            print(f"   ⚠️  Total potential leak types: {total_leaks}")
        else:
            print(f"   ✅ No significant leaks detected")
        
        return stage_results
        
    except ImportError as e:
        print(f"❌ Enhanced memory tools not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in stage profiling: {e}")
        return None

def test_memory_leaks():
    """Test for memory leaks in processing pipeline"""
    try:
        from src.eeg_processor.utils.memory_tools import monitor_gc_effectiveness
        # Use local implementations for debugging
        import gc
        from collections import defaultdict
        
        def get_object_counts():
            object_counts = defaultdict(int)
            for obj in gc.get_objects():
                object_counts[type(obj).__name__] += 1
            return dict(object_counts)
        
        def detect_memory_leaks(before_objects, after_objects):
            potential_leaks = {}
            for obj_type, after_count in after_objects.items():
                before_count = before_objects.get(obj_type, 0)
                increase = after_count - before_count
                if increase > 10000:
                    potential_leaks[obj_type] = {
                        'before': before_count,
                        'after': after_count,
                        'increase': increase
                    }
            return {
                'potential_leaks': potential_leaks,
                'total_objects_before': sum(before_objects.values()),
                'total_objects_after': sum(after_objects.values()),
                'net_object_increase': sum(after_objects.values()) - sum(before_objects.values())
            }
        
        print(f"\n🧪 MEMORY LEAK DETECTION TEST")
        print("=" * 50)
        
        print("Running multiple processing cycles to detect persistent leaks...")
        
        leak_results = []
        initial_objects = get_object_counts()
        
        # Simulate multiple processing runs
        for cycle in range(3):
            print(f"\n🔄 Cycle {cycle + 1}")
            
            # Simulate processing (in real use, run actual pipeline)
            cycle_start = memory_checkpoint(f"Cycle {cycle + 1} start")
            
            # Force garbage collection
            gc_result = monitor_gc_effectiveness()
            print(f"   GC collected: {gc_result['collected_objects']} objects")
            
            cycle_end = memory_checkpoint(f"Cycle {cycle + 1} end")
            current_objects = get_object_counts()
            
            # Analyze leaks
            leak_analysis = detect_memory_leaks(initial_objects, current_objects)
            leak_results.append({
                'cycle': cycle + 1,
                'leak_analysis': leak_analysis,
                'gc_effectiveness': gc_result['effectiveness_score']
            })
            
            # Report cycle findings
            if leak_analysis['potential_leaks']:
                print(f"   ⚠️  Persistent objects detected:")
                for obj_type, info in leak_analysis['potential_leaks'].items():
                    print(f"      {obj_type}: {info['after']} objects (+{info['increase']})")
        
        # Final analysis
        print(f"\n📋 LEAK TEST SUMMARY")
        print("=" * 30)
        
        persistent_leaks = {}
        for result in leak_results:
            for obj_type, leak_info in result['leak_analysis']['potential_leaks'].items():
                if obj_type not in persistent_leaks:
                    persistent_leaks[obj_type] = []
                persistent_leaks[obj_type].append(leak_info['increase'])
        
        # Identify consistent leaks
        consistent_leaks = {
            obj_type: increases for obj_type, increases in persistent_leaks.items()
            if all(inc > 0 for inc in increases)  # Consistently increasing
        }
        
        if consistent_leaks:
            print(f"   ❌ CONSISTENT LEAKS DETECTED:")
            for obj_type, increases in consistent_leaks.items():
                print(f"      {obj_type}: {increases} (consistently increasing)")
        else:
            print(f"   ✅ NO CONSISTENT LEAKS DETECTED")
        
        avg_gc_effectiveness = sum(r['gc_effectiveness'] for r in leak_results) / len(leak_results)
        print(f"   Average GC effectiveness: {avg_gc_effectiveness:.3f}")
        
        return {
            'leak_results': leak_results,
            'consistent_leaks': consistent_leaks,
            'avg_gc_effectiveness': avg_gc_effectiveness
        }
        
    except ImportError as e:
        print(f"❌ Enhanced memory tools not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in leak testing: {e}")
        return None