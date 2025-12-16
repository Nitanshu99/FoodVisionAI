"""
Configuration file for Optimized YOLO Processor.

Easily tune performance parameters for your MacBook Air M4.
"""

import multiprocessing as mp

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# MacBook Air M4 Specifications
TOTAL_CPU_CORES = mp.cpu_count()  # Should be 10 (4P + 6E cores)
TOTAL_RAM_GB = 16

# ============================================================
# PERFORMANCE TUNING
# ============================================================

# Worker Threads
# - Higher = Faster processing, more CPU usage, more battery drain
# - Lower = Slower processing, less CPU usage, better battery life
# Recommended: 6-8 for balanced performance
# Maximum: TOTAL_CPU_CORES (10)
# Minimum: 2
MAX_WORKERS = 8

# Batch Size (images processed before garbage collection)
# - Higher = Better throughput, more memory usage
# - Lower = Less memory usage, more GC overhead
# Recommended: 50 for 16GB RAM
# If you get OOM errors: Reduce to 25
# If you have memory to spare: Increase to 100
BATCH_SIZE = 50

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# YOLO Confidence Threshold
# - Higher = Fewer detections, faster processing
# - Lower = More detections, slower processing
# Range: 0.0 to 1.0
CONF_THRESHOLD = 0.25

# Image Size (width, height)
# - Larger = Better quality, slower processing, more memory
# - Smaller = Lower quality, faster processing, less memory
# Standard: (512, 512)
IMG_SIZE = (512, 512)

# ============================================================
# APPLE SILICON OPTIMIZATION
# ============================================================

# CoreML Acceleration for U2-Net
# Set to True to use Apple Neural Engine
USE_COREML = True

# ONNX Runtime Providers (in priority order)
# For MacBook Air M4, CoreML is optimal
ONNX_PROVIDERS = [
    'CoreMLExecutionProvider',  # Apple Neural Engine
    'CPUExecutionProvider'       # Fallback
]

# ============================================================
# MEMORY MANAGEMENT
# ============================================================

# Force garbage collection after each batch
FORCE_GC = True

# Clear YOLO cache periodically (every N batches)
# Set to 0 to disable
CLEAR_CACHE_INTERVAL = 10

# ============================================================
# LOGGING & PROGRESS
# ============================================================

# Show progress bars
SHOW_PROGRESS = True

# Verbose logging
VERBOSE = False

# Log file (set to None to disable file logging)
LOG_FILE = None  # or "yolo_processor.log"

# ============================================================
# PRESETS
# ============================================================

def get_preset(preset_name: str) -> dict:
    """
    Get predefined configuration presets.
    
    Available presets:
    - 'max_speed': Maximum processing speed (high CPU/memory)
    - 'balanced': Balanced performance and resource usage
    - 'battery_saver': Minimal resource usage for battery life
    - 'low_memory': Optimized for systems with limited RAM
    """
    presets = {
        'max_speed': {
            'MAX_WORKERS': 10,
            'BATCH_SIZE': 100,
            'CONF_THRESHOLD': 0.3,
            'FORCE_GC': False,
            'CLEAR_CACHE_INTERVAL': 0,
        },
        'balanced': {
            'MAX_WORKERS': 8,
            'BATCH_SIZE': 50,
            'CONF_THRESHOLD': 0.25,
            'FORCE_GC': True,
            'CLEAR_CACHE_INTERVAL': 10,
        },
        'battery_saver': {
            'MAX_WORKERS': 4,
            'BATCH_SIZE': 25,
            'CONF_THRESHOLD': 0.3,
            'FORCE_GC': True,
            'CLEAR_CACHE_INTERVAL': 5,
        },
        'low_memory': {
            'MAX_WORKERS': 6,
            'BATCH_SIZE': 20,
            'CONF_THRESHOLD': 0.25,
            'FORCE_GC': True,
            'CLEAR_CACHE_INTERVAL': 3,
        }
    }
    
    return presets.get(preset_name, presets['balanced'])

def apply_preset(preset_name: str):
    """Apply a preset configuration."""
    preset = get_preset(preset_name)
    globals().update(preset)
    print(f"Applied preset: {preset_name}")
    print(f"Configuration: {preset}")

# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
# In yolo_processor_optimized.py, import this config:

from processor_config import (
    MAX_WORKERS, 
    BATCH_SIZE, 
    CONF_THRESHOLD,
    IMG_SIZE,
    FORCE_GC
)

# Or use a preset:

from processor_config import apply_preset
apply_preset('max_speed')  # For fastest processing
apply_preset('battery_saver')  # For longer battery life
"""

# ============================================================
# PERFORMANCE ESTIMATES
# ============================================================

def estimate_performance():
    """Estimate processing performance based on current config."""
    
    # Baseline: Single-threaded performance
    baseline_speed = 2.5  # images/sec on M4
    
    # Multi-threading speedup (not linear due to I/O)
    threading_factor = min(MAX_WORKERS * 0.7, TOTAL_CPU_CORES * 0.8)
    
    # Batch size impact (larger batches = less overhead)
    batch_factor = 1.0 + (BATCH_SIZE / 100) * 0.1
    
    # Confidence threshold impact (higher = faster)
    conf_factor = 1.0 + (CONF_THRESHOLD - 0.25) * 0.2
    
    estimated_speed = baseline_speed * threading_factor * batch_factor * conf_factor
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE ESTIMATE")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Workers:     {MAX_WORKERS}")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print(f"  Confidence:  {CONF_THRESHOLD}")
    print(f"\nEstimated Performance:")
    print(f"  Speed:       ~{estimated_speed:.1f} images/sec")
    print(f"  Speedup:     ~{estimated_speed/baseline_speed:.1f}x vs single-threaded")
    print(f"\nFor 10,000 images:")
    print(f"  Time:        ~{10000/estimated_speed/60:.1f} minutes")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print(f"\nCurrent Configuration:")
    print(f"  CPU Cores:   {TOTAL_CPU_CORES}")
    print(f"  RAM:         {TOTAL_RAM_GB} GB")
    print(f"  Workers:     {MAX_WORKERS}")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print(f"  Confidence:  {CONF_THRESHOLD}")
    
    estimate_performance()
    
    print("\nAvailable Presets:")
    for preset_name in ['max_speed', 'balanced', 'battery_saver', 'low_memory']:
        print(f"\n  {preset_name}:")
        preset = get_preset(preset_name)
        for key, value in preset.items():
            print(f"    {key}: {value}")

