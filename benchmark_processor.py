"""
Quick Benchmark Script for YOLO Processor Performance Testing.

Tests both original and optimized versions on a small sample.
"""

import time
import sys
from pathlib import Path
import subprocess
import psutil
import os

def get_sample_images(raw_dir: Path, sample_size: int = 100):
    """Get a sample of images for testing."""
    all_images = []
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            images = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith('.')]
            all_images.extend(images[:10])  # Max 10 per folder
            if len(all_images) >= sample_size:
                break
    return all_images[:sample_size]

def monitor_resources(pid):
    """Monitor CPU and memory usage of a process."""
    try:
        process = psutil.Process(pid)
        cpu_percent = process.cpu_percent(interval=1.0)
        memory_mb = process.memory_info().rss / 1024 / 1024
        return cpu_percent, memory_mb
    except:
        return 0, 0

def run_benchmark(script_name: str, args: list, label: str):
    """Run a processor and measure performance."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {label}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_name] + args
    
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=0.1)
    
    # Run process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Monitor resources
    max_cpu = 0
    max_memory = 0
    
    while process.poll() is None:
        cpu, mem = monitor_resources(process.pid)
        max_cpu = max(max_cpu, cpu)
        max_memory = max(max_memory, mem)
        time.sleep(0.5)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Get output
    stdout, stderr = process.communicate()
    
    # Parse results from output
    processed = 0
    for line in stdout.split('\n'):
        if 'Successfully processed:' in line or 'New:' in line:
            try:
                processed = int(line.split(':')[1].strip().replace(',', ''))
            except:
                pass
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"Duration:        {duration:.2f} seconds")
    print(f"Images:          {processed}")
    print(f"Speed:           {processed/duration:.2f} images/sec")
    print(f"Max CPU:         {max_cpu:.1f}%")
    print(f"Max Memory:      {max_memory:.1f} MB")
    print(f"{'='*60}\n")
    
    return {
        'duration': duration,
        'processed': processed,
        'speed': processed/duration if duration > 0 else 0,
        'max_cpu': max_cpu,
        'max_memory': max_memory
    }

def main():
    """Run comparative benchmark."""
    BASE = Path(__file__).resolve().parent
    
    # Check if scripts exist
    original_script = BASE / "src" / "data_tools" / "yolo_processor.py"
    optimized_script = BASE / "yolo_processor_optimized.py"
    
    if not original_script.exists():
        print(f"ERROR: Original script not found at {original_script}")
        return
    
    if not optimized_script.exists():
        print(f"ERROR: Optimized script not found at {optimized_script}")
        return
    
    # Create temporary test output directories
    test_output_original = BASE / "data" / "test_output_original"
    test_output_optimized = BASE / "data" / "test_output_optimized"
    
    print(f"\n{'='*60}")
    print(f"YOLO PROCESSOR BENCHMARK")
    print(f"{'='*60}")
    print(f"System: MacBook Air M4")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"{'='*60}\n")
    
    print("NOTE: This will process a small sample of images with both processors.")
    print("The original processor will be run first, then the optimized version.")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.")
        return
    
    # Run benchmarks
    results = {}
    
    # Benchmark 1: Original Processor
    print("\n[1/2] Testing ORIGINAL processor...")
    results['original'] = run_benchmark(
        str(original_script),
        [],
        "Original Single-Threaded Processor"
    )
    
    # Benchmark 2: Optimized Processor
    print("\n[2/2] Testing OPTIMIZED processor...")
    results['optimized'] = run_benchmark(
        str(optimized_script),
        [],
        "Optimized Multi-Threaded Processor"
    )
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    speedup = results['optimized']['speed'] / results['original']['speed'] if results['original']['speed'] > 0 else 0
    time_saved = results['original']['duration'] - results['optimized']['duration']
    time_saved_pct = (time_saved / results['original']['duration'] * 100) if results['original']['duration'] > 0 else 0
    
    print(f"\nSpeed:")
    print(f"  Original:  {results['original']['speed']:.2f} images/sec")
    print(f"  Optimized: {results['optimized']['speed']:.2f} images/sec")
    print(f"  Speedup:   {speedup:.2f}x faster ⚡")
    
    print(f"\nTime:")
    print(f"  Original:  {results['original']['duration']:.2f} seconds")
    print(f"  Optimized: {results['optimized']['duration']:.2f} seconds")
    print(f"  Saved:     {time_saved:.2f} seconds ({time_saved_pct:.1f}% faster)")
    
    print(f"\nResource Usage:")
    print(f"  CPU (Original):   {results['original']['max_cpu']:.1f}%")
    print(f"  CPU (Optimized):  {results['optimized']['max_cpu']:.1f}%")
    print(f"  Memory (Original):  {results['original']['max_memory']:.1f} MB")
    print(f"  Memory (Optimized): {results['optimized']['max_memory']:.1f} MB")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    
    if speedup > 3:
        print(f"✓ Optimized version is {speedup:.1f}x faster!")
        print(f"  Use it for large-scale processing.")
    elif speedup > 1.5:
        print(f"✓ Optimized version is {speedup:.1f}x faster.")
        print(f"  Good improvement for batch processing.")
    else:
        print(f"⚠ Speedup is only {speedup:.1f}x.")
        print(f"  Check if multi-threading is working correctly.")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

