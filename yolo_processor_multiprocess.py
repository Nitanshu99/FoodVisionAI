"""
High-Performance Multiprocessing YOLO Processor with Auto-Configuration.

DYNAMIC HARDWARE DETECTION:
- Automatically detects MacBook Air M4, AMD EPYC Server, or other systems
- Configures optimal worker count based on CPU cores
- Selects best execution providers (CoreML, CUDA, or CPU)
- Adjusts for thermal limits (passive vs active cooling)

TRUE PARALLEL PROCESSING - Bypasses Python's GIL:
- Uses multiprocessing instead of threading
- Each worker is a separate Python process
- Achieves optimal CPU utilization for your hardware

IDENTICAL OUTPUT to original processor:
- Same background removal (U2-Net)
- Same YOLO detection
- Same cropping logic
- Same image processing
- Same 90/10 train/val split (random_state=42)
- Same output location

Works on:
- MacBook Air M4 (16GB RAM, 10 cores) → ~4 workers, ~2-3 hours for 100k images
- AMD EPYC Server (32 cores, 64GB+ RAM) → ~24 workers, ~30 min for 100k images
- Any Linux/Mac/Windows system → Auto-configured
"""

import logging
import cv2
import numpy as np
import sys
from pathlib import Path

# Fix import path for multiprocessing workers
import os
SCRIPT_DIR = Path(__file__).parent.absolute()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from urllib import request
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import argparse
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time
import json

# Import mapping & BG remover
from src.data_tools.folder_mapper import get_manual_mapping, group_sources_by_target
from src.data_tools.background_removal import BackgroundRemover
from src.vision_utils import process_crop

# Auto-configuration
try:
    from config.hardware import get_auto_config
    AUTO_CONFIG = get_auto_config()
    IMG_SIZE = AUTO_CONFIG['IMG_SIZE']
    CONF_THRESHOLD = AUTO_CONFIG['CONF_THRESHOLD']
    NUM_WORKERS = AUTO_CONFIG['NUM_WORKERS']
    print(f">> Using auto-detected configuration:")
    print(f"   Profile: {AUTO_CONFIG['DESCRIPTION']}")
    print(f"   Workers: {NUM_WORKERS}")
except ImportError:
    print(">> Auto-config not found, using defaults")
    IMG_SIZE = (512, 512)
    CONF_THRESHOLD = 0.25
    NUM_WORKERS = 8

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables for worker processes
_model = None
_remover = None
_model_path = None


def init_worker(model_path):
    """Initialize worker process with its own model instances."""
    global _model, _remover, _model_path

    # Fix import path for worker process
    import sys
    from pathlib import Path
    script_dir = Path(__file__).parent.absolute()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    _model_path = model_path

    # Each process gets its own model instances
    _remover = BackgroundRemover()
    _model = YOLO(str(model_path))
    
    # Suppress YOLO verbose output in workers
    _model.verbose = False


def process_single_image(args):
    """
    Process a single image (runs in worker process).
    
    Args:
        args: tuple of (image_path, output_path, skip_existing)
    
    Returns:
        tuple: (success, output_path, error_msg)
    """
    image_path, output_path, skip_existing = args
    
    try:
        # Skip if exists and skip mode enabled
        if skip_existing and output_path.exists():
            return (True, output_path, "skipped")
        
        # 1. Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return (False, output_path, "failed_to_load")
        
        # 2. Remove background FIRST (U2-Net)
        clean_img = _remover.process_image(img)
        
        # 3. YOLO detection on clean image
        results = _model(clean_img, verbose=False, conf=CONF_THRESHOLD)[0]
        
        # 4. Find LARGEST object only
        best_box = None
        if results.boxes:
            max_area = 0
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)
        
        # 5. Crop and process
        if best_box:
            target_crop = process_crop(clean_img, best_box)
            if target_crop is not None:
                cv2.imwrite(str(output_path), target_crop)
                return (True, output_path, "processed")
        
        # 6. Fallback: Center crop
        h, w, _ = clean_img.shape
        min_dim = min(h, w)
        sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
        fallback_bbox = (sx, sy, sx + min_dim, sy + min_dim)
        result = process_crop(clean_img, fallback_bbox)
        
        if result is not None:
            cv2.imwrite(str(output_path), result)
            return (True, output_path, "processed_fallback")
        
        return (False, output_path, "processing_failed")
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"\n!!! ERROR processing {image_path.name}: {error_msg}\n")
        return (False, output_path, error_msg)


def prepare_tasks(raw_dir: Path, output_dir: Path, skip_existing: bool):
    """
    Prepare all processing tasks with train/val split.
    
    Returns:
        list: List of (image_path, output_path, skip_existing) tuples
    """
    print(f"\n{'='*60}")
    print(f"PREPARING TASKS...")
    print(f"{'='*60}")
    
    # Scan raw directory
    actual_folders = [f.name for f in raw_dir.iterdir() if f.is_dir()]
    print(f">> Found {len(actual_folders)} folders in raw directory")
    
    # Get mapping
    mapping = get_manual_mapping()
    target_groups = group_sources_by_target(mapping)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    all_tasks = []
    total_images = 0

    # Process each target class
    for target_code, source_folders in target_groups.items():
        # Collect all image paths for this target
        paths = []
        for src in source_folders:
            src_p = raw_dir / src

            # Case-insensitive folder matching
            if not src_p.exists():
                for actual in actual_folders:
                    if actual.lower() == src.lower():
                        src_p = raw_dir / actual
                        break

            if src_p.exists():
                candidates = [f for f in src_p.iterdir()
                            if f.is_file() and not f.name.startswith('.')]
                paths.extend(candidates)

        if not paths:
            continue

        # IDENTICAL 90/10 split with same random seed as original
        if len(paths) < 2:
            train_p, val_p = paths, []
        else:
            train_p, val_p = train_test_split(paths, train_size=0.9, random_state=42)

        # Create output directories
        train_folder = train_dir / target_code
        val_folder = val_dir / target_code
        train_folder.mkdir(parents=True, exist_ok=True)
        val_folder.mkdir(parents=True, exist_ok=True)

        # Add tasks for train set
        for p in train_p:
            output_path = train_folder / f"{p.stem}.jpg"
            all_tasks.append((p, output_path, skip_existing))
            total_images += 1

        # Add tasks for val set
        for p in val_p:
            output_path = val_folder / f"{p.stem}.jpg"
            all_tasks.append((p, output_path, skip_existing))
            total_images += 1

    print(f">> Total images to process: {total_images:,}")
    print(f">> Total tasks prepared: {len(all_tasks):,}")
    print(f"{'='*60}\n")

    return all_tasks, total_images


def execute_pipeline(raw_dir: Path, output_dir: Path, skip_existing: bool = False):
    """
    Main pipeline with multiprocessing.
    """
    print(f"\n{'='*60}")
    print(f"MULTIPROCESSING YOLO PROCESSOR - MacBook Air M4")
    print(f"{'='*60}")
    print(f"CPU Cores: {cpu_count()}")
    print(f"Worker Processes: {NUM_WORKERS}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*60}\n")

    # Define model path
    BASE = raw_dir.parents[2]  # FoodVisionAI/
    MODEL_PATH = BASE / "models" / "yolov8m-seg.pt"

    if not raw_dir.exists():
        print(f"ERROR: Raw directory does not exist: {raw_dir}")
        return

    # Download YOLO model if needed
    if not MODEL_PATH.exists():
        print(f">> Downloading YOLOv8m-seg model...")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt"
        try:
            request.urlretrieve(url, str(MODEL_PATH))
            print(">> Download complete.")
        except Exception as e:
            print(f"ERROR: Could not download model. {e}")
            return

    # Prepare all tasks
    all_tasks, total_images = prepare_tasks(raw_dir, output_dir, skip_existing)

    if not all_tasks:
        print("No images to process!")
        return

    # Statistics
    processed = 0
    skipped = 0
    failed = 0
    errors = []

    # Start timer
    start_time = time.time()

    print(f"{'='*60}")
    print(f"STARTING MULTIPROCESSING...")
    print(f"{'='*60}\n")

    # Create process pool and process images
    with Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
        # Process with progress bar
        with tqdm(total=len(all_tasks), desc="Processing Images", unit="img") as pbar:
            for success, output_path, status in pool.imap_unordered(process_single_image, all_tasks):
                if status == "skipped":
                    skipped += 1
                elif success:
                    processed += 1
                else:
                    failed += 1
                    errors.append((output_path, status))

                pbar.update(1)

                # Update progress bar description with stats
                pbar.set_postfix({
                    'processed': processed,
                    'skipped': skipped,
                    'failed': failed
                })

    # End timer
    end_time = time.time()
    duration = end_time - start_time

    # Calculate statistics
    total_processed_new = processed
    images_per_sec = total_processed_new / duration if duration > 0 else 0

    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {processed:,}")
    print(f"⊘ Skipped (existing):     {skipped:,}")
    print(f"✗ Failed:                 {failed:,}")
    print(f"{'='*60}")
    print(f"Time elapsed:     {duration/60:.1f} minutes ({duration:.1f} seconds)")
    print(f"Processing speed: {images_per_sec:.2f} images/second")
    print(f"{'='*60}\n")

    # Save error log if there were failures
    if errors:
        error_log_path = BASE / "processing_errors.json"
        with open(error_log_path, 'w') as f:
            json.dump([{"path": str(p), "error": e} for p, e in errors], f, indent=2)
        print(f"⚠ Error log saved to: {error_log_path}\n")


if __name__ == "__main__":
    # Fix for CUDA multiprocessing on Linux/GPU servers
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Multiprocessing YOLO Processor for MacBook Air M4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images
  python yolo_processor_multiprocess.py

  # Resume mode (skip existing files)
  python yolo_processor_multiprocess.py --skip

  # Custom paths
  python yolo_processor_multiprocess.py --raw /path/to/raw --output /path/to/output
        """
    )

    parser.add_argument("-s", "--skip", action="store_true",
                       help="Skip existing files (resume mode)")
    parser.add_argument("--raw", type=Path, default=None,
                       help="Custom raw images directory")
    parser.add_argument("--output", type=Path, default=None,
                       help="Custom output directory")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                       help=f"Number of worker processes (default: {NUM_WORKERS})")

    args = parser.parse_args()

    # Update worker count if specified
    NUM_WORKERS = args.workers

    # Default paths
    BASE = Path(__file__).resolve().parent
    RAW_DIR = args.raw or (BASE / "data" / "raw" / "images")
    OUTPUT_DIR = args.output or (BASE / "data" / "yolo_processed")

    # Verify paths
    if not RAW_DIR.exists():
        print(f"ERROR: Raw directory not found: {RAW_DIR}")
        sys.exit(1)

    # Run pipeline
    execute_pipeline(RAW_DIR, OUTPUT_DIR, skip_existing=args.skip)


