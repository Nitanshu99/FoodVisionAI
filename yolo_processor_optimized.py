"""
Optimized Multi-Threaded YOLO Processor for MacBook Air M4 (16GB RAM).

Optimizations:
1. Multi-threaded processing using ThreadPoolExecutor (utilizes all CPU cores)
2. Memory-efficient batch processing with controlled queue size
3. CoreML acceleration for U2-Net on Apple Silicon
4. Optimized YOLO inference with batch processing
5. Progress tracking with tqdm
6. Graceful error handling and recovery

Hardware Target: MacBook Air M4, 16GB RAM, 10-core CPU
"""

import logging
import cv2
import numpy as np
import sys
from pathlib import Path
from urllib import request
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from tqdm import tqdm
import os
import gc

# Import mapping & BG remover
from src.data_tools.folder_mapper import get_manual_mapping, group_sources_by_target
from src.data_tools.background_removal import BackgroundRemover
from src.vision_utils import process_crop

# Import configuration (with fallback to defaults)
try:
    from processor_config import (
        MAX_WORKERS, BATCH_SIZE, CONF_THRESHOLD,
        IMG_SIZE, FORCE_GC, SHOW_PROGRESS
    )
    print(">> Using custom configuration from processor_config.py")
except ImportError:
    # Default settings for MacBook Air M4
    IMG_SIZE = (512, 512)
    CONF_THRESHOLD = 0.25
    MAX_WORKERS = min(8, mp.cpu_count() - 2)
    BATCH_SIZE = 50
    FORCE_GC = True
    SHOW_PROGRESS = True
    print(">> Using default configuration")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class OptimizedProcessor:
    """Thread-safe processor with shared model instances."""
    
    def __init__(self, target_model_path: Path):
        print(f">> Initializing Optimized Processor for MacBook Air M4...")
        print(f">> CPU Cores Available: {mp.cpu_count()}")
        print(f">> Using {MAX_WORKERS} worker threads")
        
        # Thread-safe background remover (singleton pattern)
        self.remover = BackgroundRemover()
        
        # Download YOLO model if needed
        if not target_model_path.exists():
            print(f">> Model not found at {target_model_path}")
            print(f">> Downloading YOLOv8m-seg...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt"
            try:
                request.urlretrieve(url, str(target_model_path))
                print(">> Download complete.")
            except Exception as e:
                print(f"ERROR: Could not download model. {e}")
                raise e
        
        print(f">> Loading YOLO from {target_model_path}...")
        # YOLO model is thread-safe for inference
        self.model = YOLO(str(target_model_path))
        
        # Lock for thread-safe operations
        self.lock = Lock()
        
    def process_single_image(self, image_path: Path) -> tuple:
        """
        Process a single image (thread-safe).
        Returns: (success: bool, result: np.ndarray or None, path: Path)
        """
        try:
            # 1. Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return (False, None, image_path)

            # 2. Remove background (U2-Net with CoreML acceleration)
            clean_img = self.remover.process_image(img)

            # 3. YOLO detection on clean image
            results = self.model(clean_img, verbose=False, conf=CONF_THRESHOLD)[0]
            
            # 4. Find largest object
            if results.boxes:
                max_area = 0
                best_box = None
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)
                
                if best_box:
                    target_crop = process_crop(clean_img, best_box)
                    if target_crop is not None:
                        return (True, target_crop, image_path)

            # 5. Fallback: Center crop
            h, w, _ = clean_img.shape
            min_dim = min(h, w)
            sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
            fallback_bbox = (sx, sy, sx + min_dim, sy + min_dim)
            result = process_crop(clean_img, fallback_bbox)
            
            return (True, result, image_path)

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return (False, None, image_path)

def process_batch(processor: OptimizedProcessor, 
                  batch_paths: list, 
                  dest_folder: Path,
                  skip_existing: bool) -> tuple:
    """
    Process a batch of images in parallel.
    Returns: (processed_count, skipped_count, failed_count)
    """
    processed = 0
    skipped = 0
    failed = 0
    
    # Filter out existing files if skip mode is enabled
    tasks = []
    for p in batch_paths:
        final_save_path = dest_folder / f"{p.stem}.jpg"
        if skip_existing and final_save_path.exists():
            skipped += 1
        else:
            tasks.append((p, final_save_path))
    
    # Process in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(processor.process_single_image, p): (p, save_path)
            for p, save_path in tasks
        }
        
        # Collect results with progress bar (if enabled)
        futures_iter = as_completed(future_to_path)
        if SHOW_PROGRESS:
            futures_iter = tqdm(futures_iter,
                              total=len(future_to_path),
                              desc=f"Processing {dest_folder.name}",
                              leave=False)

        for future in futures_iter:
            p, save_path = future_to_path[future]
            success, result, _ = future.result()

            if success and result is not None:
                cv2.imwrite(str(save_path), result)
                processed += 1
            else:
                failed += 1

    return processed, skipped, failed

def execute_pipeline(raw_dir: Path, output_dir: Path, skip_existing: bool = False):
    """
    Main pipeline with optimized multi-threaded processing.
    """
    print(f"\n{'='*60}")
    print(f"OPTIMIZED YOLO PROCESSOR - MacBook Air M4")
    print(f"{'='*60}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*60}\n")

    # Define model path
    BASE = raw_dir.parents[2]  # FoodVisionAI/
    MODEL_PATH = BASE / "models" / "yolov8m-seg.pt"

    if not raw_dir.exists():
        print(f"ERROR: Raw directory does not exist: {raw_dir}")
        return

    # Scan disk
    print(f">> Scanning raw directory...")
    actual_folders = [f.name for f in raw_dir.iterdir() if f.is_dir()]
    print(f">> Found {len(actual_folders)} folders")

    # Initialize processor (shared across threads)
    processor = OptimizedProcessor(MODEL_PATH)

    # Get mapping
    mapping = get_manual_mapping()
    target_groups = group_sources_by_target(mapping)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Statistics
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    print(f"\n>> Processing {len(target_groups)} target classes...")

    # Process each target class
    target_items = target_groups.items()
    if SHOW_PROGRESS:
        target_items = tqdm(target_items,
                           desc="Overall Progress",
                           unit="class")

    for target_code, source_folders in target_items:
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

        # Train/val split
        if len(paths) < 2:
            train_p, val_p = paths, []
        else:
            train_p, val_p = train_test_split(paths, train_size=0.9, random_state=42)

        # Process train and val sets
        for p_list, dest in [(train_p, train_dir), (val_p, val_dir)]:
            if not p_list:
                continue

            dest_folder = dest / target_code
            dest_folder.mkdir(parents=True, exist_ok=True)

            # Process in batches to manage memory
            for i in range(0, len(p_list), BATCH_SIZE):
                batch = p_list[i:i + BATCH_SIZE]

                processed, skipped, failed = process_batch(
                    processor, batch, dest_folder, skip_existing
                )

                total_processed += processed
                total_skipped += skipped
                total_failed += failed

                # Force garbage collection after each batch (if enabled)
                if FORCE_GC:
                    gc.collect()

    # Final summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {total_processed}")
    print(f"⊘ Skipped (existing):     {total_skipped}")
    print(f"✗ Failed:                 {total_failed}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized YOLO Processor for MacBook Air M4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images (overwrite existing)
  python yolo_processor_optimized.py

  # Resume mode (skip existing files)
  python yolo_processor_optimized.py --skip

  # Custom paths
  python yolo_processor_optimized.py --raw /path/to/raw --output /path/to/output
        """
    )

    parser.add_argument("-s", "--skip", action="store_true",
                       help="Skip existing files (resume mode)")
    parser.add_argument("--raw", type=Path, default=None,
                       help="Custom raw images directory")
    parser.add_argument("--output", type=Path, default=None,
                       help="Custom output directory")

    args = parser.parse_args()

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
