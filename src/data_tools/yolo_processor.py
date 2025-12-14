"""
YOLO Processor Module.

This script creates the 'Clean' dataset for Phase 2 training.
It iterates through raw data, uses YOLOv8 to detect the main food item,
crops it (removing background), and organizes it into the standard
INDB Code structure (Train/Val split).
"""

import os
import shutil
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Import mapping logic from your existing tool
from src.data_tools.folder_mapper import get_manual_mapping, group_sources_by_target

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
IMG_SIZE = (512, 512)
CONF_THRESHOLD = 0.25

class ImageCleaner:
    def __init__(self, model_path: Path):
        """Initialize YOLO model for cleaning."""
        if not model_path.exists():
            # If model is missing, try to use the default name to trigger auto-download
            logger.warning(f"Model not found at {model_path}. letting Ultralytics download 'yolov8m-seg.pt'...")
            self.model = YOLO("yolov8m-seg.pt") 
        else:
            self.model = YOLO(model_path)
            
    def process_image(self, image_path: Path) -> np.ndarray:
        """
        Loads image, detects largest object, crops, and resizes.
        Returns None if image is unreadable.
        """
        try:
            # Load Image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Run Inference
            results = self.model(img, verbose=False, conf=CONF_THRESHOLD)[0]
            
            target_crop = None
            
            # Logic: Find Largest Detected Object
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
                    x1, y1, x2, y2 = best_box
                    # Clamp coordinates
                    h, w, _ = img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    target_crop = img[y1:y2, x1:x2]

            # Fallback: If no object detected (or crop failed), use Center Crop
            if target_crop is None or target_crop.size == 0:
                h, w, _ = img.shape
                min_dim = min(h, w)
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                target_crop = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

            # Resize to Standard Input (512x512)
            resized = cv2.resize(target_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
            return resized

        except Exception as e:
            logger.warning(f"Error processing {image_path.name}: {e}")
            return None


def execute_pipeline(raw_dir: Path, output_dir: Path, model_path: Path, split_ratio: float = 0.9):
    """
    Main Pipeline: Map -> Clean -> Split -> Save.
    """
    # 1. Setup
    cleaner = ImageCleaner(model_path)
    mapping = get_manual_mapping()
    target_groups = group_sources_by_target(mapping) # {'ASC001': ['hot tea', ...]}
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # 2. Iterate through Target Codes (e.g., ASC123)
    for target_code, source_folders in target_groups.items():
        
        # Collect all raw image paths for this target
        all_image_paths = []
        for src_name in source_folders:
            src_path = raw_dir / src_name
            if src_path.exists():
                images = [f for f in src_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                all_image_paths.extend(images)
        
        if not all_image_paths:
            continue

        # 3. Split Data
        if len(all_image_paths) < 2:
            train_paths, val_paths = all_image_paths, []
        else:
            train_paths, val_paths = train_test_split(
                all_image_paths, train_size=split_ratio, random_state=42, shuffle=True
            )

        # 4. Process & Save
        def _process_batch(paths, dest_root):
            dest_folder = dest_root / target_code
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            for p in paths:
                processed_img = cleaner.process_image(p)
                if processed_img is not None:
                    save_name = f"{p.stem}.jpg" # Force jpg
                    cv2.imwrite(str(dest_folder / save_name), processed_img)

        logger.info(f"Processing {target_code}: {len(train_paths)} Train, {len(val_paths)} Val")
        _process_batch(train_paths, train_dir)
        _process_batch(val_paths, val_dir)

    logger.info(f"YOLO Processing Complete. Data saved to {output_dir}")


if __name__ == "__main__":
    # Define Paths based on project structure
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    RAW_DIR = BASE_DIR / "data" / "raw" / "images"
    OUTPUT_DIR = BASE_DIR / "data" / "yolo_processed" # <--- New Clean Dataset
    MODEL_PATH = BASE_DIR / "models" / "yolov8m-seg.pt"

    execute_pipeline(RAW_DIR, OUTPUT_DIR, MODEL_PATH)