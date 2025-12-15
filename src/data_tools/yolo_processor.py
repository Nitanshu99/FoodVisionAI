"""
Unified Data Processor (Clean Directory Mode).

Pipeline:
1. Load Raw Image.
2. Remove Background (U2-Net -> models/).
3. Detect Main Object (YOLO -> models/).
4. Crop & Resize -> Save to 'data/yolo_processed'.
"""

import logging
import cv2
import numpy as np
import sys
import shutil
from pathlib import Path
from urllib import request  # Import for manual download
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Import mapping & BG remover
from src.data_tools.folder_mapper import get_manual_mapping, group_sources_by_target
from src.data_tools.background_removal import BackgroundRemover

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

IMG_SIZE = (512, 512)
CONF_THRESHOLD = 0.25

class UnifiedProcessor:
    def __init__(self, target_model_path: Path):
        print(">> Initializing Background Remover...")
        self.remover = BackgroundRemover()
        
        # --- LOGIC: DOWNLOAD DIRECTLY TO 'models/' ---
        if not target_model_path.exists():
            print(f">> Model not found at {target_model_path}")
            print(f">> Downloading YOLOv8m-seg directly to {target_model_path}...")
            
            # 1. Define Official URL (Matches the version Ultralytics uses)
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt"
            
            # 2. Download manually to the specific folder
            try:
                request.urlretrieve(url, str(target_model_path))
                print(">> Download complete.")
            except Exception as e:
                print(f"ERROR: Could not download model. {e}")
                raise e
            
            # 3. Load from the new local path
            self.model = YOLO(str(target_model_path))
        else:
            print(f">> Loading YOLO from {target_model_path}...")
            self.model = YOLO(str(target_model_path))

    def process_and_clean(self, image_path: Path) -> np.ndarray:
        try:
            # 1. Load
            img = cv2.imread(str(image_path))
            if img is None: return None

            # 2. Remove Background
            clean_img = self.remover.process_image(img)

            # 3. Detect
            results = self.model(clean_img, verbose=False, conf=CONF_THRESHOLD)[0]
            
            target_crop = None

            # Find Largest Object
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
                    h, w, _ = clean_img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    target_crop = clean_img[y1:y2, x1:x2]

            # Fallback
            if target_crop is None or target_crop.size == 0:
                h, w, _ = clean_img.shape
                min_dim = min(h, w)
                sx, sy = (w - min_dim) // 2, (h - min_dim) // 2
                target_crop = clean_img[sy:sy+min_dim, sx:sx+min_dim]

            # 4. Resize
            return cv2.resize(target_crop, IMG_SIZE, interpolation=cv2.INTER_AREA)

        except Exception as e:
            return None

def execute_pipeline(raw_dir: Path, output_dir: Path):
    print(f"\n--- STARTING PIPELINE ---")
    
    # Define Model Path inside models/ folder
    BASE = raw_dir.parents[2] # FoodVisionAI/
    MODEL_PATH = BASE / "models" / "yolov8m-seg.pt"
    
    if not raw_dir.exists():
        print(f"ERROR: Raw directory does not exist!")
        return

    # Scan Disk
    print(f"\n--- SCANNING DISK ---")
    actual_folders = [f.name for f in raw_dir.iterdir() if f.is_dir()]
    print(f"Found {len(actual_folders)} folders in raw directory.")

    # Setup Processor
    processor = UnifiedProcessor(MODEL_PATH)
    
    mapping = get_manual_mapping()
    target_groups = group_sources_by_target(mapping)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    total_images_processed = 0
    total_skipped = 0

    print(f"\n--- PROCESSING TARGETS ---")
    for target_code, source_folders in target_groups.items():
        paths = []
        for src in source_folders:
            src_p = raw_dir / src
            if not src_p.exists():
                for actual in actual_folders:
                    if actual.lower() == src.lower():
                        src_p = raw_dir / actual
                        break
            
            if src_p.exists():
                all_candidates = [f for f in src_p.iterdir() if f.is_file() and not f.name.startswith('.')]
                paths.extend(all_candidates)

        if not paths: continue

        # Split
        train_p, val_p = (paths, []) if len(paths) < 2 else train_test_split(paths, train_size=0.9, random_state=42)

        processed_count_target = 0
        
        for p_list, dest in [(train_p, train_dir), (val_p, val_dir)]:
            dest_folder = dest / target_code
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            for p in p_list:
                final_save_path = dest_folder / f"{p.stem}.jpg"
                
                # Resume Check
                if final_save_path.exists():
                    total_skipped += 1
                    continue

                final_img = processor.process_and_clean(p)
                
                if final_img is not None:
                    cv2.imwrite(str(final_save_path), final_img)
                    processed_count_target += 1
                    total_images_processed += 1
                
                if processed_count_target % 10 == 0:
                    print(f"   Processing {target_code}... ({processed_count_target} new, {total_skipped} skipped)", end='\r')
        
        print(f"   [DONE] {target_code}: {processed_count_target} new images saved.")

    print(f"\n--- SUCCESS ---")
    print(f"Skipped: {total_skipped}")
    print(f"New: {total_images_processed}")

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    RAW_DIR = BASE / "data" / "raw" / "images"
    OUTPUT_DIR = BASE / "data" / "yolo_processed"
    
    execute_pipeline(RAW_DIR, OUTPUT_DIR)