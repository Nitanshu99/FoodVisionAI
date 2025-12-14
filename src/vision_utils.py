"""
Vision Utility Module (Split-Brain Logic).

Implements the Hybrid Inference Strategy:
- Single Object -> Use Full Image with Global Model (Context-Aware).
- Multi Object -> Use Object Crops with Local Model (Texture-Specialist).
"""

import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
from keras import Model
from src import config

def get_class_names() -> List[str]:
    """Retrieves class names (ASC codes) from the training directory."""
    # Priority: Try the new YOLO processed dir first (Local Model Classes)
    train_dir = config.DATA_DIR / "yolo_processed" / "train"
    if not train_dir.exists():
        # Fallback to original processed dir (Global Model Classes)
        train_dir = config.TRAIN_DIR

    if not os.path.exists(train_dir):
        # Fallback if no data found
        return [f"ASC{i:03d}" for i in range(1, 101)]
    
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

def load_image_raw(image_file: Any) -> np.ndarray:
    """Loads uploaded file into OpenCV BGR array."""
    image = Image.open(image_file).convert("RGB")
    image_np = np.asarray(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """Resizes and formats image for EfficientNet-B5 (512x512)."""
    target_size = config.IMG_SIZE # 512x512
    resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_CUBIC)
    input_arr = resized.astype(np.float32)
    return np.expand_dims(input_arr, axis=0)

def run_classification(model: Model, img: np.ndarray, class_names: List[str], top_k: int=3):
    """Helper to run inference on a single image array."""
    input_tensor = preprocess_for_model(img)
    predictions = model(input_tensor, training=False)
    probabilities = predictions.numpy()[0]
    
    # Get Top-K indices
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_preds = []
    
    for idx in top_k_indices:
        if idx < len(class_names):
            top_preds.append((class_names[idx], float(probabilities[idx])))
        else:
            top_preds.append((f"Unknown-{idx}", float(probabilities[idx])))
            
    return top_preds[0][0], top_preds

def predict_food(
    model_global: Model,
    model_local: Model,
    assessor: Any, 
    raw_image: np.ndarray,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Main Interface: Returns a LIST of detected food items.
    Switches between Global and Local models based on scene complexity.
    """
    # 1. Run Geometric Analysis (YOLO)
    detected_objects, ppm = assessor.analyze_scene(raw_image)
    
    final_results = []
    
    # --- LOGIC BRANCH: SINGLE VS MULTI ---
    
    # CASE A: Single Dish (or None) -> Trust User Photography -> USE GLOBAL MODEL
    # We use the full image because the user likely framed the single dish well.
    if len(detected_objects) <= 1:
        
        # Determine Visual Stats (Area/Mask)
        if detected_objects:
            obj = detected_objects[0]
            visual_stats = {
                "area_cm2": obj['area_cm2'],
                "bbox": obj['bbox'],
                "mask": obj['mask'],
                "ppm": ppm,
                "occupancy_ratio": obj['area_pixels'] / (raw_image.shape[0]*raw_image.shape[1])
            }
        else:
            # Fallback if YOLO misses: Assume the whole image is the food
            visual_stats = {
                "area_cm2": 0.0, # Triggers default mass fallback
                "ppm": ppm,
                "occupancy_ratio": 1.0,
                "error": "No specific object detected"
            }

        # Context-Aware Inference (Full Image)
        class_id, top_preds = run_classification(model_global, raw_image, class_names)
        
        final_results.append({
            "class_id": class_id,
            "top_predictions": top_preds,
            "visual_stats": visual_stats,
            "crop_type": "Full Image (Global Context)"
        })

    # CASE B: Multi-Dish (Thali) -> Trust YOLO Crops -> USE LOCAL MODEL
    # We crop each item so the model isn't confused by neighbor foods.
    else:
        for obj in detected_objects:
            # Extract Crop
            x1, y1, x2, y2 = obj['bbox']
            h, w, _ = raw_image.shape
            # Clamp coordinates to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = raw_image[y1:y2, x1:x2]
            
            # Skip tiny garbage crops (noise)
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            # Crop-Specialist Inference (Tight Crop)
            class_id, top_preds = run_classification(model_local, crop, class_names)
            
            visual_stats = {
                "area_cm2": obj['area_cm2'],
                "bbox": (x1,y1,x2,y2),
                "mask": obj['mask'],
                "ppm": ppm,
                "occupancy_ratio": obj['area_pixels'] / (w*h)
            }
            
            final_results.append({
                "class_id": class_id,
                "top_predictions": top_preds,
                "visual_stats": visual_stats,
                "crop_type": "Object Crop (Local Specialist)"
            })
            
    return final_results