"""
Vision Utility Module (Split-Brain Logic).

Implements the Hybrid Inference Strategy:
- Single Object -> Use Full Image with Global Model (Context-Aware).
- Multi Object -> Use Object Crops with Local Model (Texture-Specialist).

Includes pre-decision filtering to handle noise (e.g., garnishes).
"""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from keras import Model
from src import config
from src.data_tools.background_removal import BackgroundRemover

def get_class_names() -> List[str]:
    """
    Retrieves class names (ASC codes) from the training directory.

    Returns:
        List[str]: A sorted list of class names (e.g., ['ASC001', 'ASC002']).
    """
    # Priority: Try the new YOLO processed dir first (Local Model Classes)
    train_dir = config.DATA_DIR / "yolo_processed" / "train"
    if not train_dir.exists():
        # Fallback to original processed dir (Global Model Classes)
        train_dir = config.TRAIN_DIR

    if not os.path.exists(train_dir):
        # Fallback if no data found
        return [f"ASC{i:03d}" for i in range(1, 101)]

    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

def process_crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Shared cropping logic for both training and inference.
    Ensures consistent preprocessing between data generation and model prediction.
    
    Args:
        image (np.ndarray): Input image array.
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).
    
    Returns:
        np.ndarray: Standardized crop resized to 512x512.
    """
    x1, y1, x2, y2 = bbox
    h, w, _ = image.shape
    
    # Clamp coordinates
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Extract crop
    crop = image[y1:y2, x1:x2]
    
    # Skip invalid crops
    if crop.shape[0] < 30 or crop.shape[1] < 30:
        return None
    
    # Resize using INTER_AREA (consistent with training)
    target_size = config.IMG_SIZE  # (512, 512)
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

def run_classification(
    model: Model,
    img: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Runs inference on a single image array using the provided model.

    Args:
        model (Model): The Keras model (Global or Local).
        img (np.ndarray): The image array to classify.
        class_names (List[str]): List of valid class labels.
        top_k (int, optional): Number of top predictions to return. Defaults to 3.

    Returns:
        Tuple[str, List[Tuple[str, float]]]: The top class ID and list of (class, prob) tuples.
    """
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
    model: Model,
    assessor: Any,
    raw_image: np.ndarray,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Single-Model Interface: Returns a LIST of detected food items.
    Uses split-flow approach for consistency between training and inference.

    Args:
        model (Model): The unified EfficientNet-B5 model.
        assessor (Any): YOLO segmentation engine.
        raw_image (np.ndarray): The full raw image.
        class_names (List[str]): List of class labels.

    Returns:
        List[Dict[str, Any]]: List of prediction results for each valid food item.
    """
    # Initialize background remover
    bg_remover = BackgroundRemover()
    
    # 1. Geometry Flow: Calculate PPM from raw image (preserves plate context)
    _, ppm = assessor.analyze_scene(raw_image)
    
    # 2. Topology Flow: Remove background and segment on clean image
    clean_image = bg_remover.process_image(raw_image)
    detected_objects, _ = assessor.analyze_scene(clean_image)

    # 3. Pre-Decision Filtering (Relative Area Threshold)
    filtered_objects = []
    if detected_objects:
        max_area = max(obj['area_pixels'] for obj in detected_objects)
        threshold_area = max_area * 0.3
        
        for obj in detected_objects:
            if obj['area_pixels'] >= threshold_area:
                filtered_objects.append(obj)

    final_results = []

    # 4. Process each detected object
    for obj in filtered_objects:
        # Extract crop using shared logic
        crop = process_crop(clean_image, obj['bbox'])
        
        if crop is None:
            continue

        # Single model prediction
        class_id, top_preds = run_classification(model, crop, class_names)

        # Calculate area using clean mask but raw PPM
        visual_stats = {
            "area_cm2": obj['area_pixels'] / (ppm ** 2),
            "bbox": obj['bbox'],
            "mask": obj['mask'],
            "ppm": ppm,
            "occupancy_ratio": obj['area_pixels'] / (clean_image.shape[0] * clean_image.shape[1])
        }

        final_results.append({
            "class_id": class_id,
            "top_predictions": top_preds,
            "visual_stats": visual_stats,
            "crop_type": "Clean Crop (Unified Model)"
        })

    # Fallback: If no objects detected, use full clean image
    if not final_results:
        class_id, top_preds = run_classification(model, clean_image, class_names)
        
        visual_stats = {
            "area_cm2": 0.0,
            "ppm": ppm,
            "occupancy_ratio": 1.0,
            "error": "No specific object detected"
        }

        final_results.append({
            "class_id": class_id,
            "top_predictions": top_preds,
            "visual_stats": visual_stats,
            "crop_type": "Full Clean Image (Fallback)"
        })

    return final_results
