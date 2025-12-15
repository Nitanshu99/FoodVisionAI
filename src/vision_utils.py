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

def load_image_raw(image_file: Any) -> np.ndarray:
    """
    Loads uploaded file into OpenCV BGR array.

    Args:
        image_file (Any): The uploaded file object (e.g., from Streamlit).

    Returns:
        np.ndarray: The image in BGR format.
    """
    image = Image.open(image_file).convert("RGB")
    image_np = np.asarray(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def preprocess_for_model(img_array: np.ndarray) -> np.ndarray:
    """
    Resizes and formats image for EfficientNet-B5 (512x512).

    Args:
        img_array (np.ndarray): Input image array.

    Returns:
        np.ndarray: Preprocessed batch tensor (1, 512, 512, 3).
    """
    target_size = config.IMG_SIZE # 512x512
    resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_CUBIC)
    input_arr = resized.astype(np.float32)
    return np.expand_dims(input_arr, axis=0)

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
    model_global: Model,
    model_local: Model,
    assessor: Any,
    raw_image: np.ndarray,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Main Interface: Returns a LIST of detected food items.
    Switches between Global and Local models based on scene complexity.

    Implements Relative Area Filtering:
    - Calculates the area of the largest detected object.
    - Discards any object smaller than 30% (0.3) of the max area.
    - This prevents garnishes (e.g., leaves) from triggering the Multi-Dish (Thali) logic.

    Args:
        model_global (Model): Context-Aware EfficientNet-B5.
        model_local (Model): Texture-Specialist EfficientNet-B5.
        assessor (Any): YOLO segmentation engine.
        raw_image (np.ndarray): The full raw image.
        class_names (List[str]): List of class labels.

    Returns:
        List[Dict[str, Any]]: List of prediction results for each valid food item.
    """
    # 1. Run Geometric Analysis (YOLO)
    detected_objects, ppm = assessor.analyze_scene(raw_image)

    # 2. Pre-Decision Filtering (Relative Area Threshold)
    filtered_objects = []
    if detected_objects:
        # Find the largest object (Main Dish)
        max_area = max(obj['area_pixels'] for obj in detected_objects)
        threshold_area = max_area * 0.3  # 0.3 Relative Threshold

        # Filter out minor objects (garnishes, noise)
        for obj in detected_objects:
            if obj['area_pixels'] >= threshold_area:
                filtered_objects.append(obj)
    else:
        filtered_objects = []

    final_results = []

    # --- LOGIC BRANCH: SINGLE VS MULTI ---

    # CASE A: Single Dish (or None) -> Trust User Photography -> USE GLOBAL MODEL
    # If filtering reduced count to 1 (e.g., Pizza + Leaf -> Pizza), we use Global Path.
    if len(filtered_objects) <= 1:

        # Determine Visual Stats (Area/Mask)
        if filtered_objects:
            obj = filtered_objects[0]
            visual_stats = {
                "area_cm2": obj['area_cm2'],
                "bbox": obj['bbox'],
                "mask": obj['mask'],
                "ppm": ppm,
                "occupancy_ratio": obj['area_pixels'] / (raw_image.shape[0] * raw_image.shape[1])
            }
        else:
            # Fallback if YOLO misses: Assume the whole image is the food
            visual_stats = {
                "area_cm2": 0.0,  # Triggers default mass fallback
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
        for obj in filtered_objects:
            # Extract Crop
            x1, y1, x2, y2 = obj['bbox']
            h, w, _ = raw_image.shape
            
            # Clamp coordinates (Matches yolo_processor logic)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = raw_image[y1:y2, x1:x2]

            # Skip tiny garbage crops (redundant check, but safe)
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            # Crop-Specialist Inference (Tight Crop)
            class_id, top_preds = run_classification(model_local, crop, class_names)

            visual_stats = {
                "area_cm2": obj['area_cm2'],
                "bbox": (x1, y1, x2, y2),
                "mask": obj['mask'],
                "ppm": ppm,
                "occupancy_ratio": obj['area_pixels'] / (w * h)
            }

            final_results.append({
                "class_id": class_id,
                "top_predictions": top_preds,
                "visual_stats": visual_stats,
                "crop_type": "Object Crop (Local Specialist)"
            })

    return final_results