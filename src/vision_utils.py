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
import config
from src.data_tools.background_removal import BackgroundRemover
import tensorflow as tf

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
    NEW PIPELINE: Segment first, then background removal per crop.

    Pipeline Flow:
    1. YOLOv8m-seg detects segments on RAW image (threshold 0.25)
    2. For each detected segment:
       a. Crop from raw image
       b. U2Net background removal on crop only
       c. EfficientNet classification on clean crop
       d. Log result

    Args:
        model (Model): The unified EfficientNet-B5 model.
        assessor (Any): YOLO segmentation engine.
        raw_image (np.ndarray): The full raw image.
        class_names (List[str]): List of class labels.

    Returns:
        List[Dict[str, Any]]: List of prediction results for each valid food item.
    """
    # Initialize background remover (will be used per-crop)
    bg_remover = BackgroundRemover()

    # 1. Run YOLO segmentation on RAW image (preserves all details)
    detected_objects, ppm = assessor.analyze_scene(raw_image)

    print(f"🔍 YOLO detected {len(detected_objects) if detected_objects else 0} segments on raw image")

    # 2. Adaptive Filtering: Keep top N largest segments
    MAX_SEGMENTS = 8  # Process up to 8 food items
    filtered_objects = []

    if detected_objects:
        # Sort by area (largest first)
        sorted_objects = sorted(detected_objects, key=lambda x: x['area_pixels'], reverse=True)

        # Keep top N segments
        filtered_objects = sorted_objects[:MAX_SEGMENTS]

        print(f"📊 Adaptive filtering: Keeping top {len(filtered_objects)} largest segments (max {MAX_SEGMENTS})")

        for idx, obj in enumerate(sorted_objects):
            area = obj['area_pixels']
            if idx < MAX_SEGMENTS:
                print(f"   ✅ Segment {idx+1}: area={area:.0f} px (KEPT - rank {idx+1})")
            else:
                print(f"   ❌ Segment {idx+1}: area={area:.0f} px (FILTERED OUT - too small)")

        print(f"✅ {len(filtered_objects)} segments will be processed")

    final_results = []

    # 3. Process each detected segment independently
    for idx, obj in enumerate(filtered_objects):
        print(f"\n📦 Processing segment {idx + 1}/{len(filtered_objects)}...")

        # Extract crop from RAW image (preserves original food details)
        raw_crop = process_crop(raw_image, obj['bbox'])

        if raw_crop is None:
            print(f"⚠️ Segment {idx + 1}: Failed to extract crop, skipping")
            continue

        # Apply U2Net background removal to THIS crop only
        print(f"🧹 Segment {idx + 1}: Applying U2Net background removal to crop...")
        clean_crop = bg_remover.process_image(raw_crop)

        # Classify the clean crop with EfficientNet
        print(f"🧠 Segment {idx + 1}: Running EfficientNet classification...")
        class_id, top_preds = run_classification(model, clean_crop, class_names)

        print(f"✅ Segment {idx + 1}: Classified as '{class_id}' (confidence: {top_preds[0][1]:.2%})")
        print(f"   Top 3 predictions: {[(name, f'{conf:.2%}') for name, conf in top_preds[:3]]}")

        # Calculate area using detected mask and PPM from raw image
        visual_stats = {
            "area_cm2": obj['area_pixels'] / (ppm ** 2),
            "bbox": obj['bbox'],
            "mask": obj['mask'],
            "ppm": ppm,
            "occupancy_ratio": obj['area_pixels'] / (raw_image.shape[0] * raw_image.shape[1])
        }

        final_results.append({
            "class_id": class_id,
            "top_predictions": top_preds,
            "visual_stats": visual_stats,
            "crop_type": "Raw Crop + U2Net (New Pipeline)"
        })

    # Fallback: If no objects detected, process full image
    if not final_results:
        print("⚠️ No segments detected, using full image as fallback")

        # Apply U2Net to full image
        clean_full_image = bg_remover.process_image(raw_image)
        class_id, top_preds = run_classification(model, clean_full_image, class_names)

        print(f"✅ Fallback: Classified as '{class_id}' (confidence: {top_preds[0][1]:.2%})")

        visual_stats = {
            "area_cm2": 0.0,
            "ppm": ppm if ppm else 1.0,
            "occupancy_ratio": 1.0,
            "error": "No specific object detected"
        }

        final_results.append({
            "class_id": class_id,
            "top_predictions": top_preds,
            "visual_stats": visual_stats,
            "crop_type": "Full Image + U2Net (Fallback)"
        })

    print(f"\n🎉 Pipeline complete: {len(final_results)} food items detected\n")
    return final_results

def preprocess_for_model(img: np.ndarray) -> tf.Tensor:
    """
    Preprocesses image for EfficientNet-B5 model inference.
    Matches the exact preprocessing used during training.
    
    Args:
        img (np.ndarray): Image array in BGR format, shape (512, 512, 3).
    
    Returns:
        tf.Tensor: Preprocessed tensor ready for model input, shape (1, 512, 512, 3).
    """
    # Convert BGR to RGB (OpenCV uses BGR, models expect RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] range (standard for EfficientNet)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension: (512, 512, 3) -> (1, 512, 512, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(img_batch, dtype=tf.float32)
