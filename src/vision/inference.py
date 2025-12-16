"""
Vision Inference Module.

Implements the Hybrid Inference Strategy:
- Single Object -> Use Full Image with Global Model (Context-Aware).
- Multi Object -> Use Object Crops with Local Model (Texture-Specialist).

Includes pre-decision filtering to handle noise (e.g., garnishes).
"""

from typing import List, Dict, Any
import numpy as np
from keras import Model
import config
from src.data_tools.background_removal import BackgroundRemover

# Import utilities from utils package
from src.utils.image_utils import process_crop, run_classification


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

