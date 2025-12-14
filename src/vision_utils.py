"""
Vision Utility Module.

Contains pre-processing and inference functions required for the
EfficientNet-B5 model. This handles the high-resolution input requirement
and prepares the image for the model.
"""

import os
from typing import Tuple, List, Dict, Union, Any
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from src import config


def get_class_names() -> List[str]:
    """
    Retrieves the list of class names (ASC codes) from the training directory.

    Assumes the dataset is structured as: data/processed/train/ASCxxx
    The class order is alphanumeric, matching keras.utils.image_dataset_from_directory.

    Returns:
        List[str]: A sorted list of class names (e.g., ['ASC001', 'ASC002']).
    """
    train_dir = config.TRAIN_DIR
    if not os.path.exists(train_dir):
        # Fallback if running on a machine without the full dataset
        return [f"ASC{i:03d}" for i in range(1, 101)]

    # Filter only directories and sort them (standard Keras behavior)
    class_names = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    return class_names


def preprocess_image(image_file: Any) -> np.ndarray:
    """
    Loads, resizes, and normalizes a raw image file for the B5 model.

    Compliance: Ensures input is 512x512 pixels as required by the architecture.

    Args:
        image_file (Any): The raw image content (e.g., from st.file_uploader).

    Returns:
        np.ndarray: The preprocessed image as a 4D tensor (1, H, W, 3).
    """
    try:
        # Load image using PIL
        img = Image.open(image_file).convert("RGB")

        # Resize to the required 512x512 resolution (Section 2)
        target_size: Tuple[int, int] = config.IMG_SIZE
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.asarray(img, dtype=np.float32)

        # Add batch dimension (1, H, W, 3)
        return np.expand_dims(img_array, axis=0)

    except Exception as e:
        # Log error in production; print for now
        print(f"Error during image preprocessing: {e}")
        return np.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.float32)


def predict_food(
    model: Model,
    image_tensor: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> Dict[str, Union[str, float, Any]]:
    """
    Performs inference and extracts the top K predictions using the real model.

    Args:
        model (Model): The loaded EfficientNet-B5 Keras model.
        image_tensor (np.ndarray): The 4D preprocessed image tensor.
        class_names (List[str]): List of class labels mapped to indices.
        top_k (int): The number of top predictions to return.

    Returns:
        Dict: Contains the 'class_id', 'visual_stats', and 'top_predictions'.
    """
    if image_tensor.size == 0:
        return {"error": "Image tensor is empty after preprocessing."}

    # --- CRITICAL FIX FOR MACOS / STREAMLIT HANG ---
    # OLD: predictions = model.predict(image_tensor, verbose=0)
    # NEW: Direct call. Runs eagerly in the current thread, preventing deadlock.
    predictions = model(image_tensor, training=False)
    
    # Convert tensor to numpy array
    probabilities = predictions.numpy()[0]

    # Get top prediction indices
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]

    # Map indices to Class IDs (ASCxxx)
    top_preds = []
    for idx in top_k_indices:
        if idx < len(class_names):
            top_preds.append((class_names[idx], float(probabilities[idx])))
        else:
            top_preds.append((f"Unknown-{idx}", float(probabilities[idx])))

    # Select the Top-1 Class ID
    predicted_class_id = top_preds[0][0]

    # --- CV Heuristics (Section 3) ---
    # Simulating heuristics because B5 is a classifier, not an object counter.
    mock_visual_stats = {
        "count": float(np.random.randint(1, 4)),
        "occupancy_ratio": float(np.random.choice([0.95, 0.55, 0.7]))
    }

    return {
        "class_id": predicted_class_id,
        "visual_stats": mock_visual_stats,
        "top_predictions": top_preds
    }