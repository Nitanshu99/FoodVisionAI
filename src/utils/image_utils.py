"""
Image Processing Utilities

Common image processing functions used across the codebase.
"""

from typing import List, Tuple
import numpy as np
import cv2
import tensorflow as tf
from keras import Model
import config


def process_crop(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Shared cropping logic for both training and inference.
    Ensures consistent preprocessing between data generation and model prediction.
    
    Args:
        image (np.ndarray): Input image array.
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).
    
    Returns:
        np.ndarray: Standardized crop resized to 512x512, or None if invalid.
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


def resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation=cv2.INTER_AREA) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int]): Target (width, height)
        interpolation: OpenCV interpolation method
    
    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    
    Args:
        image (np.ndarray): BGR image
    
    Returns:
        np.ndarray: RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    
    Args:
        image (np.ndarray): RGB image
    
    Returns:
        np.ndarray: BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image (np.ndarray): Input image (0-255)
    
    Returns:
        np.ndarray: Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0

