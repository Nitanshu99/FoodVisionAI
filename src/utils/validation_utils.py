"""
Validation Utilities

Common validation functions for images, bounding boxes, and data.
"""

from typing import Tuple, Optional
import numpy as np


def validate_image(image: np.ndarray, min_size: int = 30) -> bool:
    """
    Validate that image is a valid numpy array with proper dimensions.
    
    Args:
        image (np.ndarray): Image to validate
        min_size (int, optional): Minimum width/height. Defaults to 30.
    
    Returns:
        bool: True if image is valid
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3:
        height, width, channels = image.shape
        if channels not in [1, 3, 4]:
            return False
    else:
        height, width = image.shape
    
    if height < min_size or width < min_size:
        return False
    
    return True


def validate_bbox(bbox: Tuple[int, int, int, int], image_shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2)
        image_shape (Optional[Tuple[int, int]]): Image (height, width) for bounds checking
    
    Returns:
        bool: True if bbox is valid
    """
    if bbox is None or len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    # Check that coordinates are valid
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Check that coordinates are non-negative
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False
    
    # Check bounds if image shape provided
    if image_shape is not None:
        height, width = image_shape
        if x2 > width or y2 > height:
            return False
    
    return True


def validate_crop(crop: np.ndarray, min_size: int = 30) -> bool:
    """
    Validate that crop is valid and meets minimum size requirements.
    
    Args:
        crop (np.ndarray): Cropped image
        min_size (int, optional): Minimum width/height. Defaults to 30.
    
    Returns:
        bool: True if crop is valid
    """
    if crop is None:
        return False
    
    if not isinstance(crop, np.ndarray):
        return False
    
    if crop.size == 0:
        return False
    
    if len(crop.shape) < 2:
        return False
    
    height, width = crop.shape[:2]
    
    if height < min_size or width < min_size:
        return False
    
    return True


def validate_confidence(confidence: float, min_threshold: float = 0.0, max_threshold: float = 1.0) -> bool:
    """
    Validate confidence score is in valid range.
    
    Args:
        confidence (float): Confidence score
        min_threshold (float, optional): Minimum threshold. Defaults to 0.0.
        max_threshold (float, optional): Maximum threshold. Defaults to 1.0.
    
    Returns:
        bool: True if confidence is valid
    """
    try:
        conf = float(confidence)
        return min_threshold <= conf <= max_threshold
    except (ValueError, TypeError):
        return False


def validate_mask(mask: np.ndarray, expected_shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    Validate segmentation mask.
    
    Args:
        mask (np.ndarray): Segmentation mask
        expected_shape (Optional[Tuple[int, int]]): Expected (height, width)
    
    Returns:
        bool: True if mask is valid
    """
    if mask is None:
        return False
    
    if not isinstance(mask, np.ndarray):
        return False
    
    if mask.size == 0:
        return False
    
    if len(mask.shape) != 2:
        return False
    
    if expected_shape is not None:
        if mask.shape != expected_shape:
            return False
    
    return True


def validate_class_id(class_id: str, valid_classes: list) -> bool:
    """
    Validate that class ID is in list of valid classes.
    
    Args:
        class_id (str): Class ID to validate
        valid_classes (list): List of valid class IDs
    
    Returns:
        bool: True if class ID is valid
    """
    if class_id is None:
        return False
    
    if not isinstance(class_id, str):
        return False
    
    return class_id in valid_classes

