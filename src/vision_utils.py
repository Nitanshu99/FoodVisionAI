"""
Vision Utility Module.

Contains pre-processing and inference functions required for the
EfficientNet-B5 model. This handles the high-resolution input requirement
and prepares the image for the model.
"""

import numpy as np
import tensorflow as tf
from keras import Model
from PIL import Image
from typing import Tuple, List, Dict, Union
from src import config

def preprocess_image(image_file: bytes) -> np.ndarray:
    """
    Loads, resizes, and normalizes a raw image file for the B5 model.

    Compliance: Ensures input is 512x512 pixels as required by the architecture.
    
    Args:
        image_file (bytes): The raw image content as bytes (e.g., from st.file_uploader).

    Returns:
        np.ndarray: The preprocessed image as a 4D tensor (1, H, W, 3).
    """
    try:
        # Load image using PIL
        img = Image.open(image_file).convert("RGB")
        
        # Resize to the required 512x512 resolution (Section 2)
        target_size: Tuple[int, int] = config.IMG_SIZE
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.asarray(img, dtype=np.float32)
        
        # Normalize: EfficientNet pre-trained models typically expect values in [0, 255]
        # or normalized to specific mean/std, but for simplicity here, we keep it standard.
        # EfficientNet-B5 internally handles normalization if using keras.applications.
        
        # Add batch dimension (1, H, W, 3)
        return np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        # Return a tensor of zeros on failure
        return np.zeros((1, config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.float32)


def predict_food(model: Model, image_tensor: np.ndarray, top_k: int = 3) -> Dict[str, Union[str, float]]:
    """
    Performs inference and extracts the top K predictions.

    Note: In a real scenario, this function would return the actual ASCxxx code
    and a placeholder for instance counting/ratio.

    Args:
        model (Model): The loaded EfficientNet-B5 Keras model.
        image_tensor (np.ndarray): The 4D preprocessed image tensor.
        top_k (int): The number of top predictions to return.

    Returns:
        Dict: Contains the top prediction (class_id) and mock visual stats.
    """
    if image_tensor.size == 0:
        return {"error": "Image tensor is empty after preprocessing."}

    # Perform prediction
    predictions = model.predict(image_tensor, verbose=0)
    
    # Get top prediction indices
    top_k_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    # --- Mocking Prediction and CV Heuristics (Day 3 Requirement) ---
    # Since we are not running a full CV model (which would detect ASCxxx code
    # and provide the count/ratio), we mock the required output here.
    
    # Mocking the Top-1 Class ID for the Logic Engine (Section 3)
    # Using 'ASC004' (Iced Tea) and 'ASC171' (Aloo Gobi) as examples from the README.
    mock_class_ids = ['ASC004', 'ASC171', 'ASC001'] 
    
    # The actual Top-1 class ID would be mapped from the index
    # For now, we manually pick the first mock ID as the prediction.
    predicted_class_id = mock_class_ids[0] if top_k_indices[0] % 2 == 0 else mock_class_ids[1]

    # Mocking the CV Heuristics required by NutrientEngine (Section 3)
    # The heuristic needs to be realistic for both Discrete (count) and Container (ratio)
    mock_visual_stats = {
        # Instance Count for 'Piece' type foods
        "count": float(np.random.randint(1, 4)), 
        # Fill-Level Ratio for 'Bowl' type foods
        # Mock values follow the README thresholds (R > 0.8, R < 0.6)
        "occupancy_ratio": float(np.random.choice([0.95, 0.55, 0.7])) 
    }
    
    # Returning the final prediction output structure
    return {
        "class_id": predicted_class_id,
        "visual_stats": mock_visual_stats,
        "top_predictions": [(f"ASC{idx:03d}", float(predictions[0][idx])) for idx in top_k_indices]
    }