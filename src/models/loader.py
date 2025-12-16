"""
Model Loader Module

Provides utilities for loading and saving Keras models with custom objects.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import keras
from src.models.augmentation import RandomGaussianBlur


def load_model(
    model_path: Path,
    custom_objects: Optional[Dict[str, Any]] = None,
    compile: bool = False
) -> keras.Model:
    """
    Load a Keras model from disk with custom objects.
    
    Args:
        model_path (Path): Path to the saved model file
        custom_objects (Optional[Dict[str, Any]]): Custom objects to register.
            If None, defaults to {"RandomGaussianBlur": RandomGaussianBlur}
        compile (bool): Whether to compile the model after loading
    
    Returns:
        keras.Model: Loaded Keras model
    
    Raises:
        FileNotFoundError: If model_path does not exist
        ValueError: If model cannot be loaded
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Default custom objects
    if custom_objects is None:
        custom_objects = {"RandomGaussianBlur": RandomGaussianBlur}
    
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=compile
        )
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}") from e


def save_model(
    model: keras.Model,
    save_path: Path,
    overwrite: bool = True
) -> Path:
    """
    Save a Keras model to disk.
    
    Args:
        model (keras.Model): Model to save
        save_path (Path): Path where to save the model
        overwrite (bool): Whether to overwrite existing file
    
    Returns:
        Path: Path where model was saved
    
    Raises:
        FileExistsError: If file exists and overwrite=False
        ValueError: If model cannot be saved
    """
    if save_path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {save_path}")
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        model.save(save_path)
        return save_path
    except Exception as e:
        raise ValueError(f"Failed to save model to {save_path}: {e}") from e

