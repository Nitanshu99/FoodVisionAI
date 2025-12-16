"""
Core Settings and Hyperparameters for FoodVisionAI

This module contains all hyperparameters, constants, and device configuration.
"""

import tensorflow as tf


# ==============================================================================
# IMAGE SETTINGS
# ==============================================================================

# Input Resolution: 512 x 512 pixels
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)


# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================

# Batch Size
BATCH_SIZE = 56

# Training Epochs
EPOCHS = 50

# Optimizer Settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Random Seed for Reproducibility
SEED = 42


# ==============================================================================
# GEOMETRIC HEURISTICS
# ==============================================================================

# Standard Plate Diameter in Centimeters (for portion estimation)
PLATE_DIAMETER_CM = 28.0


# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================

def get_device():
    """
    Returns the primary available device (GPU, MPS, or CPU).
    
    Returns:
        str: Device type ('GPU', 'MPS', or 'CPU')
    """
    try:
        # Check for NVIDIA GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "GPU"
    except:
        pass
    
    try:
        # Check for Apple Metal (MPS)
        mps = tf.config.list_physical_devices('MPS')
        if mps:
            return "MPS"
    except:
        pass
    
    return "CPU"


# Primary device for TensorFlow operations
DEVICE = get_device()

