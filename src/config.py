"""
Configuration module for FoodVisionAI.

This module centralizes all hyperparameters, file paths, and constant definitions.
"""

import os
from pathlib import Path
import tensorflow as tf  # <--- ADDED THIS IMPORT

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Base directory relative to this config file (assuming src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories based on Project Structure
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Note: Unified Training script expects these to point to your new YOLO data
# If you are training the Local Model, ensure these point to 'yolo_processed'
TRAIN_DIR = DATA_DIR / "yolo_processed" / "train"  # <--- UPDATED for YOLO Training
VAL_DIR = DATA_DIR / "yolo_processed" / "val"      # <--- UPDATED for YOLO Training
PARQUET_DB_DIR = PROCESSED_DIR / "parquet_db"

# Logs directory for JSON inference data
LOGS_DIR = DATA_DIR / "inference_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Model artifacts
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# --- DUAL MODEL STRATEGY ---
# 1. Global Model (Context-Aware): Trained on Full Images
MODEL_GLOBAL_PATH = CHECKPOINT_DIR / "model_best.keras"

# 2. Local Model (Crop-Specialist): Trained on Tight Crops (YOLO Processed)
# If this training isn't finished yet, the app will fallback to the Global one automatically.
MODEL_LOCAL_PATH = CHECKPOINT_DIR / "model_yolo_best.keras"

# For backward compatibility with single-model scripts
FINAL_MODEL_PATH = MODEL_GLOBAL_PATH 

# YOLO Segmentation Model Path (Offline)
YOLO_MODEL_PATH = MODELS_DIR / "yolov8m-seg.pt"

# Ensure output directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

# Input Resolution: 512 x 512 pixels
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Batch Size
BATCH_SIZE = 56
EPOCHS = 50  # <--- ADDED DEFAULT EPOCHS

# Optimizer Settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
USE_MIXED_PRECISION = False
SEED = 42

# ==============================================================================
# GEOMETRIC HEURISTICS
# ==============================================================================

# Standard Plate Diameter in Centimeters
PLATE_DIAMETER_CM = 28.0 

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
LABELS_PATH = BASE_DIR / "class_names.npy"

# Database Registry
DB_FILES = {
    "nutrition": "INDB.parquet",
    "recipes": "recipes.parquet",
    "serving_size": "recipes_servingsize.parquet",
    "units": "Units.parquet",
    "links": "recipe_links.parquet"
}

# ==============================================================================
# DEVICE CONFIGURATION (ADDED)
# ==============================================================================
def get_device():
    """Returns the primary available device (GPU or CPU)."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "GPU"
    except:
        pass
    return "CPU"

DEVICE = get_device()