"""
Configuration module for FoodVisionAI.

This module centralizes all hyperparameters, file paths, and constant definitions
derived from the Architectural Specifications.

Specifications:
    - Input Resolution: 512x512
    - Batch Size: 64
    - Optimizer settings: AdamW, WD=1e-4, LR=1e-3
"""

import os
from pathlib import Path

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Base directory relative to this config file (assuming src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories based on Project Structure
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
PARQUET_DB_DIR = PROCESSED_DIR / "parquet_db"

# Model artifacts
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_PATH = MODELS_DIR / "food_vision_b5.keras"

# Ensure output directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# HYPERPARAMETERS (Section 2: Architectural Specifications)
# ==============================================================================

# Input Resolution: 512 x 512 pixels
# Justification: Required to detect fine-grained textures like oil separation.
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Batch Size: 64
# Justification: Optimal balance of speed and Batch Norm stability on A6000 (48GB VRAM).
BATCH_SIZE = 56

# Optimizer Settings
# Optimizer: AdamW
# Learning Rate: 1e-3 with Cosine Decay Schedule
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Training Settings
# Precision: FP32 (Standard 32-bit Floating Point)
USE_MIXED_PRECISION = False  # Set to True only if FP16 is explicitly requested

# Seed for reproducibility
SEED = 42

# ==============================================================================
# DATABASE CONFIGURATION (Section 1: Data Strategy)
# ==============================================================================
# Append this to the end of src/config.py

# Parquet filenames matching README Section 6 Metadata
DB_FILES = {
    "nutrition": "INDB.parquet",             # Nutritional values per 100g
    "recipes": "recipes.parquet",            # Recipe metadata
    "serving_size": "recipes_servingsize.parquet", # Serving unit types (Bowl vs Piece)
    "units": "Units.parquet"                 # Unit conversions
}