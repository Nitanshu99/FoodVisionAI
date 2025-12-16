"""
Path Configuration for FoodVisionAI

This module centralizes all file paths and directory locations.
All paths are relative to the project root directory.
"""

import os
from pathlib import Path


# ==============================================================================
# BASE DIRECTORY
# ==============================================================================

# Base directory (project root) - config/ is at root level
BASE_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# DATA DIRECTORIES
# ==============================================================================

# Main data directory
DATA_DIR = BASE_DIR / "data"

# Processed data directory
PROCESSED_DIR = DATA_DIR / "processed"

# Training and validation directories (YOLO processed data)
TRAIN_DIR = DATA_DIR / "yolo_processed" / "train"
VAL_DIR = DATA_DIR / "yolo_processed" / "val"

# Parquet database directory
PARQUET_DB_DIR = PROCESSED_DIR / "parquet_db"

# Logs directory for JSON inference data
LOGS_DIR = DATA_DIR / "inference_logs"


# ==============================================================================
# MODEL DIRECTORIES
# ==============================================================================

# Main models directory
MODELS_DIR = BASE_DIR / "models"

# Model checkpoints directory
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


# ==============================================================================
# MODEL PATHS
# ==============================================================================

# --- DUAL MODEL STRATEGY ---

# 1. Global Model (Context-Aware): Trained on Full Images
MODEL_GLOBAL_PATH = CHECKPOINT_DIR / "model_best.keras"

# 2. Local Model (Crop-Specialist): Trained on Tight Crops (YOLO Processed)
MODEL_LOCAL_PATH = CHECKPOINT_DIR / "model_yolo_best.keras"

# For backward compatibility with single-model scripts
FINAL_MODEL_PATH = MODEL_GLOBAL_PATH

# YOLO Segmentation Model Path (Offline)
YOLO_MODEL_PATH = MODELS_DIR / "yolov8m-seg.pt"

# LLM Model Path (Offline GGUF)
LLM_MODEL_PATH = MODELS_DIR / "qwen2.5-0.5b-instruct-fp16.gguf"


# ==============================================================================
# DATABASE PATHS
# ==============================================================================

# Class labels path
LABELS_PATH = BASE_DIR / "class_names.npy"

# Database file registry
DB_FILES = {
    "nutrition": "INDB.parquet",
    "recipes": "recipes.parquet",
    "serving_size": "recipes_servingsize.parquet",
    "units": "Units.parquet",
    "links": "recipe_links.parquet"
}


# ==============================================================================
# DIRECTORY CREATION
# ==============================================================================

# Ensure critical directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

