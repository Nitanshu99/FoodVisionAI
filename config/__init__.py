"""
FoodVisionAI Configuration Package

This package centralizes all configuration for the FoodVisionAI system:
- settings: Core hyperparameters and constants
- paths: All file and directory paths
- model_config: Model-specific configurations
- hardware: Hardware detection and auto-configuration
"""

# Import all configuration modules for easy access
from config.settings import *
from config.paths import *
from config.model_config import *
from config import hardware

# Expose hardware auto-config function
from config.hardware import get_auto_config

__all__ = [
    # From settings
    'IMG_HEIGHT', 'IMG_WIDTH', 'IMG_SIZE', 'INPUT_SHAPE',
    'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'SEED',
    'PLATE_DIAMETER_CM', 'DEVICE', 'get_device',
    
    # From paths
    'BASE_DIR', 'DATA_DIR', 'PROCESSED_DIR', 'TRAIN_DIR', 'VAL_DIR',
    'PARQUET_DB_DIR', 'LOGS_DIR', 'MODELS_DIR', 'CHECKPOINT_DIR',
    'MODEL_GLOBAL_PATH', 'MODEL_LOCAL_PATH', 'FINAL_MODEL_PATH',
    'YOLO_MODEL_PATH', 'LLM_MODEL_PATH', 'LABELS_PATH', 'DB_FILES',
    
    # From model_config
    'MODEL_CONFIG',
    
    # From hardware
    'hardware', 'get_auto_config',
]

