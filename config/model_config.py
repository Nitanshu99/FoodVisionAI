"""
Model-Specific Configuration for FoodVisionAI

This module contains model architecture and training configurations.
"""

from config.settings import (
    INPUT_SHAPE, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, WEIGHT_DECAY, SEED
)


# ==============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ==============================================================================

MODEL_CONFIG = {
    # Input configuration
    'input_shape': INPUT_SHAPE,
    
    # Training configuration
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'seed': SEED,
    
    # Model architecture
    'backbone': 'EfficientNetV2B0',  # Can be changed to other backbones
    'dropout_rate': 0.2,
    'use_augmentation': True,
    
    # Optimizer configuration
    'optimizer': 'AdamW',
    'beta_1': 0.9,
    'beta_2': 0.999,
    'epsilon': 1e-7,
    
    # Learning rate schedule
    'use_lr_schedule': True,
    'lr_schedule_type': 'cosine',  # 'cosine', 'exponential', or 'step'
    
    # Early stopping
    'use_early_stopping': True,
    'patience': 10,
    'min_delta': 0.001,
    
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_accuracy',
    'mode': 'max',
}


# ==============================================================================
# YOLO CONFIGURATION
# ==============================================================================

YOLO_CONFIG = {
    'model_size': 'yolov8m-seg',  # Medium size YOLO model
    'confidence_threshold': 0.25,
    'iou_threshold': 0.45,
    'max_detections': 100,
    'image_size': 512,
}


# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================

LLM_CONFIG = {
    'model_name': 'qwen2.5-0.5b-instruct-fp16',
    'context_length': 2048,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 40,
    'max_tokens': 512,
    'repeat_penalty': 1.1,
}

