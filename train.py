"""
Main Training Execution Script (Resumable).

Feature:
    - Automatically detects existing checkpoints.
    - Resumes training if interrupted (power loss, crash).
"""

import os
import numpy as np
import tensorflow as tf
import keras
from keras import callbacks
from src import config
from src import augmentation
from src import vision_model

# Force the exact same class order as the Global Model
MASTER_CLASSES = np.load(config.LABELS_PATH).tolist()

# Constants
AUTOTUNE = tf.data.AUTOTUNE # pylint: disable=E1101 # type: ignore

def load_dataset(directory: str, is_training: bool = False):
    """
    Loads and preprocesses the image dataset.
    Returns: (dataset, class_names)
    """
    raw_dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=MASTER_CLASSES,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMG_SIZE,
        shuffle=True,
        seed=config.SEED,
        interpolation="bilinear"
    )

    class_names = raw_dataset.class_names
    
    dataset = raw_dataset
    if is_training:
        aug_pipeline = augmentation.get_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (aug_pipeline(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset, class_names

def main():
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Training on device: {tf.config.list_physical_devices('GPU')}")

    # 1. Load Data
    print("\n--- Loading Data ---")
    train_ds, train_class_names = load_dataset(config.TRAIN_DIR, is_training=True)
    val_ds, val_class_names = load_dataset(config.VAL_DIR, is_training=False)
    num_classes = len(train_class_names)

    # 2. Build or Load Model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_best.keras")
    initial_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"\n--- Resuming from Checkpoint: {checkpoint_path} ---")
        # Load the model with custom objects (needed for RandomGaussianBlur)
        # Note: We rebuild the model structure first to ensure compilation is clean, 
        # or load_model directly if saving full model. 
        # For safety/consistency with custom layers, we often rebuild then load weights.
        
        model = vision_model.build_model(num_classes=num_classes)
        # We need to compile it before loading weights to ensure optimizer state matches if possible,
        # but load_weights typically handles weights only. 
        # To resume optimizer state fully, we usually need 'model = keras.models.load_model(...)'.
        
        try:
            # Try loading the full model (weights + optimizer state)
            # We must pass custom_objects so Keras knows what 'RandomGaussianBlur' is
            custom_objects = {"RandomGaussianBlur": augmentation.RandomGaussianBlur}
            model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
            print(">> Model and Optimizer State Loaded Successfully.")
            
            # Heuristic to guess 'initial_epoch': 
            # We don't explicitly save epoch count in .keras, so we assume 
            # the user tracks it or we start from 0 but with better weights.
            # However, to be cleaner, we can just let it run.
            # If you strictly want the epoch number to update, you'd need to save it externally.
            # For now, we will rely on the improved weights.
            
        except Exception as e:
            print(f">> Warning: Could not load full model state ({e}). Rebuilding and loading weights only.")
            model = vision_model.build_model(num_classes=num_classes)
            model.load_weights(checkpoint_path)
    else:
        print("\n--- Building New Model ---")
        model = vision_model.build_model(num_classes=num_classes)
    
    # 3. Define Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 4. Train
    print("\n--- Starting Training Job ---")
    # Note: Keras 'initial_epoch' argument helps logging, but doesn't magically fast-forward 
    # the optimizer if the state wasn't loaded. 
    # Since we loaded the full model above, the optimizer is resumed too.
    
    history = model.fit(
        train_ds,
        epochs=50,
        initial_epoch=initial_epoch, # Useful if we tracked epoch count externally
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    # 5. Save Final Model
    print(f"\n--- Saving Final Model to {config.FINAL_MODEL_PATH} ---")
    model.save(config.FINAL_MODEL_PATH)

if __name__ == "__main__":
    main()