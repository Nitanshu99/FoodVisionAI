"""
Model Builder Module

Constructs the High-Fidelity EfficientNet-B5 model specified in
Section 2 of the Architectural Specifications.

Specifications:
    - Backbone: EfficientNet-B5 (Pretrained on ImageNet)
    - Input Resolution: 512 x 512 pixels
    - Activation: Swish (SiLU) for internal layers
    - Optimizer: AdamW (Weight Decay: 1e-4)
"""

import tensorflow as tf
import keras
from keras import layers, applications, optimizers, losses, metrics
import config


def build_model(num_classes: int) -> keras.Model:
    """
    Constructs and compiles the EfficientNet-B5 model.

    The model utilizes Transfer Learning with a frozen backbone initially,
    adapted for the specific 512x512 input resolution required for texture detection.

    Args:
        num_classes (int): The number of food categories (from folder structure).

    Returns:
        keras.Model: The compiled Keras model ready for training.
    """
    # 1. Input Layer
    # Explicitly set to 512x512 to match Architectural Specs
    inputs = layers.Input(shape=config.INPUT_SHAPE, name="input_layer")

    # 2. Backbone: EfficientNet-B5
    # Note: EfficientNet uses Swish activation internally by default.
    # We load ImageNet weights to leverage learned feature extractors.
    backbone = applications.EfficientNetB5(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling=None  # We will add our own GlobalAveragePooling
    )

    # Freeze the backbone to prevent destroying pretrained weights during initial training
    # (Optional: Can be unfreezed later for fine-tuning)
    backbone.trainable = False

    # 3. Classification Head
    # Extract features using the backbone
    x = backbone.output

    # Pooling: Condense 512x512 feature maps into a vector
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)

    # Batch Normalization for stability
    x = layers.BatchNormalization(name="head_batch_norm")(x)

    # Dropout for regularization (standard with EfficientNet heads)
    x = layers.Dropout(0.2, name="head_dropout")(x)

    # Output Layer
    # Softmax for multi-class classification
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output"
    )(x)

    # 4. Model Assembly
    model = keras.Model(inputs, outputs, name="FoodVision_B5")

    # 5. Compilation
    # Optimizer: AdamW with Weight Decay 1e-4
    # Learning Rate: 1e-3
    optimizer = optimizers.AdamW(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer, #type: ignore
        loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            metrics.CategoricalAccuracy(name="accuracy"),
            metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
        ]
    )

    return model


if __name__ == "__main__":
    # Sanity check
    # Arbitrary number of classes for testing (e.g., 100)
    test_model = build_model(num_classes=100)
    test_model.summary()
    print(f"Model built with input shape: {test_model.input_shape}") #pylint: disable=E1101 #type: ignore

