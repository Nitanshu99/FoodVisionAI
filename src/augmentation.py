"""
Data Augmentation Strategy (The "Regularization Shield").

This module implements the stochastic "Variance Injection" pipeline specified
in Section 2.1 of the Architectural Specifications.

Verified Compatibility:
    - TensorFlow 2.16 (Uses native TF Ops for Blur)
    - Keras 3 (Uses @register_keras_serializable and tf.cond for Graph Safety)
"""

import tensorflow as tf
import keras
from keras import layers

def get_gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel suitable for depthwise_conv2d.
    Output Shape: [size, size, 3, 1] (Applied independently to RGB channels)
    """
    d = tf.cast(size, tf.float32)
    x = tf.range(d) - (d - 1) / 2.0
    x = tf.square(x)
    
    # Calculate Gaussian function G(x) = exp(-x^2 / (2*sigma^2))
    g = tf.exp(-x / (2.0 * tf.square(sigma)))
    
    # Normalize to prevent darkening/brightening
    g = g / tf.reduce_sum(g)
    
    # Make 2D kernel by outer product: G(x,y) = G(x) * G(y)
    g_2d = tf.einsum('i,j->ij', g, g)
    
    # Expand dims to match depthwise_conv2d: [H, W, In_Channels, Channel_Mult]
    # Shape: [Size, Size, 1, 1]
    g_2d = g_2d[:, :, tf.newaxis, tf.newaxis]
    
    # Tile for RGB (3 channels): [Size, Size, 3, 1]
    g_2d = tf.tile(g_2d, [1, 1, 3, 1])
    
    return g_2d

@keras.saving.register_keras_serializable()
class RandomGaussianBlur(layers.Layer):
    """
    Applies Gaussian Blur with 30% probability using Depthwise Convolution.
    
    Specs:
        - Kernel Sizes: 3x3 or 5x5
        - Probability: 0.3
        - Implementation: Graph-Safe tf.cond + tf.nn.depthwise_conv2d
    """

    def __init__(self, probability=0.3, kernel_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3, 5]

    def call(self, images, training=True):
        # Augmentation only runs during training
        if not training:
            return images

        # Define the Blur Operation as a graph-callable function
        def apply_blur():
            # Randomly select a kernel size from the list
            # We pick an index [0, len(kernel_sizes))
            k_idx = tf.random.uniform([], maxval=len(self.kernel_sizes), dtype=tf.int32)
            k_size = tf.gather(self.kernel_sizes, k_idx)
            
            # Sigma heuristic: sigma = k_size / 3.0
            sigma = tf.cast(k_size, tf.float32) / 3.0
            
            # Generate Kernel
            kernel = get_gaussian_kernel(k_size, sigma)
            
            # Ensure images are float32 for convolution
            images_f32 = tf.cast(images, tf.float32)
            
            # Apply Depthwise Conv (Independent blur per channel)
            blurred = tf.nn.depthwise_conv2d(
                images_f32, 
                kernel, 
                strides=[1, 1, 1, 1], 
                padding='SAME'
            )
            return blurred

        # Graph-Safe Conditional Logic
        # if random < prob -> apply_blur() else -> return original images
        return tf.cond(
            tf.random.uniform([]) < self.probability,
            apply_blur,
            lambda: tf.cast(images, tf.float32) # Ensure consistent dtype return
        )

    def compute_output_shape(self, input_shape):
        # Blur does not change tensor shape
        return input_shape

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            "probability": self.probability,
            "kernel_sizes": self.kernel_sizes,
        })
        return config_dict


def get_augmentation_pipeline():
    """
    Constructs the complete Keras Sequential augmentation pipeline.
    """
    # Define degree factor: 20 degrees / 360 degrees = ~0.055
    rotation_factor = 20.0 / 360.0

    return keras.Sequential(
        [
            # 1. Geometric Transformations
            layers.RandomFlip("horizontal", name="random_flip"),
            layers.RandomRotation(factor=rotation_factor, name="random_rotation"),
            layers.RandomZoom(
                height_factor=(-0.2, 0.2), 
                width_factor=(-0.2, 0.2),
                name="random_zoom"
            ),

            # 2. Photometric Transformations
            layers.RandomContrast(factor=0.2, name="random_contrast"),
            layers.RandomBrightness(factor=0.2, name="random_brightness"),

            # 3. Blur Transformations (Custom Layer)
            RandomGaussianBlur(
                probability=0.3, 
                kernel_sizes=[3, 5], 
                name="random_gaussian_blur"
            ),
        ],
        name="augmentation_pipeline"
    )