"""
Unit Tests for Models Module

Tests all model modules:
- builder.py
- augmentation.py
- loader.py
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import builder, augmentation, loader
import keras


class TestBuilder:
    """Test builder.py module"""
    
    def test_build_model_creates_model(self):
        """Test build_model creates a valid Keras model"""
        num_classes = 10
        
        model = builder.build_model(num_classes)
        
        assert model is not None
        assert isinstance(model, keras.Model)
        assert model.name == "FoodVision_B5"
    
    def test_build_model_correct_input_shape(self):
        """Test build_model has correct input shape"""
        model = builder.build_model(10)
        
        # Input shape should be (None, 512, 512, 3)
        assert model.input_shape == (None, 512, 512, 3)
    
    def test_build_model_correct_output_shape(self):
        """Test build_model has correct output shape"""
        num_classes = 25
        model = builder.build_model(num_classes)
        
        # Output shape should be (None, num_classes)
        assert model.output_shape == (None, num_classes)
    
    def test_build_model_is_compiled(self):
        """Test build_model returns compiled model"""
        model = builder.build_model(10)
        
        # Check if model is compiled (has optimizer)
        assert model.optimizer is not None
    
    def test_build_model_has_correct_layers(self):
        """Test build_model has expected layers"""
        model = builder.build_model(10)
        
        layer_names = [layer.name for layer in model.layers]
        
        # Check for key layers
        assert "input_layer" in layer_names
        assert "global_average_pooling" in layer_names
        assert "head_batch_norm" in layer_names
        assert "head_dropout" in layer_names
        assert "classification_output" in layer_names


class TestAugmentation:
    """Test augmentation.py module"""
    
    def test_random_gaussian_blur_initialization(self):
        """Test RandomGaussianBlur layer initialization"""
        layer = augmentation.RandomGaussianBlur(probability=0.5, kernel_sizes=[3, 5])
        
        assert layer.probability == 0.5
        assert layer.kernel_sizes == [3, 5]
    
    def test_random_gaussian_blur_call_training(self):
        """Test RandomGaussianBlur during training"""
        layer = augmentation.RandomGaussianBlur(probability=1.0)  # Always apply
        images = np.random.randint(0, 255, (1, 512, 512, 3), dtype=np.uint8)
        
        result = layer(images, training=True)
        
        assert result is not None
        assert result.shape == (1, 512, 512, 3)
    
    def test_random_gaussian_blur_call_inference(self):
        """Test RandomGaussianBlur during inference (no blur)"""
        layer = augmentation.RandomGaussianBlur(probability=1.0)
        images = np.random.randint(0, 255, (1, 512, 512, 3), dtype=np.uint8)
        
        result = layer(images, training=False)
        
        # Should return original images during inference
        assert result is not None
        assert result.shape == (1, 512, 512, 3)
    
    def test_random_gaussian_blur_get_config(self):
        """Test RandomGaussianBlur get_config"""
        layer = augmentation.RandomGaussianBlur(probability=0.3, kernel_sizes=[3, 5])
        
        config = layer.get_config()
        
        assert "probability" in config
        assert "kernel_sizes" in config
        assert config["probability"] == 0.3
        assert config["kernel_sizes"] == [3, 5]
    
    def test_get_augmentation_pipeline(self):
        """Test get_augmentation_pipeline creates valid pipeline"""
        pipeline = augmentation.get_augmentation_pipeline()
        
        assert pipeline is not None
        assert isinstance(pipeline, keras.Sequential)
        assert pipeline.name == "augmentation_pipeline"
    
    def test_augmentation_pipeline_has_layers(self):
        """Test augmentation pipeline has expected layers"""
        pipeline = augmentation.get_augmentation_pipeline()
        
        layer_names = [layer.name for layer in pipeline.layers]
        
        assert "random_flip" in layer_names
        assert "random_rotation" in layer_names
        assert "random_zoom" in layer_names
        assert "random_contrast" in layer_names
        assert "random_brightness" in layer_names
        assert "random_gaussian_blur" in layer_names
    
    def test_get_gaussian_kernel(self):
        """Test get_gaussian_kernel generates correct shape"""
        kernel = augmentation.get_gaussian_kernel(size=5, sigma=1.0)
        
        # Should be [5, 5, 3, 1] for RGB
        assert kernel.shape == (5, 5, 3, 1)


class TestLoader:
    """Test loader.py module"""
    
    def test_save_and_load_model(self):
        """Test save_model and load_model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple model
            model = builder.build_model(10)
            save_path = Path(tmpdir) / "test_model.keras"
            
            # Save model
            saved_path = loader.save_model(model, save_path)
            
            assert saved_path.exists()
            
            # Load model
            loaded_model = loader.load_model(saved_path, compile=False)

            assert loaded_model is not None
            assert isinstance(loaded_model, keras.Model)

    def test_load_model_file_not_found(self):
        """Test load_model raises error for non-existent file"""
        with pytest.raises(FileNotFoundError):
            loader.load_model(Path("/nonexistent/model.keras"))

    def test_save_model_creates_directory(self):
        """Test save_model creates parent directory if needed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = builder.build_model(10)
            save_path = Path(tmpdir) / "subdir" / "model.keras"

            loader.save_model(model, save_path)

            assert save_path.exists()
            assert save_path.parent.exists()

    def test_save_model_overwrite_false(self):
        """Test save_model with overwrite=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = builder.build_model(10)
            save_path = Path(tmpdir) / "model.keras"

            # Save first time
            loader.save_model(model, save_path)

            # Try to save again with overwrite=False
            with pytest.raises(FileExistsError):
                loader.save_model(model, save_path, overwrite=False)

    def test_load_model_with_custom_objects(self):
        """Test load_model with custom objects"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model with augmentation
            model = builder.build_model(10)
            save_path = Path(tmpdir) / "model.keras"

            loader.save_model(model, save_path)

            # Load with custom objects
            custom_objects = {"RandomGaussianBlur": augmentation.RandomGaussianBlur}
            loaded_model = loader.load_model(save_path, custom_objects=custom_objects, compile=False)

            assert loaded_model is not None

