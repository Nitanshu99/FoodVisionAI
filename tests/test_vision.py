"""
Unit Tests for Vision Module

Tests all vision modules:
- inference.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tensorflow as tf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision import inference
import keras


class TestInference:
    """Test inference.py module"""
    
    def test_predict_food_with_single_object(self):
        """Test predict_food with single detected object"""
        # Mock model
        mock_model = Mock(spec=keras.Model)
        # Return TensorFlow tensor (not numpy array)
        mock_model.return_value = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        
        # Mock assessor
        mock_assessor = Mock()
        mock_assessor.analyze_scene.return_value = (
            [{
                'bbox': (10, 10, 100, 100),
                'mask': np.ones((512, 512), dtype=np.uint8),
                'area_pixels': 8100,
                'area_cm2': 100.0,
                'confidence': 0.95
            }],
            9.0  # ppm
        )
        
        # Mock background remover
        with patch('src.vision.inference.BackgroundRemover') as mock_bg_remover:
            mock_bg_instance = Mock()
            mock_bg_instance.process_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_bg_remover.return_value = mock_bg_instance
            
            # Create test image
            raw_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            class_names = ['apple', 'banana', 'orange']
            
            # Run prediction
            results = inference.predict_food(mock_model, mock_assessor, raw_image, class_names)
            
            # Assertions
            assert len(results) == 1
            assert 'class_id' in results[0]
            assert 'top_predictions' in results[0]
            assert 'visual_stats' in results[0]
            assert 'crop_type' in results[0]
    
    def test_predict_food_with_multiple_objects(self):
        """Test predict_food with multiple detected objects"""
        # Mock model
        mock_model = Mock(spec=keras.Model)
        mock_model.return_value = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        
        # Mock assessor with 3 objects
        mock_assessor = Mock()
        mock_assessor.analyze_scene.return_value = (
            [
                {'bbox': (10, 10, 100, 100), 'mask': np.ones((512, 512)), 'area_pixels': 8100, 'area_cm2': 100.0, 'confidence': 0.95},
                {'bbox': (150, 150, 250, 250), 'mask': np.ones((512, 512)), 'area_pixels': 10000, 'area_cm2': 120.0, 'confidence': 0.90},
                {'bbox': (300, 300, 400, 400), 'mask': np.ones((512, 512)), 'area_pixels': 10000, 'area_cm2': 120.0, 'confidence': 0.85}
            ],
            9.0
        )
        
        with patch('src.vision.inference.BackgroundRemover') as mock_bg_remover:
            mock_bg_instance = Mock()
            mock_bg_instance.process_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_bg_remover.return_value = mock_bg_instance
            
            raw_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            class_names = ['apple', 'banana', 'orange']
            
            results = inference.predict_food(mock_model, mock_assessor, raw_image, class_names)
            
            # Should process all 3 objects
            assert len(results) == 3
    
    def test_predict_food_fallback_no_objects(self):
        """Test predict_food fallback when no objects detected"""
        # Mock model
        mock_model = Mock(spec=keras.Model)
        mock_model.return_value = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        
        # Mock assessor with no objects
        mock_assessor = Mock()
        mock_assessor.analyze_scene.return_value = ([], 9.0)
        
        with patch('src.vision.inference.BackgroundRemover') as mock_bg_remover:
            mock_bg_instance = Mock()
            mock_bg_instance.process_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_bg_remover.return_value = mock_bg_instance
            
            raw_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            class_names = ['apple', 'banana', 'orange']
            
            results = inference.predict_food(mock_model, mock_assessor, raw_image, class_names)
            
            # Should use fallback (full image)
            assert len(results) == 1
            assert results[0]['crop_type'] == "Full Image + U2Net (Fallback)"
            assert 'error' in results[0]['visual_stats']
    
    def test_predict_food_filters_max_segments(self):
        """Test predict_food filters to MAX_SEGMENTS (8)"""
        # Mock model
        mock_model = Mock(spec=keras.Model)
        mock_model.return_value = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)
        
        # Create 12 objects (should filter to 8)
        objects = []
        for i in range(12):
            objects.append({
                'bbox': (i*10, i*10, i*10+50, i*10+50),
                'mask': np.ones((512, 512)),
                'area_pixels': 10000 - i*100,  # Decreasing area
                'area_cm2': 100.0,
                'confidence': 0.9
            })
        
        mock_assessor = Mock()
        mock_assessor.analyze_scene.return_value = (objects, 9.0)
        
        with patch('src.vision.inference.BackgroundRemover') as mock_bg_remover:
            mock_bg_instance = Mock()
            mock_bg_instance.process_image.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_bg_remover.return_value = mock_bg_instance
            
            raw_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            class_names = ['apple', 'banana', 'orange']
            
            results = inference.predict_food(mock_model, mock_assessor, raw_image, class_names)
            
            # Should only process top 8 largest
            assert len(results) <= 8

