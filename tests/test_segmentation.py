"""
Unit Tests for Segmentation Module

Tests all segmentation modules:
- assessor.py (DietaryAssessor class)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.segmentation import assessor


class TestDietaryAssessor:
    """Test assessor.py module"""

    @patch('src.segmentation.assessor.YOLO')
    def test_dietary_assessor_initialization(self, mock_yolo):
        """Test DietaryAssessor initializes correctly"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        assessor_instance = assessor.DietaryAssessor()

        assert assessor_instance is not None
        assert assessor_instance.model is not None
        mock_yolo.assert_called_once()

    @patch('src.segmentation.assessor.YOLO')
    def test_detect_plate_and_scale_with_plate(self, mock_yolo):
        """Test detect_plate_and_scale when plate is detected"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # Create mock results with masks
        mock_result = Mock()
        mock_mask = Mock()
        mock_mask.cpu.return_value.numpy.return_value.astype.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_result.masks = Mock()
        mock_result.masks.data = [mock_mask]
        mock_model.return_value = [mock_result]

        assessor_instance = assessor.DietaryAssessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        ppm = assessor_instance.detect_plate_and_scale(test_image)

        assert ppm > 0
        assert isinstance(ppm, float)

    @patch('src.segmentation.assessor.YOLO')
    def test_detect_plate_and_scale_no_plate(self, mock_yolo):
        """Test detect_plate_and_scale fallback when no plate detected"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # Create mock results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_model.return_value = [mock_result]

        assessor_instance = assessor.DietaryAssessor()
        test_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        ppm = assessor_instance.detect_plate_and_scale(test_image)

        # Should use fallback: image.shape[1] / 50.0
        expected_ppm = 1000 / 50.0
        assert ppm == expected_ppm

    @patch('src.segmentation.assessor.YOLO')
    def test_analyze_scene_with_detections(self, mock_yolo):
        """Test analyze_scene with detected objects"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # Create mock results with masks and boxes
        mock_result = Mock()
        
        # Mock mask
        mock_mask = Mock()
        mock_mask.cpu.return_value.numpy.return_value.astype.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_result.masks = Mock()
        mock_result.masks.data = [mock_mask]
        
        # Mock box
        mock_box = Mock()
        mock_box.xyxy = [np.array([10, 10, 50, 50])]
        mock_box.conf = 0.95
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]

        assessor_instance = assessor.DietaryAssessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        detected_objects, ppm = assessor_instance.analyze_scene(test_image)

        assert len(detected_objects) == 1
        assert "bbox" in detected_objects[0]
        assert "mask" in detected_objects[0]
        assert "area_pixels" in detected_objects[0]
        assert "area_cm2" in detected_objects[0]
        assert "confidence" in detected_objects[0]
        assert detected_objects[0]["confidence"] == 0.95
        assert ppm > 0

    @patch('src.segmentation.assessor.YOLO')
    def test_analyze_scene_no_detections(self, mock_yolo):
        """Test analyze_scene with no detected objects"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # Create mock results with no masks
        mock_result = Mock()
        mock_result.masks = None
        mock_model.return_value = [mock_result]

        assessor_instance = assessor.DietaryAssessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        detected_objects, ppm = assessor_instance.analyze_scene(test_image)

        assert len(detected_objects) == 0
        assert ppm > 0

    @patch('src.segmentation.assessor.YOLO')
    def test_analyze_scene_multiple_objects(self, mock_yolo):
        """Test analyze_scene with multiple detected objects"""
        # Mock YOLO model
        mock_model = Mock()
        mock_yolo.return_value = mock_model

        # Create mock results with multiple masks and boxes
        mock_result = Mock()
        
        # Mock masks
        mock_mask1 = Mock()
        mock_mask1.cpu.return_value.numpy.return_value.astype.return_value = np.ones((100, 100), dtype=np.uint8)
        mock_mask2 = Mock()
        mock_mask2.cpu.return_value.numpy.return_value.astype.return_value = np.ones((100, 100), dtype=np.uint8) * 0.5
        mock_result.masks = Mock()
        mock_result.masks.data = [mock_mask1, mock_mask2]
        
        # Mock boxes
        mock_box1 = Mock()
        mock_box1.xyxy = [np.array([10, 10, 50, 50])]
        mock_box1.conf = 0.95
        mock_box2 = Mock()
        mock_box2.xyxy = [np.array([60, 60, 90, 90])]
        mock_box2.conf = 0.85
        mock_result.boxes = [mock_box1, mock_box2]
        
        mock_model.return_value = [mock_result]

        assessor_instance = assessor.DietaryAssessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        detected_objects, ppm = assessor_instance.analyze_scene(test_image)

        assert len(detected_objects) == 2
        assert detected_objects[0]["confidence"] == 0.95
        assert detected_objects[1]["confidence"] == 0.85

