"""
Unit Tests for Utilities Module

Tests all utility modules:
- image_utils.py
- file_utils.py
- data_utils.py
- validation_utils.py
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import image_utils, file_utils, data_utils, validation_utils


class TestImageUtils:
    """Test image_utils.py module"""
    
    def test_process_crop_valid(self):
        """Test process_crop with valid inputs"""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        bbox = (100, 100, 500, 500)
        
        result = image_utils.process_crop(image, bbox)
        
        assert result is not None
        assert result.shape == (512, 512, 3)
    
    def test_process_crop_invalid_small(self):
        """Test process_crop with too small crop"""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        bbox = (100, 100, 120, 120)  # Only 20x20 pixels
        
        result = image_utils.process_crop(image, bbox)
        
        assert result is None
    
    def test_process_crop_clamping(self):
        """Test process_crop clamps coordinates correctly"""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        bbox = (-50, -50, 500, 500)  # Negative coordinates
        
        result = image_utils.process_crop(image, bbox)
        
        assert result is not None
        assert result.shape == (512, 512, 3)
    
    def test_preprocess_for_model(self):
        """Test preprocess_for_model"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = image_utils.preprocess_for_model(image)
        
        assert result.shape == (1, 512, 512, 3)
        assert result.dtype.name == 'float32'
        assert result.numpy().min() >= 0.0
        assert result.numpy().max() <= 1.0
    
    def test_resize_image(self):
        """Test resize_image"""
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        result = image_utils.resize_image(image, (512, 512))
        
        assert result.shape == (512, 512, 3)
    
    def test_normalize_image(self):
        """Test normalize_image"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        result = image_utils.normalize_image(image)
        
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestFileUtils:
    """Test file_utils.py module"""
    
    def test_ensure_directory(self):
        """Test ensure_directory creates directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_subdir"
            
            result = file_utils.ensure_directory(test_dir)
            
            assert result.exists()
            assert result.is_dir()
    
    def test_save_and_load_json(self):
        """Test save_json and load_json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_data = {"key": "value", "number": 42}
            
            file_utils.save_json(test_data, test_file)
            loaded_data = file_utils.load_json(test_file)
            
            assert loaded_data == test_data
    
    def test_save_json_log(self):
        """Test save_json_log creates timestamped file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = {"test": "data"}
            
            result_path = file_utils.save_json_log(test_data, Path(tmpdir), prefix="test")
            
            assert result_path.exists()
            assert result_path.name.startswith("test_")
            assert result_path.suffix == ".json"
    
    def test_list_files(self):
        """Test list_files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").touch()
            (Path(tmpdir) / "file2.json").touch()
            (Path(tmpdir) / "file3.txt").touch()
            
            all_files = file_utils.list_files(Path(tmpdir))
            txt_files = file_utils.list_files(Path(tmpdir), extension=".txt")
            
            assert len(all_files) == 3
            assert len(txt_files) == 2
    
    def test_file_exists(self):
        """Test file_exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "exists.txt"
            existing_file.touch()
            non_existing_file = Path(tmpdir) / "not_exists.txt"
            
            assert file_utils.file_exists(existing_file) is True
            assert file_utils.file_exists(non_existing_file) is False


class TestDataUtils:
    """Test data_utils.py module"""
    
    def test_safe_float(self):
        """Test safe_float conversion"""
        assert data_utils.safe_float("3.14") == 3.14
        assert data_utils.safe_float("invalid", default=0.0) == 0.0
        assert data_utils.safe_float(None, default=1.0) == 1.0

    def test_safe_int(self):
        """Test safe_int conversion"""
        assert data_utils.safe_int("42") == 42
        assert data_utils.safe_int("invalid", default=0) == 0
        assert data_utils.safe_int(3.7) == 3

    def test_clamp(self):
        """Test clamp function"""
        assert data_utils.clamp(5, 0, 10) == 5
        assert data_utils.clamp(-5, 0, 10) == 0
        assert data_utils.clamp(15, 0, 10) == 10

    def test_normalize_value(self):
        """Test normalize_value"""
        assert data_utils.normalize_value(5, 0, 10) == 0.5
        assert data_utils.normalize_value(0, 0, 10) == 0.0
        assert data_utils.normalize_value(10, 0, 10) == 1.0

    def test_round_to_decimals(self):
        """Test round_to_decimals"""
        assert data_utils.round_to_decimals(3.14159, 2) == 3.14
        assert data_utils.round_to_decimals(3.14159, 0) == 3.0

    def test_is_valid_number(self):
        """Test is_valid_number"""
        assert data_utils.is_valid_number(42) is True
        assert data_utils.is_valid_number(3.14) is True
        assert data_utils.is_valid_number("invalid") is False
        assert data_utils.is_valid_number(None) is False
        assert data_utils.is_valid_number(float('nan')) is False
        assert data_utils.is_valid_number(float('inf')) is False

    def test_clean_string_list(self):
        """Test clean_string_list"""
        items = ["apple", "banana", None, "apple", "nan", "", "cherry"]
        result = data_utils.clean_string_list(items)

        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result
        assert len(result) == 3  # Duplicates removed

    def test_merge_dicts(self):
        """Test merge_dicts"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        dict3 = {"d": 5}

        result = data_utils.merge_dicts(dict1, dict2, dict3)

        assert result == {"a": 1, "b": 3, "c": 4, "d": 5}


class TestValidationUtils:
    """Test validation_utils.py module"""

    def test_validate_image_valid(self):
        """Test validate_image with valid image"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        assert validation_utils.validate_image(image) is True

    def test_validate_image_invalid(self):
        """Test validate_image with invalid inputs"""
        assert validation_utils.validate_image(None) is False
        assert validation_utils.validate_image(np.array([])) is False
        assert validation_utils.validate_image(np.zeros((10, 10, 3))) is False  # Too small

    def test_validate_bbox_valid(self):
        """Test validate_bbox with valid bbox"""
        bbox = (100, 100, 500, 500)

        assert validation_utils.validate_bbox(bbox) is True

    def test_validate_bbox_invalid(self):
        """Test validate_bbox with invalid inputs"""
        assert validation_utils.validate_bbox(None) is False
        assert validation_utils.validate_bbox((100, 100, 50, 50)) is False  # x2 < x1
        assert validation_utils.validate_bbox((-10, 100, 500, 500)) is False  # Negative

    def test_validate_bbox_with_image_shape(self):
        """Test validate_bbox with image shape"""
        bbox = (100, 100, 500, 500)
        image_shape = (1000, 1000)

        assert validation_utils.validate_bbox(bbox, image_shape) is True

        # Out of bounds
        bbox_oob = (100, 100, 1500, 500)
        assert validation_utils.validate_bbox(bbox_oob, image_shape) is False

    def test_validate_crop(self):
        """Test validate_crop"""
        valid_crop = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        invalid_crop = np.zeros((10, 10, 3))

        assert validation_utils.validate_crop(valid_crop) is True
        assert validation_utils.validate_crop(invalid_crop) is False
        assert validation_utils.validate_crop(None) is False

    def test_validate_confidence(self):
        """Test validate_confidence"""
        assert validation_utils.validate_confidence(0.5) is True
        assert validation_utils.validate_confidence(0.0) is True
        assert validation_utils.validate_confidence(1.0) is True
        assert validation_utils.validate_confidence(-0.1) is False
        assert validation_utils.validate_confidence(1.5) is False

    def test_validate_mask(self):
        """Test validate_mask"""
        valid_mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        invalid_mask = np.zeros((512, 512, 3))  # Wrong dimensions

        assert validation_utils.validate_mask(valid_mask) is True
        assert validation_utils.validate_mask(invalid_mask) is False
        assert validation_utils.validate_mask(None) is False

    def test_validate_class_id(self):
        """Test validate_class_id"""
        valid_classes = ["ASC001", "ASC002", "ASC003"]

        assert validation_utils.validate_class_id("ASC001", valid_classes) is True
        assert validation_utils.validate_class_id("ASC999", valid_classes) is False
        assert validation_utils.validate_class_id(None, valid_classes) is False


