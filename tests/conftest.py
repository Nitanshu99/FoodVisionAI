"""
Shared test fixtures for FoodVisionAI test suite
"""
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Fixture for sample test image (512x512x3)"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_results():
    """Fixture for sample inference results"""
    return [
        {
            "class_id": "apple",
            "confidence": 0.95,
            "visual_stats": {
                "bbox": [100, 100, 200, 200],
                "area_pixels": 10000,
                "mask": None
            }
        },
        {
            "class_id": "banana",
            "confidence": 0.87,
            "visual_stats": {
                "bbox": [250, 250, 350, 350],
                "area_pixels": 10000,
                "mask": None
            }
        }
    ]


@pytest.fixture
def temp_log_dir(tmp_path):
    """Fixture for temporary log directory"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def mock_model_path(tmp_path):
    """Fixture for temporary model path"""
    model_path = tmp_path / "test_model.keras"
    return model_path

