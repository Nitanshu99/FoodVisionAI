"""
Unit Tests for Configuration Module

Tests all configuration modules:
- settings.py
- paths.py
- model_config.py
- hardware.py
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings, paths, model_config, hardware


class TestSettings:
    """Test settings.py module"""
    
    def test_image_settings_exist(self):
        """Test that image settings are defined"""
        assert settings.IMG_HEIGHT == 512
        assert settings.IMG_WIDTH == 512
        assert settings.IMG_SIZE == (512, 512)
        assert settings.INPUT_SHAPE == (512, 512, 3)
    
    def test_hyperparameters_exist(self):
        """Test that hyperparameters are defined"""
        assert settings.BATCH_SIZE > 0
        assert settings.EPOCHS > 0
        assert settings.LEARNING_RATE > 0
        assert settings.WEIGHT_DECAY > 0
        assert settings.SEED == 42
    
    def test_hyperparameters_valid_ranges(self):
        """Test that hyperparameters are in valid ranges"""
        assert 1 <= settings.BATCH_SIZE <= 256
        assert 1 <= settings.EPOCHS <= 1000
        assert 0 < settings.LEARNING_RATE < 1
        assert 0 < settings.WEIGHT_DECAY < 1
    
    def test_plate_diameter(self):
        """Test plate diameter constant"""
        assert settings.PLATE_DIAMETER_CM > 0
        assert isinstance(settings.PLATE_DIAMETER_CM, float)
    
    def test_device_detection(self):
        """Test device detection function"""
        device = settings.get_device()
        assert device in ['GPU', 'MPS', 'CPU']
        assert settings.DEVICE in ['GPU', 'MPS', 'CPU']


class TestPaths:
    """Test paths.py module"""
    
    def test_base_dir_exists(self):
        """Test that base directory exists"""
        assert paths.BASE_DIR.exists()
        assert paths.BASE_DIR.is_dir()
    
    def test_data_directories_defined(self):
        """Test that data directories are defined"""
        assert isinstance(paths.DATA_DIR, Path)
        assert isinstance(paths.PROCESSED_DIR, Path)
        assert isinstance(paths.TRAIN_DIR, Path)
        assert isinstance(paths.VAL_DIR, Path)
        assert isinstance(paths.PARQUET_DB_DIR, Path)
        assert isinstance(paths.LOGS_DIR, Path)
    
    def test_model_directories_defined(self):
        """Test that model directories are defined"""
        assert isinstance(paths.MODELS_DIR, Path)
        assert isinstance(paths.CHECKPOINT_DIR, Path)
    
    def test_model_paths_defined(self):
        """Test that model paths are defined"""
        assert isinstance(paths.MODEL_GLOBAL_PATH, Path)
        assert isinstance(paths.MODEL_LOCAL_PATH, Path)
        assert isinstance(paths.FINAL_MODEL_PATH, Path)
        assert isinstance(paths.YOLO_MODEL_PATH, Path)
        assert isinstance(paths.LLM_MODEL_PATH, Path)
    
    def test_critical_directories_created(self):
        """Test that critical directories are created"""
        assert paths.LOGS_DIR.exists()
        assert paths.CHECKPOINT_DIR.exists()
    
    def test_labels_path_defined(self):
        """Test that labels path is defined"""
        assert isinstance(paths.LABELS_PATH, Path)
    
    def test_db_files_registry(self):
        """Test database files registry"""
        assert isinstance(paths.DB_FILES, dict)
        assert len(paths.DB_FILES) > 0
        assert "nutrition" in paths.DB_FILES
        assert "recipes" in paths.DB_FILES


class TestModelConfig:
    """Test model_config.py module"""
    
    def test_model_config_exists(self):
        """Test that MODEL_CONFIG dictionary exists"""
        assert isinstance(model_config.MODEL_CONFIG, dict)
        assert len(model_config.MODEL_CONFIG) > 0
    
    def test_model_config_keys(self):
        """Test that MODEL_CONFIG has required keys"""
        required_keys = [
            'input_shape', 'batch_size', 'epochs', 
            'learning_rate', 'weight_decay', 'seed'
        ]
        for key in required_keys:
            assert key in model_config.MODEL_CONFIG
    
    def test_yolo_config_exists(self):
        """Test that YOLO_CONFIG exists"""
        assert isinstance(model_config.YOLO_CONFIG, dict)
        assert 'confidence_threshold' in model_config.YOLO_CONFIG
        assert 'image_size' in model_config.YOLO_CONFIG
    
    def test_llm_config_exists(self):
        """Test that LLM_CONFIG exists"""
        assert isinstance(model_config.LLM_CONFIG, dict)
        assert 'temperature' in model_config.LLM_CONFIG
        assert 'max_tokens' in model_config.LLM_CONFIG


class TestHardware:
    """Test hardware.py module"""
    
    def test_hardware_detector_init(self):
        """Test HardwareDetector initialization"""
        detector = hardware.HardwareDetector()
        assert detector is not None
        assert detector.cpu_count > 0
        assert detector.ram_gb > 0
    
    def test_hardware_detector_attributes(self):
        """Test HardwareDetector attributes"""
        detector = hardware.HardwareDetector()
        assert hasattr(detector, 'system')
        assert hasattr(detector, 'machine')
        assert hasattr(detector, 'cpu_count')
        assert hasattr(detector, 'ram_gb')
        assert hasattr(detector, 'is_apple_silicon')
        assert hasattr(detector, 'has_nvidia_gpu')

    def test_get_system_type(self):
        """Test system type detection"""
        detector = hardware.HardwareDetector()
        system_type = detector.get_system_type()
        assert isinstance(system_type, str)
        assert len(system_type) > 0

    def test_auto_config_init(self):
        """Test AutoConfig initialization"""
        auto_config = hardware.AutoConfig()
        assert auto_config is not None
        assert auto_config.config is not None
        assert isinstance(auto_config.config, dict)

    def test_auto_config_keys(self):
        """Test AutoConfig has required keys"""
        auto_config = hardware.AutoConfig()
        required_keys = [
            'NUM_WORKERS', 'CONF_THRESHOLD', 'IMG_SIZE',
            'ONNX_PROVIDERS', 'USE_GPU_FOR_YOLO', 'DESCRIPTION'
        ]
        for key in required_keys:
            assert key in auto_config.config

    def test_get_auto_config_function(self):
        """Test get_auto_config function"""
        config = hardware.get_auto_config()
        assert isinstance(config, dict)
        assert 'NUM_WORKERS' in config
        assert config['NUM_WORKERS'] > 0


class TestConfigIntegration:
    """Integration tests for config module"""

    def test_config_module_imports(self):
        """Test that config module can be imported"""
        import config
        assert config is not None

    def test_all_settings_accessible_from_config(self):
        """Test that all settings are accessible from main config module"""
        import config

        # From settings
        assert hasattr(config, 'IMG_SIZE')
        assert hasattr(config, 'BATCH_SIZE')
        assert hasattr(config, 'LEARNING_RATE')

        # From paths
        assert hasattr(config, 'BASE_DIR')
        assert hasattr(config, 'DATA_DIR')
        assert hasattr(config, 'MODELS_DIR')

        # From model_config
        assert hasattr(config, 'MODEL_CONFIG')

        # From hardware
        assert hasattr(config, 'get_auto_config')

    def test_backward_compatibility(self):
        """Test backward compatibility with old config usage"""
        import config

        # These should all work as before
        assert config.IMG_SIZE == (512, 512)
        assert config.INPUT_SHAPE == (512, 512, 3)
        assert config.BASE_DIR.exists()
        assert isinstance(config.MODEL_GLOBAL_PATH, Path)
        assert isinstance(config.LABELS_PATH, Path)

