"""
Unit Tests for Data Tools Module

Tests all data_tools modules:
- background_removal.py
- folder_mapper.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_tools import background_removal, folder_mapper


class TestBackgroundRemoval:
    """Test background_removal.py module"""

    def test_background_remover_singleton(self):
        """Test BackgroundRemover is a singleton"""
        remover1 = background_removal.BackgroundRemover()
        remover2 = background_removal.BackgroundRemover()

        assert remover1 is remover2

    def test_background_remover_initialization(self):
        """Test BackgroundRemover initializes correctly"""
        remover = background_removal.BackgroundRemover()

        assert remover is not None
        assert remover._session is not None

    def test_process_image_empty_input(self):
        """Test process_image raises error for empty input"""
        remover = background_removal.BackgroundRemover()

        with pytest.raises(ValueError, match="Input image is empty"):
            remover.process_image(np.array([]))

    def test_process_image_none_input(self):
        """Test process_image raises error for None input"""
        remover = background_removal.BackgroundRemover()

        with pytest.raises(ValueError, match="Input image is empty"):
            remover.process_image(None)

    @patch('src.data_tools.background_removal.remove')
    def test_process_image_valid_input(self, mock_remove):
        """Test process_image with valid input"""
        # Mock the remove function to return a fake RGBA image
        fake_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mock_remove.return_value = fake_rgba

        remover = background_removal.BackgroundRemover()
        input_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = remover.process_image(input_image)

        assert result is not None
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    @patch('src.data_tools.background_removal.remove')
    def test_process_image_custom_bg_color(self, mock_remove):
        """Test process_image with custom background color"""
        fake_rgba = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mock_remove.return_value = fake_rgba

        remover = background_removal.BackgroundRemover()
        input_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = remover.process_image(input_image, bg_color=(255, 255, 255))

        assert result is not None
        assert result.shape == (100, 100, 3)


class TestFolderMapper:
    """Test folder_mapper.py module"""

    def test_get_manual_mapping_returns_dict(self):
        """Test get_manual_mapping returns a dictionary"""
        mapping = folder_mapper.get_manual_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_get_manual_mapping_has_expected_entries(self):
        """Test get_manual_mapping has expected food mappings"""
        mapping = folder_mapper.get_manual_mapping()

        # Check some known mappings
        assert "aloo gobi" in mapping
        assert mapping["aloo gobi"] == "ASC171"
        assert "biryani" in mapping
        assert mapping["biryani"] == "ASC123"

    def test_get_manual_mapping_all_lowercase_keys(self):
        """Test all keys in mapping are lowercase"""
        mapping = folder_mapper.get_manual_mapping()

        for key in mapping.keys():
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_group_sources_by_target(self):
        """Test group_sources_by_target inverts mapping correctly"""
        mapping = {"chicken pizza": "BFP122", "pepperoni pizza": "BFP122", "biryani": "ASC123"}

        grouped = folder_mapper.group_sources_by_target(mapping)

        assert "BFP122" in grouped
        assert len(grouped["BFP122"]) == 2
        assert "chicken pizza" in grouped["BFP122"]
        assert "pepperoni pizza" in grouped["BFP122"]
        assert "ASC123" in grouped
        assert len(grouped["ASC123"]) == 1

    def test_group_sources_by_target_empty_mapping(self):
        """Test group_sources_by_target with empty mapping"""
        grouped = folder_mapper.group_sources_by_target({})

        assert grouped == {}

    def test_group_sources_by_target_single_entry(self):
        """Test group_sources_by_target with single entry"""
        mapping = {"idli": "ASC144"}

        grouped = folder_mapper.group_sources_by_target(mapping)

        assert len(grouped) == 1
        assert "ASC144" in grouped
        assert grouped["ASC144"] == ["idli"]

    def test_mapping_consistency(self):
        """Test that all target codes are valid format"""
        mapping = folder_mapper.get_manual_mapping()

        for source, target in mapping.items():
            # Target codes should be 6 characters (3 letters + 3 digits)
            assert len(target) == 6, f"Invalid target code: {target}"
            assert target[:3].isalpha(), f"Target code should start with letters: {target}"
            assert target[3:].isdigit(), f"Target code should end with digits: {target}"

