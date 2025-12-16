"""
Data Tools Package

Provides data processing and preprocessing utilities for FoodVisionAI.
All data tools are accessible via: from src.data_tools import *
"""

from src.data_tools.background_removal import BackgroundRemover
from src.data_tools.folder_mapper import get_manual_mapping, group_sources_by_target

__all__ = [
    "BackgroundRemover",
    "get_manual_mapping",
    "group_sources_by_target",
]
