"""
File I/O Utilities

Common file operations used across the codebase.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
import config


def get_class_names() -> List[str]:
    """
    Loads class names from the saved .npy file.
    
    Returns:
        List[str]: List of class names in the same order as model output.
    
    Raises:
        FileNotFoundError: If class_names.npy doesn't exist.
    """
    labels_path = config.LABELS_PATH
    
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Class names file not found at {labels_path}. "
            "Please run 'python src/data_tools/save_labels.py' first."
        )
    
    class_names = np.load(labels_path, allow_pickle=True).tolist()
    return class_names


def save_json_log(data: Dict[str, Any], log_dir: Path = None, prefix: str = "inference") -> Path:
    """
    Save data as JSON log file with timestamp.
    
    Args:
        data (Dict[str, Any]): Data to save
        log_dir (Path, optional): Directory to save log. Defaults to config.LOGS_DIR
        prefix (str, optional): Filename prefix. Defaults to "inference"
    
    Returns:
        Path: Path to saved log file
    """
    if log_dir is None:
        log_dir = config.LOGS_DIR
    
    # Ensure directory exists
    ensure_directory(log_dir)
    
    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = log_dir / filename
    
    # Save JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def ensure_directory(directory: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory (Path): Directory path
    
    Returns:
        Path: The directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        filepath (Path): Path to JSON file
    
    Returns:
        Dict[str, Any]: Loaded JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save data as JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (Path): Path to save file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_files(directory: Path, extension: str = None) -> List[Path]:
    """
    List all files in directory, optionally filtered by extension.
    
    Args:
        directory (Path): Directory to search
        extension (str, optional): File extension filter (e.g., '.json')
    
    Returns:
        List[Path]: List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if extension:
        if not extension.startswith('.'):
            extension = f'.{extension}'
        return sorted(directory.glob(f'*{extension}'))
    
    return sorted([f for f in directory.iterdir() if f.is_file()])


def read_text_file(filepath: Path) -> str:
    """
    Read text file content.
    
    Args:
        filepath (Path): Path to text file
    
    Returns:
        str: File content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(content: str, filepath: Path) -> None:
    """
    Write content to text file.
    
    Args:
        content (str): Content to write
        filepath (Path): Path to save file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def file_exists(filepath: Path) -> bool:
    """
    Check if file exists.
    
    Args:
        filepath (Path): Path to check
    
    Returns:
        bool: True if file exists
    """
    return Path(filepath).exists()

