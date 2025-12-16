"""
Data Manipulation Utilities

Common data processing and manipulation functions.
"""

from typing import Any, List, Union
import pandas as pd


def safe_get(df_loc: Any, col: str, default: Any = None) -> Any:
    """
    Safely get value from DataFrame location or Series.
    Handles both Series and scalar values.
    
    Args:
        df_loc: DataFrame location (e.g., df.loc[index])
        col (str): Column name
        default (Any, optional): Default value if extraction fails
    
    Returns:
        Any: Extracted value or default
    """
    try:
        val = df_loc[col]
        if isinstance(val, pd.Series):
            return val.iloc[0]
        return val
    except (KeyError, IndexError, AttributeError):
        return default


def clean_string_list(items: List[Any]) -> List[str]:
    """
    Clean list of items by removing NaN, None, and duplicates.
    Converts all items to strings and strips whitespace.
    
    Args:
        items (List[Any]): List of items to clean
    
    Returns:
        List[str]: Cleaned list of unique strings
    """
    clean_items = []
    for item in items:
        if item is None:
            continue
        
        item_str = str(item).strip()
        
        # Skip NaN values
        if item_str.lower() in ['nan', 'none', '']:
            continue
        
        clean_items.append(item_str)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(clean_items))


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value (Any): Value to convert
        default (float, optional): Default value if conversion fails
    
    Returns:
        float: Converted value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.
    
    Args:
        value (Any): Value to convert
        default (int, optional): Default value if conversion fails
    
    Returns:
        int: Converted value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value (float): Value to clamp
        min_val (float): Minimum value
        max_val (float): Maximum value
    
    Returns:
        float: Clamped value
    """
    return max(min_val, min(max_val, value))


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to [0, 1] range.
    
    Args:
        value (float): Value to normalize
        min_val (float): Minimum value in range
        max_val (float): Maximum value in range
    
    Returns:
        float: Normalized value
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def round_to_decimals(value: float, decimals: int = 1) -> float:
    """
    Round value to specified number of decimals.
    
    Args:
        value (float): Value to round
        decimals (int, optional): Number of decimal places. Defaults to 1.
    
    Returns:
        float: Rounded value
    """
    return round(value, decimals)


def is_valid_number(value: Any) -> bool:
    """
    Check if value is a valid number (not NaN, not None, not infinite).

    Args:
        value (Any): Value to check

    Returns:
        bool: True if valid number
    """
    try:
        num = float(value)
        import math
        return not (pd.isna(num) or math.isinf(num))
    except (ValueError, TypeError):
        return False


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries into one.
    Later dictionaries override earlier ones.
    
    Args:
        *dicts: Variable number of dictionaries
    
    Returns:
        dict: Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

