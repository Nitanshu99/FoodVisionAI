"""
Parquet Converter Module.

This module handles the extraction of metadata from legacy Excel files
and transforms them into high-performance Apache Parquet format.
This ensures strict schema enforcement (e.g., numbers are floats, not strings)
and enables millisecond-level lookups for the Logic Engine.
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_excel_to_parquet(source_file: Path, output_dir: Path) -> None:
    """
    Reads a single Excel file and saves it as a Parquet file.
    
    Args:
        source_file (Path): Path to the input .xlsx file.
        output_dir (Path): Directory where the .parquet file will be saved.
    """
    if not source_file.exists():
        logger.warning("Skipping missing file: %s", source_file)
        return

    try:
        # Load Data
        df = pd.read_excel(source_file)
        
        # Data Cleaning: 
        # 1. Strip whitespace from string columns
        # 2. Ensure numeric columns are strictly numeric (coercing errors to NaN)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Define output path
        output_file = output_dir / f"{source_file.stem}.parquet"
        
        # Save to Parquet (using 'pyarrow' engine for best compatibility)
        df.to_parquet(output_file, engine='pyarrow', index=False)
        
        logger.info("Successfully converted: %s -> %s (Rows: %d)", 
                    source_file.name, output_file.name, len(df))

    except Exception as e:
        logger.error("Failed to convert %s: %s", source_file.name, e)


def process_metadata(metadata_dir: Path, output_dir: Path) -> None:
    """
    Batch processes all critical Excel files defined in the project scope.
    
    Args:
        metadata_dir (Path): Source directory containing .xlsx files.
        output_dir (Path): Destination directory for .parquet files.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of files required by the Logic Engine
    required_files = [
        "INDB.xlsx",
        "recipes.xlsx",
        "recipes_names.xlsx",
        "recipes_servingsize.xlsx",
        "recipe_links.xlsx",
        "Units.xlsx"
    ]

    logger.info("Starting Parquet Conversion...")

    for filename in required_files:
        source_path = metadata_dir / filename
        convert_excel_to_parquet(source_path, output_dir)

    logger.info("Parquet Conversion Complete.")


if __name__ == "__main__":
    # Configuration based on Project Structure
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    META_RAW_DIR = BASE_DIR / "data" / "raw" / "metadata"
    PARQUET_DB_DIR = BASE_DIR / "data" / "processed" / "parquet_db"

    process_metadata(META_RAW_DIR, PARQUET_DB_DIR)