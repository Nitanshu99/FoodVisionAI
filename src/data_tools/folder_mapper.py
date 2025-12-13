"""
Folder Mapper & Pruner Module.

This module implements a strict data curation pipeline:
1. Prunes raw directories not matching the approved list.
2. Merges multiple source folders into single target categories (ASC/BFP codes).
3. Normalizes file extensions.
4. Splits merged data into Train/Val sets.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set

from sklearn.model_selection import train_test_split

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_manual_mapping() -> Dict[str, str]:
    """
    Returns the specific Source-to-Target mapping provided by the user.
    Keys are raw folder names (lower case), Values are Target Codes.
    """
    return {
        "aloo gobi": "ASC171",
        "aloo methi": "ASC175",
        "aloo mutter": "ASC190",
        "aloo paratha": "ASC098",
        "anda curry": "BFP240",
        "banana chips": "OSR119",
        "besan laddu": "ASC343",
        "bhindi masala": "BFP269",
        "biryani": "ASC123",
        "boondi laddu": "ASC343",
        "chaas": "ASC022",
        "chana masala": "ASC162",
        "chapati": "ASC096",
        "chicken pizza": "BFP122",
        "chicken wings": "ASC245",
        "chikki": "ASC382",
        "chivda": "ASC052",
        "chole bhature": "ASC143",
        "dal khichdi": "BFP144",
        "dhokla": "ASC474",
        "falooda": "ASC321",
        "fish curry": "ASC246",
        "gajar ka halwa": "ASC295",
        "garlic naan": "ASC142",
        "grilled sandwich": "ASC031",
        "gujhia": "ASC347",
        "gulab jamun": "ASC348",
        "handvo": "OSR146",
        "hara bhara kabab": "ASC377",
        "idli": "ASC144",
        "kaju katli": "ASC339",
        "khakhra": "OSR102",
        "kheer": "ASC282",
        "kulfi": "ASC321",
        "margherita pizza": "ASC376",
        "masala dosa": "ASC146",
        "medu vada": "BFP436",
        "moong dal halwa": "ASC299",
        "murukku": "OSR113",
        "navratan korma": "ASC225",
        "neer dosa": "BFP148",
        "onion pakoda": "ASC352",
        "palak paneer": "ASC215",
        "paneer masala": "ASC195",
        "paneer pizza": "ASC376",
        "pani puri": "OSR114",
        "papdi chaat": "OSR148",
        "pav bhaji": "OSR112",
        "pepperoni pizza": "BFP122",
        "phirni": "ASC292",
        "poha": "BFP044",
        "pongal": "BFP144",
        "puri bhaji": "ASC107",
        "rajma chawal": "ASC165",
        "rasgulla": "BFP392",
        "rava dosa": "ASC147",
        "sabudana khichdi": "OSR099",
        "sabudana vada": "BFP425",
        "samosa": "ASC361",
        "seekh kebab": "OSR153",
        "set dosa": "BFP148",
        "sev puri": "OSR114",
        "thukpa": "ASC081",
        "uttapam": "ASC148"
    }


def prune_raw_data(raw_dir: Path, valid_folders: Set[str]) -> None:
    """
    Deletes any folder in raw_dir that is NOT in the valid_folders set.
    """
    if not raw_dir.exists():
        logger.error("Raw directory %s does not exist.", raw_dir)
        return

    logger.info("Starting Pruning Phase...")
    
    # Iterate over actual directories on disk
    for folder in raw_dir.iterdir():
        if folder.is_dir():
            folder_name = folder.name.lower() # normalize to lowercase for comparison
            
            if folder_name not in valid_folders:
                logger.warning("PRUNING: Deleting unlisted folder '%s'", folder.name)
                try:
                    shutil.rmtree(folder)
                except OSError as e:
                    logger.error("Error deleting %s: %s", folder.name, e)
    
    logger.info("Pruning complete.")


def group_sources_by_target(mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Inverts the mapping to handle merging.
    Returns: { 'BFP122': ['chicken pizza', 'pepperoni pizza'], ... }
    """
    grouped = {}
    for src, target in mapping.items():
        if target not in grouped:
            grouped[target] = []
        grouped[target].append(src)
    return grouped


def normalize_image_extension(file_path: Path) -> str:
    """Ensures file has .jpg extension."""
    if not file_path.suffix:
        return f"{file_path.name}.jpg"
    return file_path.name


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.9
) -> None:
    """
    Main execution flow: Prune -> Group -> Split -> Copy.
    """
    # 1. Get Config
    mapping = get_manual_mapping()
    valid_source_names = set(mapping.keys())

    # 2. Prune Unwanted Folders
    prune_raw_data(raw_dir, valid_source_names)

    # 3. Group for Merging
    target_groups = group_sources_by_target(mapping)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    # 4. Process Groups
    for target_code, source_folders in target_groups.items():
        
        # Collect ALL files from ALL source folders for this target
        all_files = []
        found_sources = []
        
        for src_name in source_folders:
            src_path = raw_dir / src_name
            # Check if exists (case insensitive search might be needed on Windows, 
            # but assuming standard lowercase names here)
            if not src_path.exists():
                # Try finding it if casing matches exactly just in case
                # (Simple fallback not implemented for brevity, relies on lowercase match)
                logger.debug("Source folder %s not found on disk (skipping part of merge).", src_name)
                continue
            
            found_sources.append(src_name)
            files = [f for f in src_path.iterdir() if f.is_file() and not f.name.startswith('.')]
            all_files.extend(files)

        if not all_files:
            logger.warning("Target %s has no images from sources %s. Skipping.", target_code, source_folders)
            continue

        # Split
        if len(all_files) < 2:
            train_files = all_files
            val_files = []
        else:
            train_files, val_files = train_test_split(
                all_files, train_size=split_ratio, random_state=42, shuffle=True
            )

        # Copy Files function (local helper)
        def _copy_batch(files: List[Path], dest_root: Path):
            target_path = dest_root / target_code
            target_path.mkdir(parents=True, exist_ok=True)
            for f in files:
                new_name = normalize_image_extension(f)
                try:
                    shutil.copy2(f, target_path / new_name)
                except OSError as e:
                    logger.error("Failed to copy %s: %s", f.name, e)

        _copy_batch(train_files, train_dir)
        _copy_batch(val_files, val_dir)

        logger.info(
            "Processed Target %s: Merged %s -> (Train: %d, Val: %d)", 
            target_code, found_sources, len(train_files), len(val_files)
        )

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_IMAGES = BASE_DIR / "data" / "raw" / "images"
    PROCESSED = BASE_DIR / "data" / "processed"

    process_dataset(RAW_IMAGES, PROCESSED)