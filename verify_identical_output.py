"""
Verification Script - Ensures Multiprocessing Output is Identical to Original

This script verifies that the multiprocessing version produces
pixel-perfect identical results to the original processor.

Usage:
    python verify_identical_output.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys

def compare_images(img1_path: Path, img2_path: Path) -> tuple:
    """
    Compare two images pixel by pixel.
    
    Returns:
        (identical: bool, difference: float)
    """
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        return False, -1
    
    if img1.shape != img2.shape:
        return False, -1
    
    # Pixel-perfect comparison
    difference = np.sum(np.abs(img1.astype(float) - img2.astype(float)))
    identical = difference == 0
    
    return identical, difference


def verify_outputs(original_dir: Path, new_dir: Path, sample_size: int = 100):
    """
    Verify that outputs from both processors are identical.
    """
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Comparing Processor Outputs")
    print(f"{'='*60}\n")
    
    print(f"Original dir: {original_dir}")
    print(f"New dir:      {new_dir}")
    print(f"Sample size:  {sample_size} images\n")
    
    if not original_dir.exists():
        print(f"ERROR: Original directory not found!")
        return False
    
    if not new_dir.exists():
        print(f"ERROR: New directory not found!")
        return False
    
    # Collect sample images
    original_images = []
    for split in ['train', 'val']:
        split_dir = original_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    images = list(class_dir.glob('*.jpg'))
                    original_images.extend(images[:10])  # 10 per class
                    if len(original_images) >= sample_size:
                        break
        if len(original_images) >= sample_size:
            break
    
    original_images = original_images[:sample_size]
    
    if not original_images:
        print("No images found in original directory!")
        return False
    
    print(f"Found {len(original_images)} images to compare\n")
    
    # Compare images
    identical_count = 0
    different_count = 0
    missing_count = 0
    
    for orig_path in original_images:
        # Find corresponding image in new directory
        relative_path = orig_path.relative_to(original_dir)
        new_path = new_dir / relative_path
        
        if not new_path.exists():
            missing_count += 1
            print(f"✗ Missing: {relative_path}")
            continue
        
        identical, diff = compare_images(orig_path, new_path)
        
        if identical:
            identical_count += 1
        else:
            different_count += 1
            print(f"✗ Different: {relative_path} (diff: {diff})")
    
    # Results
    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"✓ Identical:  {identical_count}/{len(original_images)}")
    print(f"✗ Different:  {different_count}/{len(original_images)}")
    print(f"⊘ Missing:    {missing_count}/{len(original_images)}")
    print(f"{'='*60}\n")
    
    if identical_count == len(original_images):
        print("✅ SUCCESS: All outputs are pixel-perfect identical!")
        return True
    else:
        print("⚠ WARNING: Some outputs differ!")
        return False


def verify_split_consistency(output_dir: Path):
    """
    Verify that train/val split is consistent (90/10).
    """
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Train/Val Split Ratio")
    print(f"{'='*60}\n")
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("ERROR: Train or val directory not found!")
        return False
    
    # Count images per class
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        train_count = len(list(class_dir.glob('*.jpg')))
        
        val_class_dir = val_dir / class_name
        val_count = len(list(val_class_dir.glob('*.jpg'))) if val_class_dir.exists() else 0
        
        total = train_count + val_count
        if total == 0:
            continue
        
        train_ratio = train_count / total
        
        # Check if ratio is approximately 90/10
        if abs(train_ratio - 0.9) > 0.05:  # Allow 5% tolerance
            print(f"⚠ {class_name}: {train_ratio*100:.1f}% train (expected ~90%)")
        else:
            print(f"✓ {class_name}: {train_ratio*100:.1f}% train, {(1-train_ratio)*100:.1f}% val")
    
    print(f"\n{'='*60}\n")
    return True


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    
    # You can modify these paths if needed
    ORIGINAL_DIR = BASE / "data" / "yolo_processed_original"  # Backup of original output
    NEW_DIR = BASE / "data" / "yolo_processed"
    
    print("\n" + "="*60)
    print("YOLO PROCESSOR OUTPUT VERIFICATION")
    print("="*60)
    print("\nThis script verifies that the multiprocessing version")
    print("produces identical output to the original processor.")
    print("\nNOTE: You need to have both outputs to compare:")
    print(f"  1. Original: {ORIGINAL_DIR}")
    print(f"  2. New:      {NEW_DIR}")
    print("\nIf you don't have the original output backed up,")
    print("this verification cannot be performed.")
    print("="*60)
    
    # Check if we have both directories
    if not ORIGINAL_DIR.exists():
        print(f"\n⚠ Original directory not found: {ORIGINAL_DIR}")
        print("\nTo use this verification:")
        print("1. Process a small sample with original processor")
        print("2. Backup output to 'yolo_processed_original'")
        print("3. Process same sample with multiprocessing version")
        print("4. Run this verification script")
        sys.exit(0)
    
    # Run verifications
    verify_outputs(ORIGINAL_DIR, NEW_DIR, sample_size=100)
    verify_split_consistency(NEW_DIR)

