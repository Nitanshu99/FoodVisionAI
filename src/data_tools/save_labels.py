"""
Helper script to freeze the class labels.
Run this ONCE to ensure the App uses the exact same class order as the Model.
"""
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to python path to import config
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src import config

def main():
    train_dir = config.TRAIN_DIR
    save_path = config.PROCESSED_DIR / "class_names.npy"
    
    print(f"Scanning {train_dir}...")
    
    if not os.path.exists(train_dir):
        print("ERROR: Training directory not found!")
        return

    # Exact logic used by Keras during training (Sorted Alphanumerically)
    class_names = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    
    if not class_names:
        print("ERROR: No class folders found!")
        return

    print(f"Found {len(class_names)} classes.")
    print(f"First 3: {class_names[:3]}")
    
    # Save to disk
    np.save(save_path, class_names)
    print(f"✅ Success! Labels saved to: {save_path}")

if __name__ == "__main__":
    main()