import numpy as np
import os
from src import config

# Check current class_names.npy
labels_path = config.LABELS_PATH
print(f"Loading class names from: {labels_path}")

if os.path.exists(labels_path):
    class_names = np.load(labels_path)
    print(f"Found {len(class_names)} classes in class_names.npy")
    print(f"First 5: {class_names[:5]}")
    print(f"Last 5: {class_names[-5:]}")
    
    # Check if index 22 exists
    if len(class_names) > 22:
        print(f"Class at index 22: {class_names[22]}")
    else:
        print(f"ERROR: Index 22 doesn't exist! Only {len(class_names)} classes found.")
else:
    print("ERROR: class_names.npy not found!")

# Check what's in your training directory
train_dir = config.TRAIN_DIR
if os.path.exists(train_dir):
    all_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"\nTraining directory has {len(all_dirs)} folders")
    print(f"First 5: {all_dirs[:5]}")
    
    # Check if excluded classes are still there
    excluded = ["ASC321", "OSR146"]
    for exc in excluded:
        if exc in all_dirs:
            print(f"WARNING: {exc} still exists in training directory!")