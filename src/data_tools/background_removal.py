"""
Background Removal Utility Module.

Configured to store U2-Net models in the project's 'models/' directory.
"""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2

# --- CONFIGURATION: FORCE MODELS TO LOCAL DIR ---
# We must set this env var BEFORE importing rembg
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Tell rembg to save/load u2net.onnx here instead of ~/.u2net
os.environ["U2NET_HOME"] = str(MODELS_DIR)

# Now it is safe to import
from rembg import remove, new_session

class BackgroundRemover:
    """
    Singleton-like handler for U2-Net Background Removal.
    """
    
    _instance = None
    _session = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackgroundRemover, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "u2net"):
        if self._session is None:
            print(f">> Initializing Background Removal Session ({model_name})...")
            print(f">> Model location: {MODELS_DIR}")
            
            # Robust Provider List: 
            # 1. CUDA (NVIDIA GPU)
            # 2. CoreML (MacBook Apple Silicon) - Good fallback if you switch machines
            # 3. CPU (Universal Fallback)
            providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
            self._session = new_session(model_name, providers=providers)

    def process_image(self, image_bgr: np.ndarray, bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Removes background using U2-Net (Fast Mode).
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty.")

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # FAST MODE: alpha_matting=False
        # This keeps your A6000 running fast (ms instead of seconds)
        result_rgba = remove(img_rgb, session=self._session, alpha_matting=False)

        img_rgb_foreground = result_rgba[:, :, :3]
        alpha_channel = result_rgba[:, :, 3]
        
        canvas = np.full_like(img_rgb_foreground, bg_color, dtype=np.uint8)
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        
        blended_rgb = (img_rgb_foreground * alpha_factor + 
                       canvas * (1.0 - alpha_factor)).astype(np.uint8)

        return cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)