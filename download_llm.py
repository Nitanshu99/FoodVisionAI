"""
Download Qwen2.5-0.5B GGUF model for offline chat functionality.
"""

import os
from pathlib import Path
from urllib.request import urlretrieve

def download_qwen_model():
    """Download Qwen2.5-0.5B GGUF model to models/ directory."""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "qwen2.5-0.5b-instruct-fp16.gguf"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return
    
    print("📥 Downloading Qwen2.5-0.5B model...")
    url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf"
    
    try:
        urlretrieve(url, model_path)
        print(f"✅ Download complete: {model_path}")
        print(f"📊 Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Manual download: Visit https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")

if __name__ == "__main__":
    download_qwen_model()