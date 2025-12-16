"""
Hardware Detection and Auto-Configuration for FoodVisionAI

Dynamically detects hardware and configures optimal settings for:
- MacBook Air M4 (16GB RAM, 10 cores, no fan)
- AMD EPYC Server (32+ cores, 64GB+ RAM, active cooling)
- NVIDIA A6000 GPU (48GB VRAM)
- Any other system

Automatically optimizes:
- Worker count based on CPU cores
- Memory settings based on available RAM
- Execution providers (CoreML for Apple, CUDA for NVIDIA, CPU fallback)
- Thermal management (passive vs active cooling)
- Training batch size, precision, and DataLoader settings

This module was moved from auto_config.py to config/hardware.py
"""

import platform
import multiprocessing as mp
import psutil
import os


class HardwareDetector:
    """Detect and analyze system hardware."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.cpu_count = mp.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.is_apple_silicon = self._detect_apple_silicon()
        self.has_nvidia_gpu = self._detect_nvidia_gpu()
        self.cooling_type = self._detect_cooling()
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon (M1/M2/M3/M4)."""
        return self.system == "Darwin" and self.machine == "arm64"
    
    def _detect_nvidia_gpu(self) -> bool:
        """Detect if NVIDIA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        # Check for nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_gpu_memory(self) -> int:
        """Get total GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except:
            pass
        return 0

    def get_gpu_name(self) -> str:
        """Get GPU name."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).name
        except:
            pass
        return "Unknown"
    
    def _detect_cooling(self) -> str:
        """Detect cooling type (passive vs active)."""
        if self.system == "Darwin":
            # MacBook Air has passive cooling (no fan)
            model = platform.mac_ver()[0]
            if "Air" in platform.node() or "air" in platform.node().lower():
                return "passive"
            return "active"  # MacBook Pro has fans
        return "active"  # Assume servers have active cooling
    
    def get_system_type(self) -> str:
        """Classify system type."""
        if self.is_apple_silicon:
            if self.cooling_type == "passive":
                return "macbook_air_m_series"
            return "macbook_pro_m_series"
        elif self.system == "Linux" and self.cpu_count >= 16:
            return "linux_server"
        elif self.system == "Linux":
            return "linux_workstation"
        elif self.system == "Windows":
            return "windows_workstation"
        return "unknown"
    
    def print_info(self):
        """Print detected hardware information."""
        print(f"\n{'='*60}")
        print(f"HARDWARE DETECTION")
        print(f"{'='*60}")
        print(f"System:          {self.system}")
        print(f"Machine:         {self.machine}")
        print(f"CPU Cores:       {self.cpu_count}")
        print(f"RAM:             {self.ram_gb:.1f} GB")
        print(f"Apple Silicon:   {self.is_apple_silicon}")
        print(f"NVIDIA GPU:      {self.has_nvidia_gpu}")
        print(f"Cooling:         {self.cooling_type}")
        print(f"System Type:     {self.get_system_type()}")
        print(f"{'='*60}\n")


class AutoConfig:
    """Automatically configure optimal settings based on hardware."""
    
    def __init__(self):
        self.hw = HardwareDetector()
        self.config = self._generate_config()
    
    def _generate_config(self) -> dict:
        """Generate optimal configuration based on detected hardware."""
        system_type = self.hw.get_system_type()
        
        if system_type == "macbook_air_m_series":
            return self._config_macbook_air()
        elif system_type == "macbook_pro_m_series":
            return self._config_macbook_pro()
        elif system_type == "linux_server":
            return self._config_linux_server()
        elif system_type == "linux_workstation":
            return self._config_linux_workstation()
        else:
            return self._config_generic()
    
    def _config_macbook_air(self) -> dict:
        """Configuration for MacBook Air (passive cooling, thermal limits)."""
        return {
            'NUM_WORKERS': min(4, self.hw.cpu_count - 2),  # Conservative for thermal
            'CONF_THRESHOLD': 0.25,
            'IMG_SIZE': (512, 512),
            'ONNX_PROVIDERS': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
            'USE_GPU_FOR_YOLO': False,
            'DESCRIPTION': 'MacBook Air M-series (Passive Cooling)',
            'THERMAL_AWARE': True,
        }
    
    def _config_macbook_pro(self) -> dict:
        """Configuration for MacBook Pro (active cooling)."""
        return {
            'NUM_WORKERS': max(6, self.hw.cpu_count - 2),  # More aggressive
            'CONF_THRESHOLD': 0.25,
            'IMG_SIZE': (512, 512),
            'ONNX_PROVIDERS': ['CoreMLExecutionProvider', 'CPUExecutionProvider'],
            'USE_GPU_FOR_YOLO': False,
            'DESCRIPTION': 'MacBook Pro M-series (Active Cooling)',
            'THERMAL_AWARE': False,
        }
    
    def _config_linux_server(self) -> dict:
        """Configuration for Linux server (many cores, active cooling)."""
        # Use 75% of cores, leave 25% for system
        optimal_workers = int(self.hw.cpu_count * 0.75)

        # Determine execution providers
        if self.hw.has_nvidia_gpu:
            # TensorRT is fastest, then CUDA, then CPU
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            use_gpu_yolo = True
        else:
            providers = ['CPUExecutionProvider']
            use_gpu_yolo = False

        # Training configuration
        gpu_mem = self.hw.get_gpu_memory()
        gpu_name = self.hw.get_gpu_name()

        # Optimize for A6000 (48GB) with 35GB limit
        if gpu_mem >= 40:  # A6000 or similar
            batch_size = 96
            dataloader_workers = 12
        elif gpu_mem >= 24:  # RTX 3090/4090
            batch_size = 64
            dataloader_workers = 10
        elif gpu_mem >= 12:  # RTX 3060/4060
            batch_size = 32
            dataloader_workers = 8
        else:
            batch_size = 16
            dataloader_workers = 6

        return {
            # Processing config
            'NUM_WORKERS': optimal_workers,
            'CONF_THRESHOLD': 0.25,
            'IMG_SIZE': (512, 512),
            'ONNX_PROVIDERS': providers,
            'USE_GPU_FOR_YOLO': use_gpu_yolo,
            'DESCRIPTION': f'Linux Server ({self.hw.cpu_count} cores, GPU: {self.hw.has_nvidia_gpu})',
            'THERMAL_AWARE': False,

            # Training config
            'TRAIN_BATCH_SIZE': batch_size,
            'TRAIN_WORKERS': dataloader_workers,
            'TRAIN_PRECISION': 'fp16' if self.hw.has_nvidia_gpu else 'fp32',
            'TRAIN_PIN_MEMORY': True,
            'TRAIN_PERSISTENT_WORKERS': True,
            'TRAIN_PREFETCH_FACTOR': 2,
            'GPU_NAME': gpu_name,
            'GPU_MEMORY_GB': gpu_mem,
        }

    def _config_linux_workstation(self) -> dict:
        """Configuration for Linux workstation."""
        optimal_workers = max(4, self.hw.cpu_count - 2)

        if self.hw.has_nvidia_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            use_gpu_yolo = True
        else:
            providers = ['CPUExecutionProvider']
            use_gpu_yolo = False

        return {
            'NUM_WORKERS': optimal_workers,
            'CONF_THRESHOLD': 0.25,
            'IMG_SIZE': (512, 512),
            'ONNX_PROVIDERS': providers,
            'USE_GPU_FOR_YOLO': use_gpu_yolo,
            'DESCRIPTION': f'Linux Workstation ({self.hw.cpu_count} cores)',
            'THERMAL_AWARE': False,
        }

    def _config_generic(self) -> dict:
        """Generic fallback configuration."""
        optimal_workers = max(2, self.hw.cpu_count // 2)

        return {
            'NUM_WORKERS': optimal_workers,
            'CONF_THRESHOLD': 0.25,
            'IMG_SIZE': (512, 512),
            'ONNX_PROVIDERS': ['CPUExecutionProvider'],
            'USE_GPU_FOR_YOLO': False,
            'DESCRIPTION': 'Generic Configuration',
            'THERMAL_AWARE': False,
        }

    def print_config(self):
        """Print the generated configuration."""
        print(f"\n{'='*60}")
        print(f"AUTO-GENERATED CONFIGURATION")
        print(f"{'='*60}")
        print(f"Profile:         {self.config['DESCRIPTION']}")
        print(f"Workers:         {self.config['NUM_WORKERS']}")
        print(f"YOLO GPU:        {self.config['USE_GPU_FOR_YOLO']}")
        print(f"ONNX Providers:  {', '.join(self.config['ONNX_PROVIDERS'])}")
        print(f"Thermal Aware:   {self.config['THERMAL_AWARE']}")

        # Print training config if available
        if 'TRAIN_BATCH_SIZE' in self.config:
            print(f"\n--- TRAINING OPTIMIZATION ---")
            print(f"GPU:             {self.config.get('GPU_NAME', 'N/A')}")
            print(f"GPU Memory:      {self.config.get('GPU_MEMORY_GB', 0)} GB")
            print(f"Batch Size:      {self.config['TRAIN_BATCH_SIZE']}")
            print(f"DataLoader Workers: {self.config['TRAIN_WORKERS']}")
            print(f"Precision:       {self.config['TRAIN_PRECISION']}")
            print(f"Pin Memory:      {self.config['TRAIN_PIN_MEMORY']}")
            print(f"Persistent Workers: {self.config['TRAIN_PERSISTENT_WORKERS']}")

        print(f"{'='*60}\n")

    def estimate_performance(self, num_images: int = 100000):
        """Estimate processing time based on configuration."""
        # Baseline speeds (images/sec per worker)
        if self.hw.has_nvidia_gpu:
            base_speed = 8.0  # With GPU acceleration
        elif self.hw.is_apple_silicon:
            base_speed = 4.0  # With CoreML
        else:
            base_speed = 2.5  # CPU only

        # Multiprocessing efficiency (not linear)
        workers = self.config['NUM_WORKERS']
        if workers <= 4:
            efficiency = 0.9
        elif workers <= 8:
            efficiency = 0.85
        elif workers <= 16:
            efficiency = 0.80
        else:
            efficiency = 0.75

        estimated_speed = base_speed * workers * efficiency

        # Thermal throttling for passive cooling
        if self.config.get('THERMAL_AWARE', False):
            estimated_speed *= 0.7  # 30% reduction for thermal limits

        total_time_sec = num_images / estimated_speed
        total_time_min = total_time_sec / 60
        total_time_hr = total_time_min / 60

        print(f"\n{'='*60}")
        print(f"PERFORMANCE ESTIMATE")
        print(f"{'='*60}")
        print(f"Dataset:         {num_images:,} images")
        print(f"Estimated Speed: {estimated_speed:.1f} images/sec")
        print(f"Estimated Time:  {total_time_min:.1f} minutes ({total_time_hr:.2f} hours)")
        print(f"{'='*60}\n")

        return estimated_speed, total_time_sec


def get_auto_config():
    """Get auto-configured settings for current hardware."""
    auto = AutoConfig()
    auto.hw.print_info()
    auto.print_config()
    return auto.config


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO PROCESSOR - AUTO CONFIGURATION")
    print("="*60)

    auto = AutoConfig()
    auto.hw.print_info()
    auto.print_config()
    auto.estimate_performance(num_images=100000)

    print("\nConfiguration ready to use!")
    print("Import with: from auto_config import get_auto_config")


