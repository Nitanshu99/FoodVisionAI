# 🚀 Dynamic YOLO Processor - Universal Guide

## ✨ What's New: Auto-Configuration!

The processor now **automatically detects your hardware** and configures itself optimally!

### Supported Systems:

| System | Auto-Detected Settings | Expected Speed |
|--------|------------------------|----------------|
| **MacBook Air M4** | 4 workers, CoreML, thermal-aware | ~8-12 img/sec |
| **MacBook Pro M-series** | 6-8 workers, CoreML | ~15-20 img/sec |
| **AMD EPYC Server (32 cores)** | 24 workers, CPU/CUDA | **~60-120 img/sec** |
| **Linux Workstation** | Auto-scaled workers | ~20-40 img/sec |
| **Any other system** | Conservative defaults | ~10-15 img/sec |

---

## 🎯 Quick Start (Works Everywhere!)

### Step 1: Test Auto-Configuration
```bash
python auto_config.py
```

This will show:
- Detected hardware (CPU, RAM, GPU)
- Auto-generated configuration
- Performance estimate for your system

### Step 2: Run the Processor
```bash
# Same command works on Mac AND Server!
python yolo_processor_multiprocess.py --skip
```

**That's it!** The processor automatically:
- ✅ Detects if you're on Mac or Server
- ✅ Configures optimal worker count
- ✅ Selects best execution providers (CoreML/CUDA/CPU)
- ✅ Adjusts for thermal limits (MacBook Air)
- ✅ Estimates completion time

---

## 📊 What You'll See

### On MacBook Air M4:
```
============================================================
HARDWARE DETECTION
============================================================
System:          Darwin
Machine:         arm64
CPU Cores:       10
RAM:             16.0 GB
Apple Silicon:   True
NVIDIA GPU:      False
Cooling:         passive
System Type:     macbook_air_m_series
============================================================

============================================================
AUTO-GENERATED CONFIGURATION
============================================================
Profile:         MacBook Air M-series (Passive Cooling)
Workers:         4
YOLO GPU:        False
ONNX Providers:  CoreMLExecutionProvider, CPUExecutionProvider
Thermal Aware:   True
============================================================

============================================================
PERFORMANCE ESTIMATE
============================================================
Dataset:         108,624 images
Estimated Speed: 10.1 images/sec
Estimated Time:  179.4 minutes (2.99 hours)
============================================================
```

### On AMD EPYC Server (32 cores):
```
============================================================
HARDWARE DETECTION
============================================================
System:          Linux
Machine:         x86_64
CPU Cores:       32
RAM:             64.0 GB
Apple Silicon:   False
NVIDIA GPU:      True
Cooling:         active
System Type:     linux_server
============================================================

============================================================
AUTO-GENERATED CONFIGURATION
============================================================
Profile:         Linux Server (32 cores, GPU: True)
Workers:         24
YOLO GPU:        True
ONNX Providers:  CUDAExecutionProvider, CPUExecutionProvider
Thermal Aware:   False
============================================================

============================================================
PERFORMANCE ESTIMATE
============================================================
Dataset:         108,624 images
Estimated Speed: 122.4 images/sec
Estimated Time:  14.8 minutes (0.25 hours)
============================================================
```

---

## 🔧 How It Works

### Auto-Detection Logic:

1. **Detects CPU cores** → Calculates optimal workers
   - MacBook Air: 4 workers (thermal limits)
   - MacBook Pro: 6-8 workers (active cooling)
   - Server (32 cores): 24 workers (75% of cores)

2. **Detects RAM** → Ensures safe memory usage
   - Each worker needs ~700MB
   - Auto-scales if RAM is limited

3. **Detects GPU** → Selects execution providers
   - Apple Silicon → CoreML (Apple Neural Engine)
   - NVIDIA GPU → CUDA (GPU acceleration)
   - No GPU → CPU (still works!)

4. **Detects Cooling** → Adjusts for thermal limits
   - Passive (MacBook Air) → Conservative workers
   - Active (Server/Pro) → Aggressive workers

---

## 📈 Performance Comparison

### Your 108,624 Images:

| System | Workers | Speed | Total Time | vs Original |
|--------|---------|-------|------------|-------------|
| **Original (single-thread)** | 1 | 2.5 img/sec | ~12 hours | Baseline |
| **MacBook Air M4** | 4 | 10 img/sec | **~3 hours** | **4x faster** |
| **MacBook Pro M4** | 8 | 20 img/sec | **~1.5 hours** | **8x faster** |
| **AMD EPYC (CPU only)** | 24 | 60 img/sec | **~30 min** | **24x faster** |
| **AMD EPYC (with GPU)** | 24 | 120 img/sec | **~15 min** | **48x faster!** |

---

## 🎛️ Manual Override (Optional)

### Override Worker Count:
```bash
# Use specific number of workers
python yolo_processor_multiprocess.py --skip --workers 16
```

### Override for Maximum Speed (Server):
```bash
# Use all cores (not recommended, leave some for system)
python yolo_processor_multiprocess.py --skip --workers 30
```

### Override for Battery Saving (Mac):
```bash
# Use minimal workers
python yolo_processor_multiprocess.py --skip --workers 2
```

---

## 🖥️ Running on Server

### Transfer Files to Server:
```bash
# From your Mac
scp yolo_processor_multiprocess.py user@server:/path/to/project/
scp auto_config.py user@server:/path/to/project/
scp -r src/ user@server:/path/to/project/
scp -r data/ user@server:/path/to/project/
scp -r models/ user@server:/path/to/project/
```

### Install Dependencies on Server:
```bash
# SSH into server
ssh user@server
cd /path/to/project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base dependencies
pip install ultralytics opencv-python numpy scikit-learn tqdm rembg psutil

# If server has NVIDIA GPU (recommended!)
pip install onnxruntime-gpu torch torchvision
```

### Run on Server:
```bash
# Auto-detects server hardware and configures optimally
python yolo_processor_multiprocess.py --skip
```

---

## ✅ Verification

### Test Auto-Config First:
```bash
python auto_config.py
```

Check the output:
- ✅ Correct CPU count detected?
- ✅ Correct RAM detected?
- ✅ GPU detected (if you have one)?
- ✅ Reasonable worker count?
- ✅ Performance estimate looks good?

---

## 🔍 Troubleshooting

### "Import Error: No module named 'psutil'"
```bash
pip install psutil
```

### "Import Error: No module named 'auto_config'"
Make sure `auto_config.py` is in the same directory as the processor.

### Workers Too High (Out of Memory)
```bash
# Manually reduce workers
python yolo_processor_multiprocess.py --skip --workers 8
```

### Workers Too Low (Slow Processing)
```bash
# Manually increase workers
python yolo_processor_multiprocess.py --skip --workers 16
```

---

## 🎯 Best Practices

### On MacBook Air:
- ✅ Use auto-config (defaults to 4 workers)
- ✅ Plug in power adapter
- ✅ Close other apps
- ⚠️ Don't override to >6 workers (thermal throttling)

### On Server:
- ✅ Use auto-config (optimizes for your cores)
- ✅ Install GPU drivers if you have NVIDIA GPU
- ✅ Run in `screen` or `tmux` for long jobs
- ✅ Monitor with `htop` to verify CPU usage

---

## 📝 Files Summary

| File | Purpose |
|------|---------|
| `auto_config.py` | Hardware detection & auto-configuration |
| `yolo_processor_multiprocess.py` | Main processor (now with auto-config!) |
| `DYNAMIC_PROCESSOR_GUIDE.md` | This guide |

---

## 🚀 Ready to Go!

### On Mac:
```bash
python yolo_processor_multiprocess.py --skip
```

### On Server:
```bash
python yolo_processor_multiprocess.py --skip
```

**Same command, optimized for each system!** 🎉

---

## 💡 Pro Tips

1. **Always test auto-config first**: `python auto_config.py`
2. **Use `--skip` flag**: Resume from interruptions
3. **Monitor first few minutes**: Verify speed is as expected
4. **On server, use screen/tmux**: Don't lose progress if SSH disconnects
5. **Check error log**: `processing_errors.json` if any failures

---

**Happy Processing!** 🚀

