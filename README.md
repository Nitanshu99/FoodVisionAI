# FoodVisionAI 🥘

### High-Fidelity Monocular Dietary Assessment System

**FoodVisionAI** is a research-grade, offline computer vision system designed to estimate the nutritional content of food from single images. It moves beyond simple classification by integrating **Geometric Deep Learning** (Volume Estimation) with a **Dual-Model "Split-Brain" Architecture** (Context vs. Texture Analysis) to handle complex dietary scenarios like Indian Thalis.

---

## 🚀 Key Features

* **🧠 "Split-Brain" Inference Engine:** Automatically switches strategies based on scene complexity:
  * **Global Context Model:** Used for single dishes to capture plate context and scale.
  * **Local Specialist Model:** Used for multi-item meals (Thalis) to classify specific food textures from tight crops.
* **💬 Conversational UI:** A ChatGPT-style interface that allows users to upload meals, receive visual/nutritional reports, and (future) ask follow-up questions.
* **📊 Physics-Based Nutrient Engine:** Calculates mass using volumetric primitives (Cylinders, Spherical Caps) and density lookups, rather than simple regression.
* **🔗 Rich Metadata Integration:** Fetches Ingredients, Standard Serving Sizes, and Recipe Source URLs from a local Parquet database.
* **📝 RAG-Ready Logging:** Automatically saves structured JSON logs (`food_item_1`, `food_item_2`...) for future LLM integration.

---

## 🏗️ System Architecture

The system operates on a **Hybrid Pipeline**:

### 1. Vision Layer (The Eyes)

* **YOLOv8-Seg:** Detects food items, generates segmentation masks, and estimates Real-World Scale (Pixels-Per-Metric) using a reference plate diameter (28cm).
* **Decision Logic:**
  * If N ≤ 1 Objects: **Global Path** (Uses Full Image).
  * If N > 1 Objects: **Local Path** (Uses Cropped Objects).

### 2. Classification Layer (The Brain)

* **Global Model (`model_best.keras`):** EfficientNet-B5 trained on standardized, full-context images.
* **Local Model (`model_yolo_best.keras`):** EfficientNet-B5 trained on "Clean" YOLO-generated crops (Background removed).

### 3. Nutrient Layer (The Logic)

* Combines **Volume (cm³)** × **Density (g/cm³)** to get Mass.
* Queries **INDB (Indian Food Database)** for macros.
* Enriches data with Recipe Links and Ingredients.

---

## 📂 Directory Structure

```
FoodVisionAI/
├── config/                   # ✨ NEW: Modular configuration
│   ├── settings.py           # App settings (image size, hyperparameters)
│   ├── paths.py              # File paths (models, data, databases)
│   ├── model_config.py       # Model configurations (EfficientNet, YOLO, LLM)
│   └── hardware.py           # Hardware detection and auto-configuration
│
├── data/
│   ├── raw/                  # Original raw image datasets
│   ├── processed/            # Global Model Data (Standardized & Split)
│   ├── yolo_processed/       # Local Model Data (YOLO Cropped & Split)
│   ├── parquet_db/           # Database Engine (INDB, Recipes, Units, Links)
│   └── inference_logs/       # JSON logs stored here for RAG
│
├── models/
│   ├── checkpoints/
│   │   ├── model_best.keras       # Global Model (Context-Aware)
│   │   └── model_yolo_best.keras  # Local Model (Crop-Specialist)
│   ├── yolov8m-seg.pt             # Segmentation Model
│   └── qwen2.5-0.5b-instruct-fp16.gguf  # LLM for chat features
│
├── src/
│   ├── chat/                 # ✨ NEW: Chat/LLM modules
│   │   ├── engine.py         # ChatEngine (LangGraph state machine)
│   │   ├── llm.py            # QwenLLM wrapper (Qwen2.5-0.5B)
│   │   └── rag.py            # SimpleRAG (session log search)
│   │
│   ├── data_tools/           # Data processing modules
│   │   ├── background_removal.py  # U2Net background removal
│   │   ├── folder_mapper.py       # Preprocessor A: Maps/Prunes raw data
│   │   ├── parquet_converter.py   # Excel → Parquet converter
│   │   ├── inspect_headers.py     # Database diagnostic tool
│   │   └── save_labels.py         # Class labels freezer
│   │
│   ├── models/               # ✨ NEW: Model building modules
│   │   ├── builder.py        # Model building (build_model)
│   │   ├── augmentation.py   # Data augmentation (RandomGaussianBlur)
│   │   └── loader.py         # Model loading/saving utilities
│   │
│   ├── nutrition/            # ✨ NEW: Nutrition calculation modules
│   │   └── engine.py         # NutrientEngine (Physics & Database Logic)
│   │
│   ├── segmentation/         # ✨ NEW: Segmentation modules
│   │   └── assessor.py       # DietaryAssessor (YOLOv8 Geometry Engine)
│   │
│   ├── utils/                # ✨ NEW: Utility modules
│   │   ├── image_utils.py    # Image processing utilities
│   │   ├── file_utils.py     # File I/O utilities
│   │   ├── data_utils.py     # Data manipulation utilities
│   │   └── validation_utils.py  # Validation utilities
│   │
│   └── vision/               # ✨ NEW: Vision inference modules
│       └── inference.py      # "Split-Brain" Inference Logic
│
├── tests/                    # ✨ NEW: Comprehensive test suite (124 tests)
│   ├── test_chat.py          # Chat module tests (19 tests)
│   ├── test_config.py        # Config module tests (25 tests)
│   ├── test_data_tools.py    # Data tools tests (13 tests)
│   ├── test_models.py        # Model module tests (17 tests)
│   ├── test_nutrition.py     # Nutrition module tests (12 tests)
│   ├── test_segmentation.py  # Segmentation tests (6 tests)
│   ├── test_utils.py         # Utility tests (28 tests)
│   └── test_vision.py        # Vision module tests (4 tests)
│
├── app.py                    # Streamlit Frontend (Chat Interface)
├── train.py                  # Training script (Global Model)
├── after_yolo_train.py       # Training script (Local Model)
├── ARCHITECTURE.md           # ✨ NEW: Detailed architecture documentation
├── MIGRATION_GUIDE.md        # ✨ NEW: Migration guide for new structure
└── requirements.txt          # Project Dependencies
```

> **✨ What's New?** The codebase has been refactored into a modular package structure with 124 comprehensive tests. See `ARCHITECTURE.md` for details and `MIGRATION_GUIDE.md` for import changes.

---

## 🔄 Data Workflow

The system employs two distinct data preprocessing pipelines to create specialized datasets for the "Split-Brain" architecture.

### 1. The Database Pipeline

Raw Excel/CSV files are converted into high-speed **Parquet** files for sub-millisecond querying:

* `INDB.parquet`: Macronutrients (Energy, Protein, Carbs, Fat).
* `recipes.parquet`: Ingredient lists.
* `recipe_links.parquet`: Source URLs.
* `Units.parquet`: Density estimation (Cup/Bowl weights).

### 2. The Training Pipeline (Dual-Stream)

#### Stream A: The Global Model Pipeline (Context-Aware)

* **Input:** `data/raw/images`
* **Preprocessing:** Run `src/data_tools/folder_mapper.py`.
  * **Normalization:** Standardizes folder names (e.g., "chicken pizza" → `BFP122`).
  * **Pruning:** Removes irrelevant or unmapped folders.
  * **Splitting:** Generates Train/Val splits in `data/processed`.
* **Training:**
  ```bash
  python train.py
  ```

#### Stream B: The Local Model Pipeline (Texture-Specialist)

* **Input:** `data/raw/images`
* **Preprocessing:** Run `src/data_tools/yolo_processor.py`.
  * **Detection:** Uses YOLOv8 to find the specific food object.
  * **Cropping:** Removes background/plate context to isolate texture.
  * **Splitting:** Generates Train/Val splits in `data/yolo_processed`.
* **Training:**
  ```bash
  python after_yolo_train.py
  ```

---

## 👤 User Guide

### Installation

```bash
# Clone and Enter
git clone https://github.com/yourusername/FoodVisionAI.git
cd FoodVisionAI

# Install Dependencies
pip install -r requirements.txt
```

### Running the App

Launch the Chat Interface:

```bash
streamlit run app.py
```

### Using the System

1. **Upload:** Drop a meal photo in the sidebar.
2. **Analysis:**
   * **Single Dish:** The AI detects the context and identifies the dish (e.g., *Uttapam*) using the **Global Model**.
   * **Thali:** The AI detects multiple items (e.g., *Roti, Paneer, Dal*), crops them, and identifies each individually using the **Local Model**.
3. **Result:**
   * View the **annotated image** (Green boxes).
   * Read the **Nutritional Summary** card.
   * Click **"Recipe Link"** to see how to make it.
4. **Logging:**
   * Check `data/inference_logs/` to see the generated JSON file with detailed breakdown (`food_item_1`, `food_item_2`...) ready for LLM usage.

---

## 🧩 JSON Log Format (RAG-Ready)

Every prediction is saved automatically in this structure:

```json
{
    "timestamp": "2023-10-27T10:00:00",
    "total_summary": {
        "Energy (kcal)": 650,
        "Protein (g)": 25
    },
    "food_item_1": {
        "name": "Paneer Butter Masala",
        "mass_g": 150.0,
        "macros": { "calories": 450, "protein": 18 },
        "metadata": {
            "ingredients": ["Paneer", "Tomato", "Cashew"],
            "source": "https://hebbarskitchen.com/...",
            "model_used": "Object Crop (Local Specialist)"
        }
    },
    "food_item_2": {
        "name": "Tandoori Roti",
        "mass_g": 60.0,
        ...
    }
}
```

---

## 🔬 Technical Specifications

### Model Architecture

Both Global and Local models share the same core architecture, optimized for the NVIDIA RTX A6000:

* **Backbone:** EfficientNet-B5 (Pretrained on ImageNet)
* **Input Resolution:** 512 × 512 pixels
  * **Why:** High resolution is required to detect fine-grained textures (e.g., oil separation, spice patterns)
* **Activation Function:** Swish (SiLU)
  * **Why:** Unlike ReLU (used in older ResNets), Swish is a smooth, non-monotonic function (f(x) = x · sigmoid(x)) that allows a small amount of negative information to flow through. This is mathematically proven to help deep networks like B5 learn complex patterns (like food textures) faster.
* **Training Hardware:** NVIDIA RTX A6000 (48GB VRAM)
* **Model Parameters:** ~30M parameters per model

#### Hyperparameters

* **Batch Size:** 64 (Optimal balance of speed and Batch Norm stability)
* **Optimizer:** AdamW (Weight Decay: 1e-4)
* **Learning Rate:** 1e-3 with Cosine Decay Schedule
* **Precision:** FP32 (Standard 32-bit Floating Point)

### Data Augmentation Strategy (The "Regularization Shield")

**Theory:** EfficientNet-B5 is a high-capacity model (30M+ parameters). Given the "small local-food dataset," the risk of overfitting (memorizing noise instead of features) is mathematically high. We introduce a stochastic "Variance Injection" pipeline to enforce Rotational, Scale, and Photometric Invariance, ensuring the model generalizes to unseen real-world photos.

#### Pipeline Specifications (Applied Pre-Batch)

**Geometric Transformations (Spatial Invariance):**

* **Random Rotation:** ±20°
  * **Justification:** User photos will rarely be perfectly aligned.
* **Random Zoom:** Range [0.8, 1.2] (20% zoom in/out)
  * **Justification:** Addresses the "Distance Ambiguity" problem where the food's scale varies relative to the camera frame.
* **Random Flip:** Horizontal
  * **Justification:** Food presentation is chirality-invariant (a samosa facing left is still a samosa).

**Photometric Transformations (Sensor Invariance):**

* **Random Contrast & Brightness:** Factor 0.2
  * **Justification:** Simulates varying lighting conditions (e.g., dim restaurant vs. bright kitchen).

**Blur Transformations (Focus Invariance):**

* **Gaussian Blur:** Kernel Size 3×3 or 5×5, applied with 30% probability
  * **Why:** This simulates Depth of Field and Motion Blur (common in hasty smartphone photography). It forces the model to learn "shape" and "global structure" rather than relying solely on sharp, high-frequency textures that might disappear in a blurry photo.

### Split-Brain Architecture

The dual-model approach solves a fundamental challenge in food recognition:

* **Global Model** excels at understanding spatial relationships and portion context
* **Local Model** specializes in identifying foods when isolated from their surroundings

This architecture automatically adapts based on scene complexity, providing optimal accuracy for both simple plates and complex multi-dish meals.

### Physics-Based Volume Estimation

Unlike traditional regression approaches, FoodVisionAI uses geometric primitives to calculate actual food volume:

* Cylinders for flat/round foods (rotis, dosas)
* Spherical caps for mounded foods (rice, curries)
* Real-world scale calibration using standard plate dimensions (28cm reference)

### RAG Integration Ready

The structured JSON logging format enables seamless integration with Large Language Models for:

* Conversational dietary advice
* Meal planning recommendations
* Nutritional queries and explanations

### Technical Justifications

**Why Parquet?**

"We converted legacy Excel datasets to Apache Parquet. This provides schema enforcement (preventing data type errors) and allows our Logic Engine to query nutritional priors in milliseconds, ensuring a seamless user experience."

**Why Swish Activation?**

"We utilized the Swish activation function (standard in EfficientNet) because its smooth, non-linear property prevents 'dying neurons' during the training of deep networks, allowing our B5 model to capture subtle food texture gradients better than ReLU."

**Why Batch Size 64?**

"Leveraging the 48GB VRAM of the A6000, we maximized the batch size to 64. This ensures stable statistical estimation in the Batch Normalization layers, which is critical for generalizing across the high intra-class variance of Indian food."

---

## 📊 Performance Characteristics

* **Inference Speed:** ~2-3 seconds per image (CPU), ~0.5 seconds (GPU)
* **Model Size:** 
  * Global Model: ~115MB
  * Local Model: ~115MB
  * YOLO Segmentation: ~50MB
* **Database Query:** Sub-millisecond (Parquet optimization)
* **Supported Foods:** Indian cuisine focus with extensible architecture

---

## 🛠️ Development Workflow

### Training the Global Model

```bash
# 1. Prepare raw data
# Place images in data/raw/images/

# 2. Preprocess and map to food codes
python src/data_tools/folder_mapper.py

# 3. Train the global model
python train.py
```

### Training the Local Model

```bash
# 1. Generate YOLO crops
python src/data_tools/yolo_processor.py

# 2. Train the local specialist
python after_yolo_train.py
```

### Database Setup

All database files should be placed in `data/raw/metadata/`:
* `INDB.xlsx`
* `recipes.xlsx`
* `recipes_names.xlsx`
* `recipes_servingsize.xlsx`
* `recipe_links.xlsx`
* `Units.xlsx`

These will be automatically converted to Parquet format during preprocessing.

---

## 📦 Package Structure & Imports

The codebase follows a modular package structure for better organization and maintainability.

### Quick Import Examples

```python
# Configuration
from config import settings, paths, model_config

# Vision inference
from src.vision import predict_food

# Models
from src.models import build_model, load_model, save_model

# Nutrition
from src.nutrition import NutrientEngine

# Segmentation
from src.segmentation import DietaryAssessor

# Chat/LLM
from src.chat import ChatEngine, get_llm

# Utilities
from src.utils import preprocess_for_model, process_crop, validate_image, get_class_names
```

For detailed documentation:
- **Architecture:** See `ARCHITECTURE.md` for complete package structure
- **Migration:** See `MIGRATION_GUIDE.md` for import changes from old structure

### Testing

Run the comprehensive test suite (124 tests):

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=config --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

* Expanding food database coverage
* Improving volume estimation algorithms
* Adding support for additional cuisines
* Enhancing the conversational UI
* Adding more comprehensive tests

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

* Indian Food Database (INDB) for nutritional data
* YOLOv8 for state-of-the-art object detection
* EfficientNet for efficient image classification
* Hebbar's Kitchen and other recipe sources

---

## 📧 Contact

[Your Contact Information]