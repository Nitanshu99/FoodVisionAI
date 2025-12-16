# FoodVisionAI 🥘

### High-Fidelity Monocular Dietary Assessment System

**FoodVisionAI** is a research-grade, offline computer vision system designed to estimate the nutritional content of food from single images. It moves beyond simple classification by integrating **Geometric Deep Learning** (Volume Estimation) with a **Unified Deep Learning Architecture** trained on clean, background-removed crops to handle complex dietary scenarios like Indian Thalis.

---

## 🚀 Key Features

* **🧠 Unified Inference Engine:** Uses a single EfficientNet-B5 model trained on clean, background-removed crops for optimal texture recognition
* **🎯 Adaptive Segmentation:** YOLOv8-seg detects up to 8 food items per image with intelligent filtering
* **🧹 Smart Background Removal:** U2Net removes backgrounds per-crop for cleaner classification
* **💬 Conversational UI:** ChatGPT-style interface with meal upload, visual/nutritional reports, and LLM-powered trivia
* **📊 Physics-Based Nutrient Engine:** Calculates mass using 3 strategies:
  * **Container Strategy:** For liquids (bowls, cups) using volume × density
  * **Geometric Strategy:** For solids (rotis, pieces) using cylinder approximation
  * **Mound Strategy:** For piled foods (rice, curries) using spherical cap geometry
* **🔗 Rich Metadata Integration:** Fetches ingredients, serving sizes, and recipe URLs from 6 Parquet databases
* **📝 RAG-Ready Logging:** Structured JSON logs (`food_item_1`, `food_item_2`...) for LLM integration

---

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "User Interface"
        A[Streamlit App] --> B[Image Upload]
    end

    subgraph "Vision Pipeline"
        B --> C[YOLOv8-Seg<br/>Segmentation]
        C --> D[Adaptive Filtering<br/>Top 8 Segments]
        D --> E[Per-Crop Processing]
        E --> F[U2Net Background<br/>Removal]
        F --> G[EfficientNet-B5<br/>Classification]
    end

    subgraph "Nutrition Pipeline"
        G --> H[NutrientEngine]
        H --> I{Strategy Selection}
        I -->|Container| J[Volume × Density]
        I -->|Geometric| K[Cylinder Formula]
        I -->|Mound| L[Spherical Cap]
        J --> M[Database Lookup]
        K --> M
        L --> M
    end

    subgraph "Data Layer"
        M --> N[(6 Parquet DBs)]
        N --> O[INDB - Nutrition]
        N --> P[Recipes - Ingredients]
        N --> Q[Links - URLs]
        N --> R[Serving Sizes]
        N --> S[Units - Density]
        N --> T[Names - Display]
    end

    subgraph "Output"
        M --> U[Structured JSON Log]
        U --> V[Chat Interface]
        V --> W[LLM Trivia Generation]
    end

    style C fill:#e1f5ff
    style G fill:#ffe1f5
    style H fill:#fff5e1
    style N fill:#e1ffe1
```

### Component Architecture

The system operates on a **Modular Pipeline**:

#### 1. Vision Layer (The Eyes)

* **YOLOv8-Seg:** Detects food items and generates segmentation masks
* **Scale Recovery:** Estimates Pixels-Per-Metric (PPM) using reference plate diameter (28cm)
* **Adaptive Filtering:** Keeps top 8 largest segments to focus on main food items
* **U2Net Background Removal:** Removes backgrounds per-crop for cleaner classification

#### 2. Classification Layer (The Brain)

* **Unified Model (`model_yolo_best.keras`):** EfficientNet-B5 trained on clean, background-removed crops
* **Input Resolution:** 512×512 pixels for fine-grained texture detection
* **Architecture:** 30M parameters with Swish activation for smooth gradients

#### 3. Nutrition Layer (The Logic)

* **3 Calculation Strategies:**
  * **Container:** Volume (ml) × Density (g/ml) for liquids
  * **Geometric:** Cylinder formula for flat/round foods
  * **Mound:** Spherical cap for piled foods
* **6 Database Integration:**
  * INDB.parquet - Macronutrients (Energy, Protein, Carbs, Fat)
  * recipes.parquet - Ingredient lists
  * recipe_links.parquet - Source URLs
  * recipes_servingsize.parquet - Standard serving sizes
  * Units.parquet - Density estimation
  * recipes_names.parquet - Display names

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

## 🔄 System Workflows

### Inference Workflow (User Perspective)

```mermaid
sequenceDiagram
    participant U as User
    participant App as Streamlit App
    participant YOLO as YOLOv8-Seg
    participant U2Net as U2Net BG Remover
    participant Model as EfficientNet-B5
    participant Engine as NutrientEngine
    participant DB as Parquet Databases
    participant LLM as Qwen LLM

    U->>App: Upload meal image
    App->>YOLO: Detect & segment food items
    YOLO-->>App: Return 8 largest segments + PPM

    loop For each segment
        App->>U2Net: Remove background from crop
        U2Net-->>App: Clean crop
        App->>Model: Classify clean crop
        Model-->>App: Food class + confidence
        App->>Engine: Calculate nutrition
        Engine->>DB: Query macros, ingredients, links
        DB-->>Engine: Return data
        Engine-->>App: Nutrition profile
    end

    App->>App: Generate structured JSON log
    App->>LLM: Generate food trivia
    LLM-->>App: Fun facts
    App-->>U: Display annotated image + nutrition + trivia
```

### Training Workflow

```mermaid
graph LR
    subgraph "Data Preparation"
        A[Raw Images] --> B[folder_mapper.py<br/>Normalize & Prune]
        B --> C[yolo_processor.py<br/>Detect & Crop]
        C --> D[U2Net Background<br/>Removal]
        D --> E[Train/Val Split<br/>data/yolo_processed/]
    end

    subgraph "Model Training"
        E --> F[train.py]
        F --> G[EfficientNet-B5<br/>+ Augmentation]
        G --> H[AdamW Optimizer<br/>Cosine Decay LR]
        H --> I[Batch Size: 96<br/>A6000 GPU]
        I --> J[Best Model<br/>model_yolo_best.keras]
    end

    subgraph "Validation"
        J --> K[Validation Set<br/>Accuracy Check]
        K --> L[Save to<br/>models/checkpoints/]
    end

    style E fill:#e1f5ff
    style J fill:#ffe1e1
    style L fill:#e1ffe1
```

### Nutrition Calculation Workflow

```mermaid
graph TB
    A[Visual Stats from YOLO] --> B{Check Serving Unit}

    B -->|Bowl/Cup/Glass| C[Container Strategy]
    C --> C1[Volume = 200ml × Fill Ratio]
    C1 --> C2[Mass = Volume × Density]

    B -->|Piece/Slice/Number| D[Geometric Strategy]
    D --> D1[Estimate Cylinder<br/>Radius & Height]
    D1 --> D2[Volume = π × r² × h]
    D2 --> D3[Mass = Volume × Density]

    B -->|Mound/Heap| E[Mound Strategy]
    E --> E1[Estimate Spherical Cap<br/>Radius & Height]
    E1 --> E2[Volume = π × h² × (3r - h) / 3]
    E2 --> E3[Mass = Volume × Density]

    C2 --> F[Query INDB Database]
    D3 --> F
    E3 --> F

    F --> G[Macros per 100g]
    G --> H[Scale to Actual Mass]
    H --> I[Enrich with Metadata]
    I --> J[Return Complete Profile]

    style C fill:#e1f5ff
    style D fill:#ffe1f5
    style E fill:#fff5e1
    style F fill:#e1ffe1
```

### Data Pipeline Workflow

The system employs a comprehensive data preprocessing pipeline:

#### 1. Database Pipeline

Raw Excel/CSV files → High-speed **Parquet** files for sub-millisecond querying:

* `INDB.parquet`: Macronutrients (Energy, Protein, Carbs, Fat)
* `recipes.parquet`: Ingredient lists
* `recipe_links.parquet`: Source URLs
* `recipes_servingsize.parquet`: Standard serving sizes
* `Units.parquet`: Density estimation (Cup/Bowl weights)
* `recipes_names.parquet`: Display names

#### 2. Training Pipeline

* **Input:** `data/raw/images`
* **Step 1:** Run `src/data_tools/folder_mapper.py`
  * Normalizes folder names (e.g., "chicken pizza" → `BFP122`)
  * Prunes irrelevant or unmapped folders
  * Generates initial Train/Val splits
* **Step 2:** Run `src/data_tools/yolo_processor.py`
  * Uses YOLOv8 to detect and crop food objects
  * Removes background/plate context to isolate texture
  * Applies U2Net background removal
  * Generates Train/Val splits in `data/yolo_processed/`
* **Step 3:** Run `train.py`
  * Trains EfficientNet-B5 on clean crops
  * Saves best model to `models/checkpoints/model_yolo_best.keras`

---

## 👤 Complete User Guide

### Prerequisites

* **Python:** 3.10 or higher
* **Git:** For cloning the repository
* **Hardware:**
  * Minimum: 8GB RAM, CPU
  * Recommended: 16GB+ RAM, NVIDIA GPU (for faster inference)
  * Optimal: 48GB VRAM GPU (NVIDIA A6000) for training

### Installation (From Scratch)

#### Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/Nitanshu99/FoodVisionAI.git

# Navigate to project directory
cd FoodVisionAI
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
* `tensorflow>=2.16.0` - Deep learning framework
* `keras>=3.0.0` - High-level neural networks API
* `ultralytics` - YOLOv8 segmentation
* `streamlit` - Web UI framework
* `llama-cpp-python` - LLM inference (Qwen2.5)
* `opencv-python` - Image processing
* `pandas` - Data manipulation
* `pyarrow` - Parquet file support

#### Step 4: Download Required Models

```bash
# Download LLM model (Qwen2.5-0.5B GGUF format)
python download_llm.py
```

This will download:
* `qwen2.5-0.5b-instruct-fp16.gguf` (~500MB) to `models/`

**Note:** YOLOv8 model (`yolov8m-seg.pt`) will be auto-downloaded on first run.

#### Step 5: Set Up Data Directories

```bash
# Create required directories
mkdir -p data/raw/images
mkdir -p data/raw/metadata
mkdir -p data/processed
mkdir -p data/yolo_processed
mkdir -p data/parquet_db
mkdir -p data/inference_logs
mkdir -p models/checkpoints
```

#### Step 6: Prepare Database Files

Place your database files in `data/raw/metadata/`:
* `INDB.xlsx` - Nutrition database
* `recipes.xlsx` - Ingredients
* `recipes_names.xlsx` - Display names
* `recipes_servingsize.xlsx` - Serving sizes
* `recipe_links.xlsx` - Recipe URLs
* `Units.xlsx` - Density data

Then convert to Parquet format:

```bash
python src/data_tools/parquet_converter.py
```

#### Step 7: Download Pre-trained Models (Optional)

If you have pre-trained models, place them in `models/checkpoints/`:
* `model_yolo_best.keras` - Main classification model

If not, you'll need to train the model (see Training section below).

---

### Running the Application

#### Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

#### Using the System

**1. Upload a Meal Image**
* Click "Browse files" in the sidebar
* Select a food image (JPG, PNG, JPEG)
* Supported: Single dishes, multi-item meals (Thalis), complex plates

**2. Automatic Analysis**

The system will:
* ✅ Detect food items using YOLOv8 segmentation
* ✅ Remove backgrounds using U2Net
* ✅ Classify each item using EfficientNet-B5
* ✅ Calculate nutrition using physics-based formulas
* ✅ Fetch ingredients and recipe links from database
* ✅ Generate fun food trivia using LLM

**3. View Results**

* **Annotated Image:** Green boxes around detected food items
* **Nutrition Cards:** Per-item breakdown with:
  * Food name
  * Mass (grams)
  * Calories, Protein, Carbs, Fat
  * Ingredients list
  * Recipe link (clickable)
* **Total Summary:** Aggregated nutrition for entire meal
* **Fun Trivia:** LLM-generated interesting facts about your meal

**4. Chat with Your Meal (Future)**

* Ask questions like "What did I eat today?"
* Get dietary recommendations
* Track nutrition over time

**5. Access Logs**

All analyses are saved as structured JSON in `data/inference_logs/`:

```json
{
  "timestamp": "2024-12-16T10:30:00",
  "total_summary": {
    "Energy (kcal)": 650,
    "Protein (g)": 25
  },
  "food_item_1": {
    "name": "Paneer Butter Masala",
    "mass_g": 150.0,
    "macros": {...},
    "metadata": {...}
  }
}
```

---

## 📊 How Nutrition Calculation Works

### Overview

FoodVisionAI uses a **physics-based approach** to calculate nutrition, not simple regression. The system combines:

1. **Visual geometry** from YOLO segmentation (area, bounding box)
2. **Real-world scale** from plate detection (Pixels-Per-Metric)
3. **Food-specific strategies** based on serving unit type
4. **Database lookups** for macronutrients and density

### The 3 Calculation Strategies

#### Strategy 1: Container (For Liquids)

**Used for:** Bowls, cups, glasses, katoris (liquid/semi-liquid foods)

**Formula:**
```
Volume (ml) = Container_Volume × Fill_Ratio
Mass (g) = Volume × Density
```

**Example: Dal in a Bowl**
```python
Container_Volume = 200 ml  # Standard bowl
Fill_Ratio = 0.8  # 80% full (from YOLO occupancy)
Density = 1.0 g/ml  # Liquid density

Volume = 200 × 0.8 = 160 ml
Mass = 160 × 1.0 = 160 g
```

**Fill Ratio Calculation:**
* Detected from YOLO segmentation mask
* Ratio of food pixels to container pixels
* Adjusted: >80% → 100%, <40% → 50%

#### Strategy 2: Geometric (For Flat/Round Foods)

**Used for:** Rotis, dosas, parathas, pieces, slices

**Formula (Cylinder Approximation):**
```
Radius (cm) = √(Area_cm² / π)
Height (cm) = Estimated from food type
Volume (cm³) = π × r² × h
Mass (g) = Volume × Density
```

**Example: Roti**
```python
Area_cm2 = 150 cm²  # From YOLO + PPM
Radius = √(150 / π) = 6.9 cm
Height = 0.3 cm  # Typical roti thickness
Density = 0.6 g/cm³  # Bread density

Volume = π × 6.9² × 0.3 = 44.8 cm³
Mass = 44.8 × 0.6 = 26.9 g
```

**Height Estimation:**
* Looked up from serving size database
* Typical values: Roti (0.3cm), Dosa (0.2cm), Paratha (0.5cm)

#### Strategy 3: Mound (For Piled Foods)

**Used for:** Rice, curries, sabzi (mounded/heaped foods)

**Formula (Spherical Cap):**
```
Radius (cm) = √(Area_cm² / π)
Height (cm) = Radius × 0.4  # Typical mound ratio
Volume (cm³) = π × h² × (3r - h) / 3
Mass (g) = Volume × Density
```

**Example: Rice Mound**
```python
Area_cm2 = 200 cm²  # From YOLO + PPM
Radius = √(200 / π) = 7.98 cm
Height = 7.98 × 0.4 = 3.19 cm
Density = 0.85 g/cm³  # Cooked rice density

Volume = π × 3.19² × (3×7.98 - 3.19) / 3 = 167.5 cm³
Mass = 167.5 × 0.85 = 142.4 g
```

**Mound Height Ratio:**
* Empirically determined: 0.4 for typical serving
* Adjustable based on visual cues

### Strategy Selection Logic

```mermaid
graph TD
    A[Visual Stats + Food Code] --> B{Check Serving Unit<br/>from Database}

    B -->|bowl, cup, glass,<br/>katori| C[Container Strategy]
    B -->|piece, slice,<br/>number| D[Geometric Strategy]
    B -->|mound, heap,<br/>default| E[Mound Strategy]

    C --> F[Calculate Volume<br/>from Fill Ratio]
    D --> G[Calculate Volume<br/>from Cylinder]
    E --> H[Calculate Volume<br/>from Spherical Cap]

    F --> I[Mass = Volume × Density]
    G --> I
    H --> I

    I --> J[Query INDB Database<br/>for Macros per 100g]
    J --> K[Scale Macros<br/>to Actual Mass]
    K --> L[Enrich with<br/>Ingredients & Links]
    L --> M[Return Complete<br/>Nutrition Profile]

    style C fill:#e1f5ff
    style D fill:#ffe1f5
    style E fill:#fff5e1
    style J fill:#e1ffe1
```

### Density Lookup

Density values are retrieved from `Units.parquet`:

**Common Densities:**
* **Liquids:** 1.0 g/ml (water-based)
* **Rice:** 0.85 g/cm³ (cooked)
* **Bread:** 0.6 g/cm³ (roti, paratha)
* **Curry:** 0.9 g/cm³ (semi-solid)
* **Vegetables:** 0.7 g/cm³ (cooked)

**Lookup Process:**
1. Search `Units.parquet` for food name
2. Find "cup" or "bowl" measurement
3. Extract weight (e.g., "135g per cup")
4. Calculate density: weight / 240ml (standard cup)
5. Fallback to 0.85 g/cm³ if not found

### Macronutrient Calculation

Once mass is determined:

```python
# Query INDB database for macros per 100g
macros_per_100g = {
    "Energy (kcal)": 120,
    "Protein (g)": 3.5,
    "Carbohydrate (g)": 25.0,
    "Fat (g)": 0.5
}

# Scale to actual mass
actual_mass = 150  # grams
scaling_factor = actual_mass / 100

final_macros = {
    "Energy (kcal)": 120 × 1.5 = 180,
    "Protein (g)": 3.5 × 1.5 = 5.25,
    "Carbohydrate (g)": 25.0 × 1.5 = 37.5,
    "Fat (g)": 0.5 × 1.5 = 0.75
}
```

### Complete Nutrition Profile

The final output includes:

```python
{
    "food_name": "Paneer Butter Masala",
    "mass_g": 150.0,
    "calculation_strategy": "Container",
    "macros": {
        "Energy (kcal)": 450,
        "Protein (g)": 18,
        "Carbohydrate (g)": 12,
        "Fat (g)": 35
    },
    "metadata": {
        "ingredients": ["Paneer", "Tomato", "Cashew", "Butter", "Cream"],
        "serving_size": "1 bowl (200ml)",
        "source_link": "https://hebbarskitchen.com/paneer-butter-masala/",
        "density_g_ml": 0.9
    }
}
```

---

## 🧩 JSON Log Format (RAG-Ready)

Every prediction is saved automatically in this structure:

```json
{
    "timestamp": "2024-12-16T10:30:00",
    "total_summary": {
        "Energy (kcal)": 650,
        "Protein (g)": 25,
        "Carbohydrate (g)": 85,
        "Fat (g)": 18
    },
    "food_item_1": {
        "name": "Paneer Butter Masala",
        "class_id": "ASC022",
        "mass_g": 150.0,
        "calculation_strategy": "Container",
        "macros": {
            "Energy (kcal)": 450,
            "Protein (g)": 18,
            "Carbohydrate (g)": 12,
            "Fat (g)": 35
        },
        "metadata": {
            "ingredients": ["Paneer", "Tomato", "Cashew", "Butter", "Cream"],
            "serving_size": "1 bowl (200ml)",
            "source": "https://hebbarskitchen.com/paneer-butter-masala/",
            "density_g_ml": 0.9
        },
        "visual_stats": {
            "area_cm2": 120.5,
            "bbox": [100, 150, 300, 400],
            "confidence": 0.95
        }
    },
    "food_item_2": {
        "name": "Tandoori Roti",
        "class_id": "BFP122",
        "mass_g": 60.0,
        "calculation_strategy": "Geometric",
        "macros": {
            "Energy (kcal)": 200,
            "Protein (g)": 7,
            "Carbohydrate (g)": 73,
            "Fat (g)": 3
        },
        "metadata": {
            "ingredients": ["Wheat Flour", "Water", "Salt"],
            "serving_size": "1 piece",
            "source": "https://hebbarskitchen.com/tandoori-roti/",
            "density_g_cm3": 0.6
        },
        "visual_stats": {
            "area_cm2": 150.0,
            "bbox": [350, 100, 550, 350],
            "confidence": 0.92
        }
    }
}
```

---

## 🔬 Technical Specifications

### Model Architecture

**Unified Model:** Single EfficientNet-B5 trained on clean, background-removed crops

* **Backbone:** EfficientNet-B5 (Pretrained on ImageNet)
* **Input Resolution:** 512 × 512 pixels
  * **Why:** High resolution captures fine-grained textures (oil separation, spice patterns, grain structure)
* **Activation Function:** Swish (SiLU)
  * **Formula:** f(x) = x · sigmoid(x)
  * **Why:** Smooth, non-monotonic function allows small negative gradients to flow, helping deep networks learn complex patterns faster than ReLU
* **Model Parameters:** ~30M parameters
* **Output:** 55 food classes (Indian cuisine focus)

### Training Configuration

**Hardware Optimization:**

The training script auto-detects hardware and optimizes settings:

| Hardware | Batch Size | Precision | Workers | Training Time (50 epochs) |
|----------|-----------|-----------|---------|---------------------------|
| NVIDIA A6000 (48GB) | 96 | FP16 | 12 | ~2-3 hours |
| RTX 3090/4090 (24GB) | 64 | FP16 | 10 | ~4-5 hours |
| RTX 3060 (12GB) | 32 | FP16 | 8 | ~6-8 hours |
| MacBook Air M4 | 16 | FP32 | 4 | ~12-15 hours |
| CPU Only | 8 | FP32 | 4 | ~24-48 hours |

**Hyperparameters:**

* **Optimizer:** AdamW
  * Learning Rate: 1e-3
  * Weight Decay: 1e-4
  * β₁ = 0.9, β₂ = 0.999
* **Learning Rate Schedule:** Cosine Decay
  * Starts at 1e-3
  * Decays to 1e-6 over 50 epochs
  * Smooth annealing for better convergence
* **Loss Function:** Categorical Crossentropy
* **Metrics:** Accuracy, Top-3 Accuracy
* **Early Stopping:** Patience 10 epochs on validation loss
* **Model Checkpoint:** Save best model based on validation accuracy

### Data Augmentation Strategy

**Theory:** EfficientNet-B5 has 30M+ parameters. With limited food data, overfitting risk is high. We use **stochastic variance injection** to enforce invariances:

#### Augmentation Pipeline

**1. Geometric Transformations (Spatial Invariance)**

* **Random Rotation:** ±20° (0.2 factor)
  * **Why:** User photos are rarely perfectly aligned
  * **Effect:** Model learns rotation-invariant features

* **Random Zoom:** 0.8× to 1.2× (±20%)
  * **Why:** Addresses "distance ambiguity" - food scale varies with camera distance
  * **Effect:** Model learns scale-invariant features

* **Random Flip:** Horizontal
  * **Why:** Food presentation is chirality-invariant (samosa facing left = samosa facing right)
  * **Effect:** Doubles effective dataset size

**2. Photometric Transformations (Sensor Invariance)**

* **Random Contrast:** ±20% (factor 0.2)
  * **Why:** Simulates different lighting conditions (dim restaurant vs. bright kitchen)
  * **Effect:** Model learns lighting-invariant features

* **Random Brightness:** ±20% (factor 0.2)
  * **Why:** Handles exposure variations across cameras
  * **Effect:** Model learns exposure-invariant features

**3. Blur Transformations (Focus Invariance)**

* **Gaussian Blur:** 3×3 or 5×5 kernel, 30% probability
  * **Why:** Simulates depth-of-field and motion blur (common in smartphone photos)
  * **Effect:** Forces model to learn shape/structure, not just sharp textures
  * **Custom Layer:** `RandomGaussianBlur` (Keras-serializable)

**Mathematical Justification:**

```
Augmented_Dataset_Size = Original_Size × (1 + flip) × rotation_range × zoom_range
                       = N × 2 × 40° × 1.4
                       ≈ 112N effective samples
```

This 112× effective increase in dataset size significantly reduces overfitting.

### Segmentation & Background Removal

**YOLOv8-Seg:**
* **Model:** YOLOv8m-seg (medium variant)
* **Task:** Instance segmentation
* **Confidence Threshold:** 0.25 (captures more objects, filtered later)
* **Max Segments:** 8 (adaptive filtering keeps largest)
* **Output:** Bounding boxes + segmentation masks

**U2Net Background Removal:**
* **Model:** U2Net (Salient Object Detection)
* **Architecture:** Two-level nested U-structure
* **Input:** 320×320 pixels (per crop)
* **Output:** Alpha matte (0-255)
* **Processing:** Applied per-crop, not full image
* **Speed:** ~0.5-1 second per crop (CPU)

### Physics-Based Nutrition Calculation

**3 Strategies Based on Food Type:**

| Strategy | Food Types | Formula | Key Parameters |
|----------|-----------|---------|----------------|
| **Container** | Liquids, curries, dal | Volume = Container_Vol × Fill_Ratio<br/>Mass = Volume × Density | Container: 200ml<br/>Density: 1.0 g/ml |
| **Geometric** | Rotis, dosas, pieces | Volume = π × r² × h<br/>Mass = Volume × Density | Height: 0.3-0.5cm<br/>Density: 0.6 g/cm³ |
| **Mound** | Rice, sabzi, heaps | Volume = π × h² × (3r - h) / 3<br/>Mass = Volume × Density | Height: 0.4 × radius<br/>Density: 0.85 g/cm³ |

**Scale Recovery:**
* **Method:** Plate detection + ellipse fitting
* **Reference:** 28cm standard plate diameter
* **PPM Calculation:** Pixels-Per-Metric = Major_Axis_Pixels / 28cm
* **Accuracy:** ±5% error in controlled settings

### Database Architecture

**6 Parquet Databases:**

| Database | Size | Rows | Key Column | Purpose |
|----------|------|------|------------|---------|
| INDB.parquet | ~2MB | 500+ | food_code | Macronutrients per 100g |
| recipes.parquet | ~1MB | 2000+ | recipe_code | Ingredient lists |
| recipe_links.parquet | ~500KB | 300+ | Food Code | Recipe URLs |
| recipes_servingsize.parquet | ~300KB | 300+ | recipe_code | Serving sizes & units |
| Units.parquet | ~200KB | 100+ | Food items | Density data |
| recipes_names.parquet | ~100KB | 300+ | recipe_code | Display names |

**Query Performance:**
* **Format:** Apache Parquet (columnar storage)
* **Compression:** Snappy
* **Query Time:** <1ms per lookup (in-memory)
* **Schema Enforcement:** Prevents data type errors

### LLM Integration

**Qwen2.5-0.5B:**
* **Format:** GGUF (quantized)
* **Precision:** FP16
* **Size:** ~500MB
* **Context Length:** 2048 tokens
* **Inference:** llama-cpp-python (CPU-optimized)
* **Speed:** ~20 tokens/second (CPU)
* **Use Case:** Food trivia generation, future chat features

**LangGraph State Machine:**
* **States:** greeting, analyzing, answered, error
* **Transitions:** User input → State change → LLM response
* **Memory:** Session-based (in-memory)
* **RAG:** Simple keyword search in JSON logs

### Performance Characteristics

**Inference Speed:**
* **YOLO Segmentation:** ~0.2-0.5s (CPU), ~0.05s (GPU)
* **U2Net Background Removal:** ~0.5-1s per crop (CPU)
* **EfficientNet Classification:** ~0.1-0.3s per crop (CPU), ~0.02s (GPU)
* **Total per Image:** ~2-4s (CPU), ~0.5-1s (GPU)

**Model Sizes:**
* **EfficientNet-B5:** ~115MB (.keras format)
* **YOLOv8m-seg:** ~50MB (.pt format)
* **U2Net:** ~176MB (auto-downloaded)
* **Qwen2.5-0.5B:** ~500MB (.gguf format)
* **Total:** ~841MB

**Accuracy (Validation Set):**
* **Top-1 Accuracy:** ~85-90% (55 classes)
* **Top-3 Accuracy:** ~95-98%
* **Segmentation mAP:** ~0.75 (YOLO)

### Technical Justifications

**Why Parquet?**

Apache Parquet provides:
* **Schema enforcement** - Prevents data type errors
* **Columnar storage** - Fast queries on specific columns
* **Compression** - 5-10× smaller than Excel
* **Speed** - Sub-millisecond queries vs. seconds for Excel

**Why Swish Activation?**

Swish (f(x) = x · sigmoid(x)) advantages:
* **Smooth gradients** - No "dying neurons" like ReLU
* **Non-monotonic** - Allows small negative values to flow
* **Empirically better** - Proven to improve deep network training
* **Standard in EfficientNet** - Optimized for this architecture

**Why Batch Size 96 (A6000)?**

* **48GB VRAM** - Can fit large batches
* **Batch Normalization** - Larger batches = more stable statistics
* **Gradient Estimation** - Better gradient estimates with more samples
* **Training Speed** - Fewer iterations per epoch

**Why Background Removal?**

* **Texture Focus** - Removes distracting backgrounds
* **Generalization** - Model learns food features, not plate/table patterns
* **Robustness** - Works across different settings (home, restaurant, outdoor)
* **Accuracy Boost** - ~5-10% improvement in classification accuracy

---

## 📦 Modular Package Architecture

FoodVisionAI follows a clean, modular architecture with 124 comprehensive tests:

```mermaid
graph TB
    subgraph "Configuration Layer"
        A1[config/settings.py<br/>App Settings]
        A2[config/paths.py<br/>File Paths]
        A3[config/model_config.py<br/>Model Configs]
        A4[config/hardware.py<br/>Auto-Config]
    end

    subgraph "Data Layer"
        B1[src/data_tools/<br/>Background Removal]
        B2[src/data_tools/<br/>Folder Mapper]
        B3[src/data_tools/<br/>Parquet Converter]
        B4[src/data_tools/<br/>Save Labels]
    end

    subgraph "Model Layer"
        C1[src/models/builder.py<br/>Build Model]
        C2[src/models/augmentation.py<br/>Augmentation Pipeline]
        C3[src/models/loader.py<br/>Load/Save Model]
    end

    subgraph "Vision Layer"
        D1[src/segmentation/assessor.py<br/>YOLO Segmentation]
        D2[src/vision/inference.py<br/>Predict Food]
    end

    subgraph "Business Logic Layer"
        E1[src/nutrition/engine.py<br/>Nutrition Calculation]
        E2[src/chat/engine.py<br/>Chat Engine]
        E3[src/chat/llm.py<br/>LLM Wrapper]
        E4[src/chat/rag.py<br/>RAG Search]
    end

    subgraph "Utility Layer"
        F1[src/utils/image_utils.py<br/>Image Processing]
        F2[src/utils/file_utils.py<br/>File I/O]
        F3[src/utils/data_utils.py<br/>Data Manipulation]
        F4[src/utils/validation_utils.py<br/>Validation]
    end

    subgraph "Application Layer"
        G1[app.py<br/>Streamlit UI]
        G2[train.py<br/>Training Script]
    end

    subgraph "Testing Layer"
        H1[tests/<br/>124 Tests<br/>65% Coverage]
    end

    A1 --> G1
    A2 --> G1
    A3 --> C1
    A4 --> G2

    B1 --> D2
    B2 --> G2
    B3 --> E1
    B4 --> G1

    C1 --> G2
    C2 --> G2
    C3 --> G1

    D1 --> D2
    D2 --> G1

    E1 --> G1
    E2 --> G1
    E3 --> E2
    E4 --> E2

    F1 --> D2
    F2 --> G1
    F3 --> E1
    F4 --> D1

    H1 -.Tests.-> A1
    H1 -.Tests.-> B1
    H1 -.Tests.-> C1
    H1 -.Tests.-> D1
    H1 -.Tests.-> E1
    H1 -.Tests.-> F1

    style G1 fill:#e1f5ff
    style G2 fill:#ffe1f5
    style H1 fill:#e1ffe1
```

### Package Structure

```
FoodVisionAI/
├── config/              # Configuration (25 tests)
│   ├── settings.py      # App settings
│   ├── paths.py         # File paths
│   ├── model_config.py  # Model configs
│   └── hardware.py      # Auto-configuration
│
├── src/
│   ├── chat/            # Chat/LLM (19 tests)
│   │   ├── engine.py    # ChatEngine
│   │   ├── llm.py       # QwenLLM
│   │   └── rag.py       # SimpleRAG
│   │
│   ├── data_tools/      # Data processing (13 tests)
│   │   ├── background_removal.py
│   │   ├── folder_mapper.py
│   │   ├── parquet_converter.py
│   │   └── save_labels.py
│   │
│   ├── models/          # Model building (17 tests)
│   │   ├── builder.py
│   │   ├── augmentation.py
│   │   └── loader.py
│   │
│   ├── nutrition/       # Nutrition (12 tests)
│   │   └── engine.py
│   │
│   ├── segmentation/    # Segmentation (6 tests)
│   │   └── assessor.py
│   │
│   ├── utils/           # Utilities (28 tests)
│   │   ├── image_utils.py
│   │   ├── file_utils.py
│   │   ├── data_utils.py
│   │   └── validation_utils.py
│   │
│   └── vision/          # Vision (4 tests)
│       └── inference.py
│
├── tests/               # 124 tests, 65% coverage
│   ├── test_chat.py
│   ├── test_config.py
│   ├── test_data_tools.py
│   ├── test_models.py
│   ├── test_nutrition.py
│   ├── test_segmentation.py
│   ├── test_utils.py
│   └── test_vision.py
│
├── app.py               # Streamlit application
├── train.py             # Training script
└── requirements.txt     # Dependencies
```

### Design Patterns

* **Singleton:** BackgroundRemover, QwenLLM (heavy models loaded once)
* **Strategy:** NutrientEngine (3 calculation strategies)
* **State Machine:** ChatEngine (LangGraph conversation flow)
* **Lazy Loading:** LLM, BackgroundRemover (loaded on first use)

---

## 📊 Performance Metrics

### Inference Performance

| Metric | CPU (MacBook Air M4) | GPU (NVIDIA A6000) |
|--------|---------------------|-------------------|
| YOLO Segmentation | 0.3-0.5s | 0.05s |
| U2Net BG Removal (per crop) | 0.5-1.0s | 0.2s |
| EfficientNet Classification (per crop) | 0.2-0.3s | 0.02s |
| **Total per Image (3 items)** | **~3-5s** | **~0.5-1s** |
| Database Query | <1ms | <1ms |
| LLM Trivia Generation | 2-3s | 1-2s |

### Model Accuracy

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 Accuracy | 85-90% | 55 food classes |
| Top-3 Accuracy | 95-98% | User can verify |
| YOLO mAP@0.5 | ~0.75 | Segmentation quality |
| Nutrition Estimation Error | ±15-20% | Physics-based approach |

### Resource Usage

| Resource | Size/Usage | Notes |
|----------|-----------|-------|
| Model Files | ~841MB | All models combined |
| Database Files | ~4MB | 6 Parquet files |
| RAM Usage (Inference) | ~2-3GB | With all models loaded |
| VRAM Usage (Training) | ~35GB | Batch size 96 on A6000 |
| Disk Space (Full Setup) | ~5GB | Including datasets |

### Scalability

* **Supported Foods:** 55 classes (Indian cuisine)
* **Extensible:** Add new classes by retraining
* **Max Items per Image:** 8 (adaptive filtering)
* **Concurrent Users:** Limited by Streamlit (single-user app)
* **Database Scalability:** Parquet handles millions of rows efficiently

---

## 🛠️ Training Your Own Model

### Overview

FoodVisionAI uses a **single unified model** trained on clean, background-removed crops for optimal texture recognition. The training process involves:

1. Data preparation and normalization
2. YOLO-based cropping and background removal
3. EfficientNet-B5 training with heavy augmentation
4. Validation and model selection

### Training Workflow Diagram

```mermaid
graph TB
    A[Raw Images<br/>data/raw/images/] --> B[Step 1: Normalize<br/>folder_mapper.py]
    B --> C[Standardized Folders<br/>ASC022, BFP122, etc.]
    C --> D[Step 2: YOLO Processing<br/>yolo_processor.py]
    D --> E[Detect & Crop<br/>Food Objects]
    E --> F[U2Net Background<br/>Removal]
    F --> G[Clean Crops<br/>data/yolo_processed/]
    G --> H[Step 3: Training<br/>train.py]
    H --> I[EfficientNet-B5<br/>+ Augmentation]
    I --> J[Validation<br/>Monitor Accuracy]
    J --> K{Best Model?}
    K -->|Yes| L[Save Checkpoint<br/>model_yolo_best.keras]
    K -->|No| I

    style G fill:#e1f5ff
    style L fill:#e1ffe1
```

### Step-by-Step Training Guide

#### Step 1: Prepare Raw Data

Organize your raw images in class folders:

```
data/raw/images/
├── chicken_pizza/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── paneer_butter_masala/
│   ├── img001.jpg
│   └── ...
└── ...
```

**Requirements:**
* Minimum 50-100 images per class
* Recommended 200+ images per class
* Images should show food in natural settings
* Variety in lighting, angles, backgrounds

#### Step 2: Normalize Folder Names

Run the folder mapper to standardize names to food codes:

```bash
python src/data_tools/folder_mapper.py
```

This will:
* Map folder names to standard food codes (e.g., "chicken pizza" → `BFP122`)
* Remove unmapped or invalid folders
* Create initial train/val splits

**Output:** `data/processed/` with normalized folders

#### Step 3: Generate Clean Crops

Run YOLO processor to create clean, background-removed crops:

```bash
python src/data_tools/yolo_processor.py
```

This will:
* Detect food objects using YOLOv8
* Crop each object with bounding box
* Remove backgrounds using U2Net
* Save clean crops to `data/yolo_processed/train/` and `data/yolo_processed/val/`

**Processing Time:** ~2-4 images/second (CPU), ~8-10 images/second (GPU)

**Output:** `data/yolo_processed/` with clean crops

#### Step 4: Freeze Class Labels

Generate the class names file:

```bash
python src/data_tools/save_labels.py
```

This creates `class_names.npy` with the exact class order used during training.

**Important:** This file must match your model's output layer!

#### Step 5: Train the Model

Run the training script:

```bash
python train.py
```

**Training Configuration:**

* **Model:** EfficientNet-B5 (30M parameters)
* **Input Size:** 512×512 pixels
* **Batch Size:** 96 (A6000 GPU) or auto-configured for your hardware
* **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
* **Learning Rate:** Cosine decay schedule
* **Epochs:** 50 (with early stopping)
* **Augmentation:**
  * Random rotation (±20°)
  * Random zoom (0.8-1.2×)
  * Random flip (horizontal)
  * Random contrast/brightness (±20%)
  * Gaussian blur (30% probability)

**Hardware-Specific Settings:**

The training script auto-detects your hardware and optimizes settings:

* **NVIDIA A6000 (48GB):** Batch size 96, FP16 precision
* **RTX 3090/4090 (24GB):** Batch size 64, FP16 precision
* **RTX 3060 (12GB):** Batch size 32, FP16 precision
* **MacBook Air M4:** Batch size 16, FP32 precision, thermal-aware
* **CPU Only:** Batch size 8, FP32 precision

**Training Time Estimates:**

* **A6000 GPU:** ~2-3 hours for 50 epochs (100K images)
* **RTX 3090:** ~4-5 hours
* **MacBook Air M4:** ~12-15 hours
* **CPU Only:** ~24-48 hours

**Monitoring Training:**

The script will display:
* Epoch progress
* Training/validation accuracy
* Training/validation loss
* Best model checkpoints

**Output:** `models/checkpoints/model_yolo_best.keras`

#### Step 6: Validate the Model

After training, test your model:

```bash
# Run the app with your new model
streamlit run app.py

# Upload test images to verify accuracy
```

### Advanced Training Options

#### Custom Augmentation

Edit `src/models/augmentation.py` to customize augmentation:

```python
def get_augmentation_pipeline():
    return keras.Sequential([
        layers.RandomRotation(0.2),  # ±20°
        layers.RandomZoom(0.2),      # ±20%
        # Add your custom layers here
    ])
```

#### Transfer Learning

To fine-tune on a new dataset:

```python
# In train.py, load existing model
model = load_model("models/checkpoints/model_yolo_best.keras")

# Freeze early layers
for layer in model.layers[:-10]:
    layer.trainable = False

# Train with lower learning rate
optimizer = keras.optimizers.AdamW(learning_rate=1e-4)
```

#### Multi-GPU Training

For distributed training on multiple GPUs:

```python
# In train.py, add strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model(num_classes)
    model.compile(...)
```

### Database Setup

Place database files in `data/raw/metadata/`:

* `INDB.xlsx` - Nutrition data
* `recipes.xlsx` - Ingredients
* `recipes_names.xlsx` - Display names
* `recipes_servingsize.xlsx` - Serving sizes
* `recipe_links.xlsx` - Recipe URLs
* `Units.xlsx` - Density data

Convert to Parquet:

```bash
python src/data_tools/parquet_converter.py
```

**Output:** `data/parquet_db/*.parquet`

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

Contributions are welcome! Here are some areas where you can help:

### High Priority
* **Expand Food Database:** Add more Indian dishes and regional cuisines
* **Improve Volume Estimation:** Refine geometric formulas for better accuracy
* **Multi-Cuisine Support:** Extend to Chinese, Italian, Mexican cuisines
* **Mobile App:** Port to mobile (TensorFlow Lite, ONNX)

### Medium Priority
* **Enhanced LLM Features:** Better conversational AI, dietary recommendations
* **User Authentication:** Multi-user support with meal history
* **Cloud Deployment:** Deploy on AWS/GCP/Azure
* **API Development:** REST API for third-party integrations

### Low Priority
* **More Tests:** Increase coverage from 65% to 80%+
* **Documentation:** Add more examples and tutorials
* **Performance Optimization:** Faster inference, model quantization
* **UI/UX Improvements:** Better visualizations, dark mode

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make your changes** and add tests
4. **Run tests:** `pytest tests/ -v`
5. **Commit:** `git commit -m "Add your feature"`
6. **Push:** `git push origin feature/your-feature`
7. **Create Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

### Data Sources
* **Indian Food Database (INDB)** - Nutritional data for Indian foods
* **Hebbar's Kitchen** - Recipe sources and ingredient lists
* **Various Recipe Websites** - Additional recipe data

### Technologies
* **YOLOv8** (Ultralytics) - State-of-the-art object detection and segmentation
* **EfficientNet** (Google) - Efficient image classification architecture
* **U2Net** - Salient object detection for background removal
* **Qwen2.5** (Alibaba) - Lightweight LLM for chat features
* **TensorFlow/Keras** - Deep learning framework
* **Streamlit** - Rapid web app development
* **Apache Parquet** - High-performance columnar storage

### Research
* **EfficientNet Paper:** Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
* **YOLO Paper:** Redmon et al. (2016) - "You Only Look Once: Unified, Real-Time Object Detection"
* **U2Net Paper:** Qin et al. (2020) - "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection"

---

## 📧 Contact

**Developer:** Nitanshu Idnani
**GitHub:** [@Nitanshu99](https://github.com/Nitanshu99)
**Email:** nitts420@gmail.com
**Project:** [FoodVisionAI](https://github.com/Nitanshu99/FoodVisionAI)

---

## 🎯 Project Status

**Current Version:** 1.0.0
**Status:** Active Development
**Last Updated:** December 2024

### Roadmap

- [x] **Phase 1-10:** Complete modularization (124 tests, 65% coverage)
- [x] **Core Features:** YOLO segmentation, EfficientNet classification, nutrition calculation
- [x] **LLM Integration:** Qwen2.5 for trivia generation
- [ ] **Phase 11:** Enhanced chat features (Q&A, meal history)
- [ ] **Phase 12:** Mobile app development
- [ ] **Phase 13:** Cloud deployment
- [ ] **Phase 14:** Multi-cuisine support

---

## 📚 Additional Resources

* **Architecture Documentation:** See [ARCHITECTURE.md](ARCHITECTURE.md)
* **Migration Guide:** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
* **API Documentation:** Coming soon
* **Video Tutorial:** Coming soon

---

**Built with ❤️ for better dietary awareness**