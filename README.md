# FoodVisionAI – High-Fidelity Automated Dietary Assessment

## 1. Data Strategy: The "parquet" Optimization

We are modernizing the data pipeline to ensure instant lookup speeds and strict data typing (preventing "string vs number" errors in the app).

### Step A: Manual Label Mapping

Rename image folders to match Food Codes (e.g., `aloo_gobi` → `ASC171`).

**Why:** Ensures the model predicts the exact key needed for the database.

### Step B: Parquet Conversion (New Step)

**Action:** Write a simple script to read all Excel/CSV files (`INDB.xlsx`, `recipes.xlsx`, etc.) and save them as `.parquet` files.

**Benefit:** Parquet files load 10x faster than Excel and preserve schema (e.g., ensuring "Energy" is always a float, never a string), which is critical for the "Logic Engine."

---

## 2. Architectural Specifications (The "Brain")

We replace the standard B0 model with the B5 variant, optimized for your A6000 hardware.

- **Backbone:** EfficientNet-B5 (Pretrained on ImageNet)
- **Input Resolution:** 512 × 512 pixels
  - **Why:** High resolution is required to detect fine-grained textures (e.g., oil separation)
- **Activation Function:** Swish (SiLU)
  - **Why:** Unlike ReLU (used in older ResNets), Swish is a smooth, non-monotonic function (f(x) = x · sigmoid(x)) that allows a small amount of negative information to flow through. This is mathematically proven to help deep networks like B5 learn complex patterns (like food textures) faster.
- **Training Hardware:** NVIDIA RTX A6000 (48GB VRAM)

### Hyperparameters

- **Batch Size:** 64 (Optimal balance of speed and Batch Norm stability)
- **Optimizer:** AdamW (Weight Decay: 1e-4)
- **Learning Rate:** 1e-3 with Cosine Decay Schedule
- **Precision:** FP32 (Standard 32-bit Floating Point)

### 2.1 Data Augmentation Strategy (The "Regularization Shield")

**Theory:** EfficientNet-B5 is a high-capacity model (30M+ parameters). Given the "small local-food dataset," the risk of overfitting (memorizing noise instead of features) is mathematically high. We introduce a stochastic "Variance Injection" pipeline to enforce Rotational, Scale, and Photometric Invariance, ensuring the model generalizes to unseen real-world photos.

#### Pipeline Specifications (Applied Pre-Batch)

**Geometric Transformations (Spatial Invariance):**

- **Random Rotation:** ±20°
  - **Justification:** User photos will rarely be perfectly aligned.
- **Random Zoom:** Range [0.8, 1.2] (20% zoom in/out)
  - **Justification:** Addresses the "Distance Ambiguity" problem where the food's scale varies relative to the camera frame.
- **Random Flip:** Horizontal
  - **Justification:** Food presentation is chirality-invariant (a samosa facing left is still a samosa).

**Photometric Transformations (Sensor Invariance):**

- **Random Contrast & Brightness:** Factor 0.2
  - **Justification:** Simulates varying lighting conditions (e.g., dim restaurant vs. bright kitchen).

**Blur Transformations (Focus Invariance):**

- **Gaussian Blur:** Kernel Size 3×3 or 5×5, applied with 30% probability
  - **Why:** This simulates Depth of Field and Motion Blur (common in hasty smartphone photography). It forces the model to learn "shape" and "global structure" rather than relying solely on sharp, high-frequency textures that might disappear in a blurry photo.

---

## 3. The "Logic Engine": Automated Portion Control

This module replaces user sliders with Computer Vision Heuristics, addressing "Distance Ambiguity" by consulting the parquet database.

### Step A: The Prediction

- **Input:** User Photo
- **Output:** Class ID `ASC004` (Iced Tea)

### Step B: The "Smart Switch" (Database Query)

The system reads the `servings_unit` column in `recipes_servingsize.parquet` to decide how to measure.

#### Logic Path 1: "Discrete Units"

- **Trigger:** Unit is "Piece", "Slice", "Number" (e.g., Samosa, Idli)
- **Algorithm:** Instance Counting. The model counts distinct objects.
- **Math:** Count × Unit Mass

#### Logic Path 2: "Container Units"

- **Trigger:** Unit is "Bowl", "Cup", "Glass", "Plate" (e.g., Dal, Tea)
- **Algorithm:** Fill-Level Ratio (Occupancy)
- **Method:**
  - Identify Container pixels (C) vs. Food pixels (F)
  - Calculate Ratio R = F / C
  - **Thresholds:**
    - R > 0.8 → 1.0x Serving (Full Bowl)
    - R < 0.6 → 0.5x Serving (Half Bowl)

---

## 4. Execution Roadmap (4-Day Sprint)

| Day | Focus | Task Detail |
|-----|-------|-------------|
| **Day 1** | Data & Training | • **Data Prep:** Download images. Rename folders to ASCxxx.<br>• **Conversion:** Run `pandas.to_parquet()` on all Excel files.<br>• **Training:** Launch B5 job on A6000 (Batch: 64, Swish, 512px). |
| **Day 2** | Logic Backend | • **Code:** Create `NutrientEngine` class.<br>• **Integration:** Load `recipes_servingsize.parquet` and `INDB.parquet`.<br>• **Algorithm:** Write the logic to switch between "Counting" and "Ratio" based on the unit type. |
| **Day 3** | Zero-Touch UI | • **Frontend:** Streamlit App.<br>• **Flow:** Upload Image → Backend Analysis → Result.<br>• **Display:** "Detected: Paneer Curry" |
| **Day 4** | Delivery | • **Docs:** Diagram the pipeline (Image → B5 → Parquet Lookup → Output).<br>• **Presentation:** Defend the use of Swish/B5/Parquet for high-performance AI. |

---

## 5. Technical Justification (For Presentation)

### Why Parquet?

"We converted legacy Excel datasets to Apache Parquet. This provides schema enforcement (preventing data type errors) and allows our Logic Engine to query nutritional priors in milliseconds, ensuring a seamless user experience."

### Why Swish Activation?

"We utilized the Swish activation function (standard in EfficientNet) because its smooth, non-linear property prevents 'dying neurons' during the training of deep networks, allowing our B5 model to capture subtle food texture gradients better than ReLU."

### Why Batch Size 64?

"Leveraging the 48GB VRAM of the A6000, we maximized the batch size to 64. This ensures stable statistical estimation in the Batch Normalization layers, which is critical for generalizing across the high intra-class variance of Indian food."

---

## 6. Project Structure

```
FoodVisionAI/
│
├── data/
│   ├── raw/
│   │   ├── images/                 # COPY original folders here (e.g., "aloo_gobi", "samosa")
│   │   └── metadata/               # PLACE EXACTLY THESE 6 FILES HERE:
│   │       ├── INDB.xlsx
│   │       ├── recipes.xlsx
│   │       ├── recipes_names.xlsx
│   │       ├── recipes_servingsize.xlsx
│   │       ├── recipe_links.xlsx
│   │       └── Units.xlsx
│   │
│   └── processed/                  # Output of src/data_tools/
│       ├── train/                  # 90% of images (Renamed to ASCxxx)
│       ├── val/                    # 10% of images (Renamed to ASCxxx)
│       └── parquet_db/             # Converted Parquet files (for Logic Engine)
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Constants (IMG_SIZE=512, BATCH=64, SPLIT=0.9)
│   │
│   ├── data_tools/                 # Data Engineering Modules
│   │   ├── __init__.py
│   │   ├── folder_mapper.py        # Script A: Renames -> Shuffles -> Splits (90/10)
│   │   └── parquet_converter.py    # Script B: Converts Excel -> Parquet
│   │
│   ├── augmentation.py             # Gaussian Blur, Rotation, Zoom logic
│   ├── vision_model.py             # EfficientNetB5 + Swish Definition
│   └── nutrient_engine.py          # The Logic Backend (Class counting vs. Volume ratio)
│
├── models/
│   ├── checkpoints/                # Autosaves during training
│   └── food_vision_b5.keras        # Final saved model
│
├── train.py                        # Main Execution: Loads Data -> Trains B5 -> Saves Model
├── app.py                          # Frontend: Loads Model + Parquet DB -> Streamlit UI
├── requirements.txt                # Dependencies (tensorflow, pandas, pyarrow, streamlit)
└── README.md                       # Documentation
```

---

## Key Features

✅ **High-Precision Recognition:** EfficientNet-B5 with 512px resolution  
✅ **Automated Portion Control:** Logic Engine eliminates manual input  
✅ **Optimized Performance:** Parquet database for instant queries  
✅ **Robust Generalization:** Advanced augmentation pipeline  
✅ **Production-Ready:** 4-day sprint to deployment