# FoodVisionAI Architecture

## Package Structure

```
FoodVisionAI/
├── config/              # Configuration modules
│   ├── settings.py      # App settings (image size, hyperparameters, device)
│   ├── paths.py         # File paths (models, data, databases)
│   ├── model_config.py  # Model configurations (EfficientNet, YOLO, LLM)
│   └── hardware.py      # Hardware detection and auto-configuration
│
├── src/
│   ├── chat/            # Chat/LLM modules
│   │   ├── engine.py    # ChatEngine (LangGraph state machine)
│   │   ├── llm.py       # QwenLLM wrapper (Qwen2.5-0.5B GGUF)
│   │   └── rag.py       # SimpleRAG (session log search)
│   │
│   ├── data_tools/      # Data processing modules
│   │   ├── background_removal.py  # U2Net background removal
│   │   ├── folder_mapper.py       # Raw folder → food code mapping
│   │   ├── parquet_converter.py   # Excel → Parquet converter
│   │   ├── inspect_headers.py     # Parquet schema inspector
│   │   └── save_labels.py         # Class labels freezer
│   │
│   ├── models/          # Model building modules
│   │   ├── builder.py       # Model building (build_model)
│   │   ├── augmentation.py  # Data augmentation (RandomGaussianBlur)
│   │   └── loader.py        # Model loading/saving utilities
│   │
│   ├── nutrition/       # Nutrition calculation modules
│   │   └── engine.py    # NutrientEngine (6 databases, 3 strategies)
│   │
│   ├── segmentation/    # Segmentation modules
│   │   └── assessor.py  # DietaryAssessor (YOLO + geometry)
│   │
│   ├── utils/           # Utility modules
│   │   ├── image_utils.py      # Image processing utilities
│   │   ├── file_utils.py       # File I/O utilities
│   │   ├── data_utils.py       # Data manipulation utilities
│   │   └── validation_utils.py # Validation utilities
│   │
│   └── vision/          # Vision inference modules
│       └── inference.py # Vision inference pipeline (predict_food)
│
├── tests/               # Test suite (124 tests)
│   ├── test_chat.py         # Chat module tests (19 tests)
│   ├── test_config.py       # Config module tests (25 tests)
│   ├── test_data_tools.py   # Data tools tests (13 tests)
│   ├── test_models.py       # Model module tests (17 tests)
│   ├── test_nutrition.py    # Nutrition module tests (12 tests)
│   ├── test_segmentation.py # Segmentation tests (6 tests)
│   ├── test_utils.py        # Utility tests (28 tests)
│   └── test_vision.py       # Vision module tests (4 tests)
│
├── app.py               # Streamlit UI application
└── train.py             # Model training script
```

## Import Examples

### Configuration
```python
from config import settings, paths, model_config, hardware

# Access settings
img_size = settings.IMG_SIZE
batch_size = settings.BATCH_SIZE

# Access paths
model_path = paths.GLOBAL_MODEL_PATH
db_files = paths.DB_FILES

# Access model config
efficientnet_config = model_config.EFFICIENTNET_CONFIG
```

### Chat/LLM
```python
from src.chat import ChatEngine, get_llm, SimpleRAG

# Initialize chat engine
chat_engine = ChatEngine()

# Add meal log and generate trivia
chat_engine.add_meal_log(log_data)
trivia = chat_engine.generate_trivia(log_data)

# Answer questions
response = chat_engine.answer_question("What did I eat?")
```

### Data Tools
```python
from src.data_tools import BackgroundRemover, get_manual_mapping, group_sources_by_target

# Remove background
bg_remover = BackgroundRemover()
processed_img = bg_remover.process_image(image)

# Get folder mapping
mapping = get_manual_mapping()
grouped = group_sources_by_target(mapping)
```

### Models
```python
from src.models import build_model, get_augmentation_pipeline, load_model, save_model

# Build model
model = build_model(num_classes=100)

# Get augmentation
aug_pipeline = get_augmentation_pipeline()

# Load/save model
model = load_model("path/to/model.keras")
save_model(model, "path/to/save.keras")
```

### Nutrition
```python
from src.nutrition import NutrientEngine

# Initialize engine
engine = NutrientEngine()

# Get nutrition info
nutrition = engine.calculate_nutrition(
    food_code="F0001",
    mass_g=100,
    volume_cm3=150,
    strategy="container"
)
```

### Segmentation
```python
from src.segmentation import DietaryAssessor

# Initialize assessor
assessor = DietaryAssessor()

# Analyze scene
results = assessor.analyze_scene(image)
```

### Utilities
```python
from src.utils import (
    preprocess_for_model, resize_image, normalize_image,  # image_utils
    ensure_directory, save_json, load_json,                # file_utils
    safe_float, clamp, is_valid_number,                    # data_utils
    validate_image, validate_bbox, validate_confidence     # validation_utils
)
```

### Vision
```python
from src.vision import predict_food

# Run inference
results = predict_food(
    image=image,
    model=model,
    class_names=class_names,
    assessor=assessor,
    bg_remover=bg_remover
)
```

## Key Design Patterns

### Singleton Pattern
- **BackgroundRemover** - Heavy U2Net model loaded once
- **QwenLLM** - LLM model loaded once via `get_llm()`

### Strategy Pattern
- **NutrientEngine** - 3 calculation strategies (container, piece, mound)

### State Machine Pattern
- **ChatEngine** - LangGraph state machine for conversation flow

### Lazy Loading
- **LLM** - Model loaded only when first accessed
- **BackgroundRemover** - Model loaded on first use

## Testing Strategy

- **124 total tests** across 8 test files
- **Unit tests** for all modules
- **Integration tests** for cross-module functionality
- **Mock-based testing** for heavy models (YOLO, U2Net, LLM)
- **pytest** with coverage tracking

