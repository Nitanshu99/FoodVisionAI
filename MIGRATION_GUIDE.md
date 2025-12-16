# Migration Guide: Old → New Package Structure

## Overview

This guide helps you migrate from the old monolithic structure to the new modular package structure.

## Import Changes

### Configuration

**Old:**
```python
import config
from src.config import IMG_SIZE, BATCH_SIZE, GLOBAL_MODEL_PATH
```

**New:**
```python
from config import settings, paths, model_config, hardware

# Access settings
img_size = settings.IMG_SIZE
batch_size = settings.BATCH_SIZE

# Access paths
model_path = paths.GLOBAL_MODEL_PATH
```

---

### Model Building

**Old:**
```python
from src.vision_model import build_model
from src.augmentation import RandomGaussianBlur, get_augmentation_pipeline
```

**New:**
```python
from src.models import build_model, RandomGaussianBlur, get_augmentation_pipeline
```

---

### Model Loading/Saving

**Old:**
```python
import keras
model = keras.models.load_model("path.keras", custom_objects={...})
model.save("path.keras")
```

**New:**
```python
from src.models import load_model, save_model

model = load_model("path.keras")  # Handles custom objects automatically
save_model(model, "path.keras")
```

---

### Vision Inference

**Old:**
```python
from src.vision_utils import predict_food
```

**New:**
```python
from src.vision import predict_food
```

---

### Utilities

**Old:**
```python
from src.vision_utils import preprocess_for_model, resize_image
from src.nutrient_engine import safe_float, clamp
```

**New:**
```python
from src.utils import preprocess_for_model, resize_image, safe_float, clamp
```

---

### Nutrition

**Old:**
```python
from src.nutrient_engine import NutrientEngine
```

**New:**
```python
from src.nutrition import NutrientEngine
```

---

### Segmentation

**Old:**
```python
from src.segmentation import DietaryAssessor
```

**New:**
```python
from src.segmentation import DietaryAssessor  # Same import path!
```

---

### Chat/LLM

**Old:**
```python
from src.chat_engine import ChatEngine
from src.llm_utils import get_llm
from src.rag_engine import SimpleRAG
```

**New:**
```python
from src.chat import ChatEngine, get_llm, SimpleRAG
```

---

### Data Tools

**Old:**
```python
from src.data_tools.background_removal import BackgroundRemover
from src.data_tools.folder_mapper import get_manual_mapping
```

**New:**
```python
from src.data_tools import BackgroundRemover, get_manual_mapping
# Or still use the old import path - both work!
```

---

## Quick Reference Table

| Old Import | New Import |
|------------|------------|
| `from src.config import IMG_SIZE` | `from config.settings import IMG_SIZE` |
| `from src.vision_model import build_model` | `from src.models import build_model` |
| `from src.augmentation import RandomGaussianBlur` | `from src.models import RandomGaussianBlur` |
| `from src.vision_utils import predict_food` | `from src.vision import predict_food` |
| `from src.vision_utils import preprocess_for_model` | `from src.utils import preprocess_for_model` |
| `from src.nutrient_engine import NutrientEngine` | `from src.nutrition import NutrientEngine` |
| `from src.segmentation import DietaryAssessor` | `from src.segmentation import DietaryAssessor` ✅ |
| `from src.chat_engine import ChatEngine` | `from src.chat import ChatEngine` |
| `from src.llm_utils import get_llm` | `from src.chat import get_llm` |
| `from src.rag_engine import SimpleRAG` | `from src.chat import SimpleRAG` |

---

## Benefits of New Structure

### Better Organization
- Related functionality grouped together
- Clear separation of concerns
- Easier to navigate codebase

### Improved Maintainability
- Smaller, focused modules
- Easier to test individual components
- Reduced code duplication

### Enhanced Discoverability
- Logical package hierarchy
- Consistent naming conventions
- Clear import paths

### Testing
- 124 comprehensive tests
- Better test organization
- Higher code coverage

---

## Backward Compatibility

Most imports remain compatible! The new structure maintains backward compatibility where possible:

✅ **Still works:** `from src.segmentation import DietaryAssessor`  
✅ **Still works:** `from src.data_tools.background_removal import BackgroundRemover`

Only these imports need updates:
- `src.config` → `config.*`
- `src.vision_model` → `src.models`
- `src.augmentation` → `src.models`
- `src.vision_utils` → `src.vision` or `src.utils`
- `src.nutrient_engine` → `src.nutrition`
- `src.chat_engine` → `src.chat`
- `src.llm_utils` → `src.chat`
- `src.rag_engine` → `src.chat`

---

## Need Help?

Check `ARCHITECTURE.md` for detailed package structure and import examples.

