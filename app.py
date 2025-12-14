"""
Streamlit Application Frontend.

Implements the "Zero-Touch UI" for FoodVisionAI, utilizing a chat-like
structure for future LLM/RAG integration.

Flow: Upload Image -> B5 Prediction -> Nutrient Engine Lookup -> Display Result.
"""

import os
import streamlit as st
import pandas as pd
import keras

# Local imports
from src import config
from src.vision_model import build_model
from src.nutrient_engine import NutrientEngine
from src.vision_utils import preprocess_image, predict_food, get_class_names
from src.augmentation import RandomGaussianBlur  # Required for model loading


# --- 1. Initialization and Setup ---

st.set_page_config(
    page_title="FoodVisionAI - High-Fidelity Dietary Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading Vision Model (EfficientNet-B5)...")
def load_vision_model():
    """
    Load the final Keras model from disk.
    
    Returns:
        keras.Model: The trained FoodVision B5 model.
    """
    model_path = config.FINAL_MODEL_PATH
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run train.py first.")
        # Fallback for UI testing if model is missing
        return build_model(num_classes=100)

    try:
        # Load the trained model.
        # We include custom_objects because the training pipeline used
        # a custom augmentation layer (RandomGaussianBlur).
        custom_objects = {"RandomGaussianBlur": RandomGaussianBlur}
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # Return dummy if load fails to prevent crash
        return build_model(num_classes=100)


@st.cache_resource(show_spinner="Starting Logic Engine (Parquet DB Lookup)...")
def load_nutrient_engine():
    """Initialize the Nutrient Engine for instant Parquet lookups."""
    try:
        return NutrientEngine()
    except FileNotFoundError:
        st.error("FATAL ERROR: Parquet database files not found.")
        st.stop()

@st.cache_resource(show_spinner="Indexing Class Labels...")
def load_labels():
    """Load the class names (ASC codes) from the dataset directory."""
    return get_class_names()


# Load resources
MODEL = load_vision_model()
ENGINE = load_nutrient_engine()
CLASS_NAMES = load_labels()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 2. Main Logic ---

def process_upload(uploaded_file):
    """
    Handles the full image processing and assessment pipeline.
    
    Args:
        uploaded_file: The uploaded image file object.
    """
    
    # 1. User Message (Image Upload)
    with st.chat_message("user"):
        st.image(uploaded_file, caption="User Upload", width="stretch")
    
    st.session_state.messages.append({
        "role": "user",
        "content": uploaded_file.name,
        "image": uploaded_file
    })

    # 2. System Analysis (B5 and Logic Engine)
    with st.chat_message("assistant"):
        
        with st.status("Analyzing image...", expanded=True) as status:
            
            # A. Preprocessing
            st.write("1/3: Pre-processing image (512x512 resolution)...")
            image_tensor = preprocess_image(uploaded_file)

            # B. Vision Prediction (B5 Model)
            st.write("2/3: Running EfficientNet-B5 inference...")
            prediction_output = predict_food(
                model=MODEL,
                image_tensor=image_tensor,
                class_names=CLASS_NAMES
            )
            
            if "error" in prediction_output:
                st.error(f"Prediction Error: {prediction_output['error']}")
                status.update(label="Analysis Failed", state="error", expanded=False)
                return

            class_id = prediction_output["class_id"]
            visual_stats = prediction_output["visual_stats"]
            top_preds = prediction_output["top_predictions"]
            
            # C. Logic Engine Calculation (Parquet Lookup)
            st.write(f"3/3: Querying Parquet DB for nutrition profile of {class_id}...")
            nutrition_result = ENGINE.calculate_nutrition(class_id, visual_stats)

            status.update(label="Analysis Complete", state="complete", expanded=False)

        # 3. Display Result
        
        food_name = nutrition_result.get("Food Name", class_id)
        st.subheader(f"✅ Predicted Food: **{food_name}**")
        st.caption(f"Code: {class_id} | Confidence: {top_preds[0][1]:.2f}")
        
        st.markdown("---")
        
        # Display Core Nutrition
        st.metric(
            label="Total Energy", 
            value=f"{nutrition_result.get('Energy (kcal)', 0.0):.1f} kcal", 
            delta=f"Mass: {nutrition_result.get('Calculated Mass (g)', 0.0):.1f} g"
        )

        # Display Logic Engine Decision
        st.info(
            f"**Logic Engine Decision:** {nutrition_result.get('Logic Path', 'N/A')}"
            f" based on Unit: **{nutrition_result.get('Detected Unit', 'N/A')}**"
        )
        
        # Detailed Breakdown
        st.subheader("Nutritional Breakdown")
        
        data_display = {
            "Macronutrient": ["Protein", "Carbohydrate", "Fat"],
            "Amount (g)": [
                nutrition_result.get("Protein (g)", 0.0),
                nutrition_result.get("Carbohydrate (g)", 0.0),
                nutrition_result.get("Fat (g)", 0.0),
            ]
        }
        st.dataframe(pd.DataFrame(data_display).set_index("Macronutrient"))

    st.session_state.messages.append({"role": "assistant", "content": nutrition_result})


# --- 3. UI Structure (Chat and Sidebar) ---

st.title("FoodVisionAI 🍲")
st.markdown("### High-Fidelity Automated Dietary Assessment")

# Display previous messages (CORRECTED LOOP)
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            # Handle potential missing image ref on reload (optional safety)
            if "image" in message:
                st.image(message["image"], caption=message["content"], width="stretch")
            else:
                st.write(message["content"])

    elif message["role"] == "assistant":
        result = message["content"]
        with st.chat_message("assistant"):
            # FIX: Try 'Food Name' first, fallback to 'Food Code' if missing
            display_name = result.get('Food Name', result.get('Food Code', 'Unknown Food'))
            st.subheader(f"✅ Predicted Food: **{display_name}**")
            
            st.markdown("---")
            st.metric(
                label="Total Energy", 
                value=f"{result.get('Energy (kcal)', 0.0):.1f} kcal",
                delta=f"Mass: {result.get('Calculated Mass (g)', 0.0):.1f} g"
            )
            
            # Helper to safely get string values
            logic_path = result.get('Logic Path', 'N/A')
            unit = result.get('Detected Unit', 'N/A')
            st.info(f"**Logic Path:** {logic_path} (Unit: {unit})")
            
# Chat Input
prompt = st.chat_input("Ask a question or upload an image to begin...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        st.write("I am currently configured for image analysis only. Please upload a food image.")
    st.session_state.messages.append({"role": "assistant", "content": "Image analysis request."})

# Image Upload Sidebar
with st.sidebar:
    st.header("Upload Food Image")
    uploaded_file = st.file_uploader(
        "Upload a photo of your meal (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        process_upload(uploaded_file)