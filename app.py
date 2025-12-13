"""
Streamlit Application Frontend.

Implements the "Zero-Touch UI" for FoodVisionAI, utilizing a chat-like
structure for future LLM/RAG integration.

Flow: Upload Image -> B5 Prediction -> Nutrient Engine Lookup -> Display Result.
"""

import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
import time
from typing import Dict, Any, List

# Local imports
from src import config
from src.vision_model import build_model
from src.nutrient_engine import NutrientEngine
from src.vision_utils import preprocess_image, predict_food


# --- 1. Initialization and Setup ---

# Set Streamlit Page Config
st.set_page_config(
    page_title="FoodVisionAI - High-Fidelity Dietary Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading Vision Model (EfficientNet-B5)...")
def load_vision_model():
    """Load the final Keras model from disk."""
    model_path = config.FINAL_MODEL_PATH
    
    # NOTE: Since we are in Day 2/3, the model is still training. 
    # We build a dummy model for structure compatibility.
    
    if os.path.exists(model_path):
        # In production:
        # return st.keras.models.load_model(model_path)
        pass

    # Placeholder for Day 3: Build a small dummy model structure 
    # compatible with the B5 definition for the prediction call to work.
    # The actual prediction is mocked in vision_utils.predict_food.
    dummy_model = build_model(num_classes=100) # num_classes is a placeholder
    return dummy_model


@st.cache_resource(show_spinner="Starting Logic Engine (Parquet DB Lookup)...")
def load_nutrient_engine():
    """Initialize the Nutrient Engine for instant Parquet lookups."""
    # NOTE: This will fail if Parquet files are not present.
    try:
        return NutrientEngine()
    except FileNotFoundError:
        st.error("FATAL ERROR: Parquet database files not found.")
        st.stop()


# Load model and engine once
MODEL = load_vision_model()
ENGINE = load_nutrient_engine()

# Initialize chat history for chat-like structure
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- 2. Main Logic ---

def process_upload(uploaded_file):
    """Handles the full image processing and assessment pipeline."""
    
    # 1. User Message (Image Upload)
    with st.chat_message("user"):
        st.image(uploaded_file, caption="User Upload", width="stretch")
    
    st.session_state.messages.append({"role": "user", "content": uploaded_file.name, "image": uploaded_file})

    # 2. System Analysis (B5 and Logic Engine)
    with st.chat_message("assistant"):
        
        # Placeholder for analysis feedback
        with st.status("Analyzing image...", expanded=True) as status:
            
            # A. Preprocessing (512x512 resolution)
            st.write("1/3: Pre-processing image (512x512 resolution)...")
            image_tensor = preprocess_image(uploaded_file)

            # B. Vision Prediction (B5 Model)
            st.write("2/3: Running EfficientNet-B5 inference and CV heuristics...")
            prediction_output = predict_food(MODEL, image_tensor)
            
            if "error" in prediction_output:
                st.error(f"Prediction Error: {prediction_output['error']}")
                status.update(label="Analysis Failed", state="error", expanded=False)
                return

            class_id = prediction_output["class_id"]
            visual_stats = prediction_output["visual_stats"]
            top_preds = prediction_output["top_predictions"]
            
            # C. Logic Engine Calculation (Parquet Lookup)
            st.write(f"3/3: Querying Parquet DB for nutrition profile of {class_id}...")
            # This is the call to the "Smart Switch"
            nutrition_result = ENGINE.calculate_nutrition(class_id, visual_stats)

            status.update(label="Analysis Complete", state="complete", expanded=False)

        # 3. Display Result (Chat-like Output)
        
        st.subheader(f"✅ Predicted Food: **{class_id}**")
        st.caption(f"Top-1 Confidence: {top_preds[0][1]:.2f}")
        
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
        
        # Prepare data for a clean table display
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

# Display previous messages (if any)
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.image(message["image"], caption=message["content"], width="stretch")
    elif message["role"] == "assistant":
        # Simplified re-display of the results from the stored dictionary
        result = message["content"]
        with st.chat_message("assistant"):
            st.subheader(f"✅ Predicted Food: **{result.get('Food Code', 'N/A')}**")
            st.markdown("---")
            st.metric(
                label="Total Energy", 
                value=f"{result.get('Energy (kcal)', 0.0):.1f} kcal",
                delta=f"Mass: {result.get('Calculated Mass (g)', 0.0):.1f} g"
            )
            st.info(f"**Logic Path:** {result.get('Logic Path', 'N/A')}")


# Chat Input (Future LLM placeholder)
prompt = st.chat_input("Ask a question or upload an image to begin...")
if prompt:
    # Future LLM/RAG integration point
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Simple placeholder response for text input
    with st.chat_message("assistant"):
        st.write("I am currently configured for image analysis only. Please upload a food image to proceed with dietary assessment.")
    st.session_state.messages.append({"role": "assistant", "content": "Image analysis request."})


# Image Upload Sidebar (The main input for now)
with st.sidebar:
    st.header("Upload Food Image")
    uploaded_file = st.file_uploader(
        "Upload a photo of your meal (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

    # Process the file if uploaded
    if uploaded_file is not None:
        process_upload(uploaded_file)