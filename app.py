"""
Streamlit Application Frontend (Final - Dual Model & Chat UI).

Features:
- Dual Model Inference (Global vs Local).
- Chat Interface (User Image -> AI Report -> User Text).
- Structured JSON Logging (food_item_n format) for RAG.
"""

import os
import json
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import keras
from PIL import Image

# Local imports
from src import config
from src.vision_model import build_model
from src.nutrient_engine import NutrientEngine
from src.segmentation import DietaryAssessor
from src.vision_utils import predict_food, get_class_names
from src.augmentation import RandomGaussianBlur

# --- Setup ---
st.set_page_config(
    page_title="FoodVisionAI - Smart Dietary Assessment", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading AI Models...")
def load_resources():
    # 1. Load GLOBAL Model (Context Aware)
    if config.MODEL_GLOBAL_PATH.exists():
        model_global = keras.models.load_model(config.MODEL_GLOBAL_PATH, custom_objects={"RandomGaussianBlur": RandomGaussianBlur}, compile=False)
    else:
        st.warning("⚠️ Global Model not found. Using dummy model.")
        model_global = build_model(100) # Dummy

    # 2. Load LOCAL Model (Crop Specialist)
    if config.MODEL_LOCAL_PATH.exists():
        model_local = keras.models.load_model(config.MODEL_LOCAL_PATH, custom_objects={"RandomGaussianBlur": RandomGaussianBlur}, compile=False)
    else:
        # Fallback to global if local training isn't done
        model_local = model_global 

    return model_global, model_local, DietaryAssessor(), NutrientEngine(), get_class_names()

MODEL_GLOBAL, MODEL_LOCAL, ASSESSOR, ENGINE, CLASS_NAMES = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper Functions ---

def draw_annotations(image_np, results):
    """Draws boxes/masks/labels on the image."""
    annotated = image_np.copy()
    green_screen = np.zeros_like(annotated)
    green_screen[:, :, 1] = 255 
    
    for item in results:
        stats = item["visual_stats"]
        if "bbox" in stats:
            x1, y1, x2, y2 = stats["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Short label
            label = f"{item['class_id']}"
            cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if "mask" in stats and stats["mask"] is not None:
            mask = stats["mask"]
            mask_indices = mask > 0
            annotated[mask_indices] = cv2.addWeighted(
                annotated[mask_indices], 0.7, green_screen[mask_indices], 0.3, 0
            )
    return annotated

def save_log(inference_payload):
    """Saves the structured JSON log to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = config.LOGS_DIR / f"log_{timestamp}.json"
    try:
        with open(filename, "w") as f:
            json.dump(inference_payload, f, indent=4)
        return str(filename)
    except Exception as e:
        print(f"Logging error: {e}")
        return None

def process_upload(uploaded_file):
    # Display User Image in Chat
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.asarray(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Add User Message to State
    st.session_state.messages.append({"role": "user", "type": "image", "content": image_pil})
    with st.chat_message("user"):
        st.image(image_pil, caption="Uploaded Meal", width=400)

    # AI Processing
    with st.chat_message("assistant"):
        with st.spinner("🧠 Analyzing texture, volume & nutrition..."):
            
            # 1. Dual-Model Inference
            predictions = predict_food(MODEL_GLOBAL, MODEL_LOCAL, ASSESSOR, image_bgr, CLASS_NAMES)
            
            # 2. Construct Data Payload
            log_payload = {
                "timestamp": datetime.now().isoformat(),
                "total_summary": {
                    "Energy (kcal)": 0.0, "Protein (g)": 0.0, "Carbohydrate (g)": 0.0, "Fat (g)": 0.0
                }
            }
            
            meal_display_data = []

            for i, item in enumerate(predictions):
                # Calculate Nutrition (Enriched with Ingredients/Units)
                nutri = ENGINE.calculate_nutrition(item["class_id"], item["visual_stats"])
                
                # Add to Log Payload (food_item_1, food_item_2...)
                key = f"food_item_{i+1}"
                log_payload[key] = {
                    "name": nutri["Food Name"],
                    "code": nutri["Food Code"],
                    "mass_g": nutri["Calculated Mass (g)"],
                    "macros": {
                        "calories": nutri["Energy (kcal)"],
                        "protein": nutri["Protein (g)"],
                        "carbs": nutri["Carbohydrate (g)"],
                        "fat": nutri["Fat (g)"]
                    },
                    "metadata": {
                        "ingredients": nutri["Ingredients"],
                        "serving_info": nutri["Serving Metadata"],
                        "source": nutri["Source"],
                        "confidence": item["top_predictions"][0][1],
                        "model_used": item["crop_type"]
                    }
                }
                
                # Add to Display List
                meal_display_data.append(nutri)
                
                # Aggregate Totals
                log_payload["total_summary"]["Energy (kcal)"] += nutri["Energy (kcal)"]
                log_payload["total_summary"]["Protein (g)"] += nutri["Protein (g)"]
                log_payload["total_summary"]["Carbohydrate (g)"] += nutri["Carbohydrate (g)"]
                log_payload["total_summary"]["Fat (g)"] += nutri["Fat (g)"]

            # Save Log
            log_path = save_log(log_payload)
            print(f"Log saved: {log_path}")

            # 3. Render Response
            annotated_img = draw_annotations(image_np, predictions)
            st.image(annotated_img, caption="Visual Segmentation", width=400)
            
            # Summary Banner
            st.markdown(f"### 📊 Total: {log_payload['total_summary']['Energy (kcal)']:.0f} kcal")
            
            # Render Item Details (Collapsible)
            for i, data in enumerate(meal_display_data):
                with st.expander(f"Item {i+1}: {data['Food Name']}", expanded=(i==0)):
                    st.write(f"**Mass:** {data['Calculated Mass (g)']}g")
                    
                    if data['Source'] != "N/A":
                        st.markdown(f"**Source:** [Recipe Link]({data['Source']})")
                    
                    # Ingredients Tag List
                    ing_list = data.get('Ingredients', [])
                    if ing_list:
                        st.caption(f"**Ingredients:** {', '.join(ing_list[:8])}...")
                    
                    # Macro Columns
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Cal", f"{data['Energy (kcal)']:.0f}")
                    c2.metric("Prot", f"{data['Protein (g)']:.1f}g")
                    c3.metric("Carb", f"{data['Carbohydrate (g)']:.1f}g")
                    c4.metric("Fat", f"{data['Fat (g)']:.1f}g")

    # Save Assistant Response to History
    st.session_state.messages.append({
        "role": "assistant", 
        "type": "report", 
        "content": log_payload,
        "image": annotated_img
    })

# --- Main UI Layout ---
st.title("FoodVisionAI 🥘")
st.caption("Offline Monocular Dietary Assessment System")

# Sidebar
with st.sidebar:
    st.header("Actions")
    uploaded_file = st.file_uploader("Upload Meal Photo", type=["jpg","png","jpeg"])
    
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.info(f"**Status:**\n- Global Model: {'✅' if config.MODEL_GLOBAL_PATH.exists() else '❌'}\n- Local Model: {'✅' if config.MODEL_LOCAL_PATH.exists() else '❌'}")

# Process Upload
if uploaded_file:
    # Debounce: Only process if it's a new file (avoid loop on re-render)
    # Check if the last message was a report generated from *this* upload? 
    # Simplified check: if last msg is NOT report, process.
    if not st.session_state.messages or st.session_state.messages[-1].get("type") != "report":
         process_upload(uploaded_file)

# Chat History Render
for msg in st.session_state.messages:
    if msg["role"] == "user":
        if msg["type"] == "text":
            st.chat_message("user").write(msg["content"])
        elif msg["type"] == "image":
            st.chat_message("user").image(msg["content"], width=300)
            
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "image" in msg:
                st.image(msg["image"], width=300)
            
            data = msg["content"]
            # Render a mini summary for history view
            if isinstance(data, dict) and "total_summary" in data:
                st.write(f"**Total Calories:** {data['total_summary']['Energy (kcal)']:.0f} kcal")
                for k, v in data.items():
                    if k.startswith("food_item"):
                        st.caption(f"• {v['name']} ({v['mass_g']}g)")
            else:
                st.write(data)

# User Input (Future RAG)
if prompt := st.chat_input("Ask about your meal (e.g., 'Is this healthy?')..."):
    st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Placeholder for LLM Response
    response = "I have logged your question and the food context. (LLM Integration Pending)"
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})
    st.chat_message("assistant").write(response)