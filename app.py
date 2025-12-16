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
from PIL import Image

# Local imports
import config
from src.models import build_model
from src.models.loader import load_model
from src.nutrient_engine import NutrientEngine
from src.segmentation import DietaryAssessor
from src.vision_utils import predict_food
from src.utils.file_utils import get_class_names

# --- Setup ---
st.set_page_config(
    page_title="FoodVisionAI - Smart Dietary Assessment",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🥘"
)

@st.cache_resource(show_spinner="Loading AI Models...")
def load_resources():
    # Load Unified Model (trained on clean crops)
    if config.MODEL_LOCAL_PATH.exists():
        model = load_model(config.MODEL_LOCAL_PATH, compile=False)
    else:
        st.warning("⚠️ Unified Model not found. Using dummy model.")
        model = build_model(100) # Dummy

    # Initialize Chat Engine
    from src.chat_engine import ChatEngine
    chat_engine = ChatEngine()

    return model, DietaryAssessor(), NutrientEngine(), get_class_names(), chat_engine

MODEL, ASSESSOR, ENGINE, CLASS_NAMES, CHAT_ENGINE = load_resources()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

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

    # AI Processing
    with st.spinner("🧠 Analyzing meal composition, volume & nutrition..."):

        # 1. Single-Model Inference
        predictions = predict_food(MODEL, ASSESSOR, image_bgr, CLASS_NAMES)

        # 2. Construct Data Payload
        log_payload = {
            "timestamp": datetime.now().isoformat(),
            "total_summary": {
                "Energy (kcal)": 0.0,
                "Protein (g)": 0.0,
                "Carbohydrate (g)": 0.0,
                "Fat (g)": 0.0
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

        # Add to Chat Session & Generate Trivia
        CHAT_ENGINE.add_meal_log(log_payload)
        trivia = CHAT_ENGINE.generate_trivia(log_payload)

    # Store current analysis
    st.session_state.current_analysis = {
        "log_payload": log_payload,
        "meal_display_data": meal_display_data,
        "annotated_img": draw_annotations(image_np, predictions),
        "trivia": trivia,
        "original_img": image_pil
    }

    # Save Assistant Response to History
    st.session_state.messages.append({
        "role": "assistant",
        "type": "report",
        "content": log_payload,
        "image": draw_annotations(image_np, predictions)
    })

    # Enable suggestions
    st.session_state.show_suggestions = True

# --- Helper: Render Macro Dashboard ---
def render_macro_dashboard(log_payload):
    """Render professional macro summary dashboard."""
    total = log_payload['total_summary']
    calories = total['Energy (kcal)']
    protein = total['Protein (g)']
    carbs = total['Carbohydrate (g)']
    fat = total['Fat (g)']

    # Daily recommended values (example: 2000 kcal diet)
    daily_cal = 2000
    daily_protein = 50
    daily_carbs = 275
    daily_fat = 78

    st.markdown("### 📊 Nutritional Summary")

    # Macro cards in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔥 Calories",
            value=f"{calories:.0f}",
            delta=f"{(calories/daily_cal*100):.1f}% of daily"
        )
        st.progress(min(calories/daily_cal, 1.0))

    with col2:
        st.metric(
            label="💪 Protein",
            value=f"{protein:.1f}g",
            delta=f"{(protein/daily_protein*100):.1f}% of daily"
        )
        st.progress(min(protein/daily_protein, 1.0))

    with col3:
        st.metric(
            label="🌾 Carbs",
            value=f"{carbs:.1f}g",
            delta=f"{(carbs/daily_carbs*100):.1f}% of daily"
        )
        st.progress(min(carbs/daily_carbs, 1.0))

    with col4:
        st.metric(
            label="🥑 Fat",
            value=f"{fat:.1f}g",
            delta=f"{(fat/daily_fat*100):.1f}% of daily"
        )
        st.progress(min(fat/daily_fat, 1.0))

    # Macro distribution pie chart (text-based)
    total_macros = protein + carbs + fat
    if total_macros > 0:
        st.markdown("**Macro Distribution:**")
        dist_col1, dist_col2, dist_col3 = st.columns(3)
        with dist_col1:
            st.info(f"💪 Protein: {(protein/total_macros*100):.1f}%")
        with dist_col2:
            st.warning(f"🌾 Carbs: {(carbs/total_macros*100):.1f}%")
        with dist_col3:
            st.error(f"🥑 Fat: {(fat/total_macros*100):.1f}%")


# --- Main UI Layout ---
st.title("🥘 FoodVisionAI")
st.caption("AI-Powered Dietary Assessment & Nutrition Analysis")

# Sidebar
with st.sidebar:
    st.header("📤 Upload")
    uploaded_file = st.file_uploader("Upload Meal Photo", type=["jpg", "png", "jpeg"])

    st.markdown("---")

    st.header("⚙️ Actions")
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_analysis = None
        CHAT_ENGINE.clear_session()
        st.rerun()

    st.markdown("---")

    # System Status
    st.header("📡 System Status")
    model_status = "✅ Ready" if config.MODEL_LOCAL_PATH.exists() else "❌ Not Found"
    chat_status = "✅ Ready" if CHAT_ENGINE.chat_enabled else "❌ Disabled"

    st.success(f"**Vision Model:** {model_status}")
    st.success(f"**Chat Engine:** {chat_status}")

    # Session Summary
    if CHAT_ENGINE.chat_enabled:
        session_summary = CHAT_ENGINE.get_session_summary()
        if "No meals" not in session_summary:
            st.markdown("---")
            st.header("📈 Session Stats")
            st.info(session_summary)

# Process Upload
if uploaded_file:
    # Debounce: Only process if it's a new file (avoid loop on re-render)
    if not st.session_state.messages or st.session_state.messages[-1].get("type") != "report":
        process_upload(uploaded_file)

# Main Content Area
if st.session_state.current_analysis:
    analysis = st.session_state.current_analysis

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "💬 Chat Assistant", "📜 History"])

    with tab1:
        # Analysis Tab - Professional Dashboard
        col_img, col_data = st.columns([1, 1])

        with col_img:
            st.markdown("#### 📸 Detected Food Items")
            st.image(analysis["annotated_img"], use_container_width=True)

        with col_data:
            st.markdown("#### 🔍 Detection Details")
            for i, data in enumerate(analysis["meal_display_data"]):
                with st.expander(f"**{i+1}. {data['Food Name']}**", expanded=(i == 0)):
                    st.write(f"**Mass:** {data['Calculated Mass (g)']}g")
                    st.write(f"**Code:** {data['Food Code']}")

                    # Macros in compact format
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Cal", f"{data['Energy (kcal)']:.0f}")
                    m2.metric("Prot", f"{data['Protein (g)']:.1f}g")
                    m3.metric("Carb", f"{data['Carbohydrate (g)']:.1f}g")
                    m4.metric("Fat", f"{data['Fat (g)']:.1f}g")

                    # Ingredients
                    ing_list = data.get('Ingredients', [])
                    if ing_list:
                        st.markdown("**Ingredients:**")
                        st.caption(", ".join(ing_list[:12]))

                    # Source
                    if data['Source'] not in ["N/A", "Source not available"]:
                        st.markdown(f"**Source:** [View Recipe]({data['Source']})")

        st.markdown("---")

        # Macro Dashboard
        render_macro_dashboard(analysis["log_payload"])

        # Trivia Section
        if analysis.get("trivia"):
            st.markdown("---")
            st.info(f"🧠 **Did you know?** {analysis['trivia']}")

    with tab2:
        # Chat Tab - Interactive Q&A
        st.markdown("### 💬 Ask About Your Meal")

        # Suggested Questions
        if st.session_state.show_suggestions:
            st.markdown("**💡 Suggested Questions:**")
            suggestions = [
                "What are the ingredients in this meal?",
                "I have diabetes, can I consume this?",
                "Is this meal healthy?",
                "Show me the sources for these recipes",
                "What's the protein content?",
                "Can I eat this on a keto diet?"
            ]

            # Display as clickable buttons in 2 columns
            col1, col2 = st.columns(2)
            for idx, suggestion in enumerate(suggestions):
                with col1 if idx % 2 == 0 else col2:
                    if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                        # Add user message
                        st.session_state.messages.append({
                            "role": "user",
                            "type": "text",
                            "content": suggestion
                        })

                        # Generate response immediately
                        response = CHAT_ENGINE.answer_question(suggestion)

                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "text",
                            "content": response
                        })

                        st.session_state.show_suggestions = False
                        st.rerun()

            st.markdown("---")

        # Chat History
        for msg in st.session_state.messages:
            if msg["type"] == "text":
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Chat Input
        if prompt := st.chat_input("Ask me anything about your meal..."):
            st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
            st.session_state.show_suggestions = False

            # Generate Response
            with st.spinner("🤔 Thinking..."):
                response = CHAT_ENGINE.answer_question(prompt)

            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": response
            })
            st.rerun()

    with tab3:
        # History Tab - All Past Analyses
        st.markdown("### 📜 Analysis History")

        if len(st.session_state.messages) == 0:
            st.info("No analysis history yet. Upload a meal photo to get started!")
        else:
            # Show all past reports
            report_count = 0
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant" and msg["type"] == "report":
                    report_count += 1
                    data = msg["content"]

                    with st.expander(
                        f"**Meal {report_count}** - {data['total_summary']['Energy (kcal)']:.0f} kcal",
                        expanded=False
                    ):
                        if "image" in msg:
                            st.image(msg["image"], width=300)

                        st.write(f"**Timestamp:** {data.get('timestamp', 'N/A')}")

                        # Food items
                        for k, v in data.items():
                            if k.startswith("food_item"):
                                st.caption(f"• {v['name']} ({v['mass_g']}g)")

                        # Macros
                        total = data['total_summary']
                        st.write(
                            f"**Macros:** {total['Protein (g)']:.1f}g protein, "
                            f"{total['Carbohydrate (g)']:.1f}g carbs, "
                            f"{total['Fat (g)']:.1f}g fat"
                        )

else:
    # Welcome Screen
    st.markdown("---")
    st.markdown("### 👋 Welcome to FoodVisionAI!")
    st.info(
        "📸 **Get Started:** Upload a meal photo using the sidebar to analyze its "
        "nutritional content, ingredients, and get personalized health insights!"
    )

    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔍 AI Detection")
        st.write("Advanced computer vision to identify food items")
    with col2:
        st.markdown("#### 📊 Nutrition Analysis")
        st.write("Detailed macro breakdown and calorie tracking")
    with col3:
        st.markdown("#### 💬 Smart Chat")
        st.write("Ask health questions with ingredient-based reasoning")
