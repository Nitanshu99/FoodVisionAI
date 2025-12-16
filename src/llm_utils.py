"""
LLM Utilities for Offline Qwen2.5-0.5B Model.

Provides a simple wrapper around llama-cpp-python for:
- Food trivia generation
- Nutritional Q&A responses
- Context-aware chat
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama
from src import config

class QwenLLM:
    """
    Lightweight wrapper for Qwen2.5-0.5B GGUF model.
    Handles model loading, prompt formatting, and inference.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model with error handling."""
        try:
            if not config.LLM_MODEL_PATH.exists():
                print(f"⚠️ LLM model not found at {config.LLM_MODEL_PATH}")
                print("💡 Run: python download_llm.py")
                return
            
            print(f"🧠 Loading Qwen2.5-0.5B from {config.LLM_MODEL_PATH}...")
            
            # Conservative settings for 0.5B model
            self.model = Llama(
                model_path=str(config.LLM_MODEL_PATH),
                n_ctx=2048,        # Context window
                n_threads=4,       # CPU threads
                verbose=False,     # Reduce logging
                n_gpu_layers=0     # CPU-only for compatibility
            )
            
            self.model_loaded = True
            print("✅ LLM loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            print("💡 Chat features will be disabled")
            self.model_loaded = False
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model_loaded and self.model is not None
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum response length
            temperature: Creativity (0.0-1.0)
            
        Returns:
            Generated text response
        """
        if not self.is_available():
            return "Chat unavailable (model not loaded)"
        
        try:
            # Format prompt for Qwen2.5-Instruct
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "\n\n"],  # Stop tokens
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "Sorry, I couldn't generate a response."
    
    def generate_food_trivia(self, food_items: list) -> str:
        """Generate interesting trivia about detected food items."""
        if not food_items:
            return self._get_fallback_trivia()

        food_names = [item.get('name', 'Unknown') for item in food_items]
        food_list = ", ".join(food_names[:3])  # Limit to 3 items

        prompt = f"""Generate a short, interesting food trivia about: {food_list}

Keep it under 50 words and focus on nutrition, culture, or cooking facts."""

        trivia = self.generate_response(prompt, max_tokens=80, temperature=0.8)

        # Fallback if generation fails
        if not trivia or "unavailable" in trivia.lower() or len(trivia) < 10:
            return self._get_fallback_trivia(food_names[0] if food_names else None)

        return trivia

    def _get_fallback_trivia(self, food_name: Optional[str] = None) -> str:
        """Get fallback trivia when LLM fails."""
        fallback_trivias = [
            "Indian cuisine is known for its rich use of spices, which have antioxidant properties!",
            "Protein helps build muscle, while carbs provide energy for your daily activities.",
            "Eating a balanced meal with all macronutrients supports overall health and wellness.",
            "Traditional Indian meals often include a balance of proteins, carbs, and healthy fats.",
            "Spices like turmeric and cumin have anti-inflammatory benefits beyond just flavor!",
        ]

        if food_name:
            food_specific = {
                'dosa': "Dosa is a fermented food, which means it contains probiotics good for gut health!",
                'rice': "Rice is a staple for over half the world's population and provides quick energy!",
                'dal': "Dal (lentils) is an excellent plant-based protein source with high fiber content!",
                'paneer': "Paneer is rich in calcium and protein, making it great for bone health!",
                'roti': "Whole wheat roti provides complex carbs and fiber for sustained energy!",
            }
            for key, trivia in food_specific.items():
                if key in food_name.lower():
                    return trivia

        import random
        return random.choice(fallback_trivias)

    def answer_health_question(self, question: str, health_context: Dict[str, Any]) -> str:
        """
        Answer health-related questions with ingredient-based reasoning.

        Args:
            question: User's health question
            health_context: Context from RAG with ingredients and macros

        Returns:
            Detailed answer with reasoning
        """
        condition = health_context.get('condition', 'general health')
        total_carbs = health_context.get('total_carbs', 0)
        food_items = health_context.get('food_items', [])
        all_ingredients = health_context.get('all_ingredients', [])

        # Build detailed context
        foods_detail = []
        for item in food_items:
            name = item.get('name', 'Unknown')
            carbs = item.get('macros', {}).get('carbs', 0)
            ingredients = item.get('ingredients', [])
            ing_text = ", ".join(ingredients[:8]) if ingredients else "ingredients not listed"
            foods_detail.append(f"- {name}: {carbs:.1f}g carbs (contains: {ing_text})")

        foods_text = "\n".join(foods_detail)

        prompt = f"""You are a nutrition advisor. Analyze this meal for someone with {condition}.

Meal Details:
{foods_text}

Total Carbohydrates: {total_carbs:.1f}g
All Ingredients: {", ".join(all_ingredients[:20])}

Question: {question}

Provide a helpful answer with:
1. Direct answer (yes/no/moderate)
2. Reasoning based on ingredients and macros
3. Specific concerns or benefits
Keep it under 100 words."""

        return self.generate_response(prompt, max_tokens=150, temperature=0.6)

    def answer_nutrition_question(self, question: str, context: Dict[str, Any]) -> str:
        """Answer questions about nutrition using inference log context."""

        # Extract key info from context
        total_calories = context.get('total_summary', {}).get('Energy (kcal)', 0)
        food_items = []

        for key, value in context.items():
            if key.startswith('food_item'):
                food_items.append(f"{value.get('name', 'Unknown')} ({value.get('mass_g', 0)}g)")

        foods_text = ", ".join(food_items) if food_items else "No specific foods"

        prompt = f"""Based on this meal analysis:
- Total calories: {total_calories:.0f} kcal
- Foods detected: {foods_text}

Question: {question}

Provide a helpful, brief answer about the nutrition or food items."""

        return self.generate_response(prompt, max_tokens=120, temperature=0.6)

# Global instance (singleton pattern)
_llm_instance = None

def get_llm() -> QwenLLM:
    """Get the global LLM instance (lazy loading)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = QwenLLM()
    return _llm_instance
