"""
Enhanced RAG Engine for Current Session Logs.

Searches session inference logs with full metadata extraction.
Supports health condition queries with ingredient-based reasoning.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

class SimpleRAG:
    """
    Enhanced RAG that searches current session logs with full metadata.
    Extracts ingredients, sources, confidence, and serving info for intelligent responses.
    """

    def __init__(self):
        self.session_logs = []  # Store current session logs

    def add_log(self, log_data: Dict[str, Any]):
        """Add a new inference log to current session."""
        self.session_logs.append(log_data)

    def clear_session(self):
        """Clear current session logs."""
        self.session_logs = []

    def search_context(self, query: str, max_results: int = 3) -> str:
        """
        Search current session logs for relevant context with full metadata.

        Args:
            query: User's question
            max_results: Maximum number of logs to include

        Returns:
            Formatted context string with metadata
        """
        if not self.session_logs:
            return "No meal data available in current session."

        # Enhanced keyword matching
        query_lower = query.lower()

        # Categorize query type
        health_keywords = ['diabetes', 'diabetic', 'sugar', 'blood sugar', 'allergy', 'allergic',
                          'gluten', 'lactose', 'vegan', 'vegetarian', 'keto', 'healthy']
        ingredient_keywords = ['ingredient', 'contain', 'made of', 'what\'s in', 'recipe']
        source_keywords = ['source', 'recipe', 'link', 'where', 'reference']
        nutrition_keywords = ['calorie', 'protein', 'carb', 'fat', 'nutrition', 'macro']

        is_health_query = any(kw in query_lower for kw in health_keywords)
        is_ingredient_query = any(kw in query_lower for kw in ingredient_keywords)
        is_source_query = any(kw in query_lower for kw in source_keywords)
        is_nutrition_query = any(kw in query_lower for kw in nutrition_keywords)

        relevant_logs = []

        for log in self.session_logs[-max_results:]:  # Get recent logs
            # Include logs based on query type
            if is_health_query or is_ingredient_query or is_source_query or is_nutrition_query:
                relevant_logs.append(log)
            else:
                # For other questions, check if food names match
                for key, value in log.items():
                    if key.startswith('food_item'):
                        food_name = value.get('name', '').lower()
                        if any(word in food_name for word in query_lower.split()):
                            relevant_logs.append(log)
                            break

        # If no specific matches, include the most recent log
        if not relevant_logs and self.session_logs:
            relevant_logs = [self.session_logs[-1]]

        # Format context based on query type
        return self._format_context(
            relevant_logs,
            include_ingredients=is_health_query or is_ingredient_query,
            include_sources=is_source_query,
            include_full_macros=is_nutrition_query or is_health_query
        )

    def _format_context(
        self,
        logs: List[Dict[str, Any]],
        include_ingredients: bool = False,
        include_sources: bool = False,
        include_full_macros: bool = False
    ) -> str:
        """Format logs into readable context with optional metadata."""
        if not logs:
            return "No relevant meal data found."

        context_parts = []

        for i, log in enumerate(logs, 1):
            total = log.get('total_summary', {})
            calories = total.get('Energy (kcal)', 0)
            protein = total.get('Protein (g)', 0)
            carbs = total.get('Carbohydrate (g)', 0)
            fat = total.get('Fat (g)', 0)

            # Build meal summary
            meal_info = [f"Meal {i}:"]

            # Food items with details
            for key, value in log.items():
                if key.startswith('food_item'):
                    name = value.get('name', 'Unknown')
                    mass = value.get('mass_g', 0)
                    macros = value.get('macros', {})
                    metadata = value.get('metadata', {})

                    food_line = f"  - {name} ({mass}g)"

                    if include_full_macros:
                        food_line += f" | {macros.get('calories', 0):.0f} kcal, "
                        food_line += f"{macros.get('protein', 0):.1f}g protein, "
                        food_line += f"{macros.get('carbs', 0):.1f}g carbs, "
                        food_line += f"{macros.get('fat', 0):.1f}g fat"

                    meal_info.append(food_line)

                    # Add ingredients if requested
                    if include_ingredients:
                        ingredients = metadata.get('ingredients', [])
                        if ingredients:
                            ing_text = ", ".join(ingredients[:15])  # Limit to 15 ingredients
                            meal_info.append(f"    Ingredients: {ing_text}")

                    # Add source if requested
                    if include_sources:
                        source = metadata.get('source', 'N/A')
                        if source != 'N/A' and source != 'Source not available':
                            meal_info.append(f"    Source: {source}")

                    # Add confidence
                    confidence = metadata.get('confidence', 0)
                    meal_info.append(f"    Confidence: {confidence*100:.1f}%")

            # Add total summary
            if include_full_macros:
                meal_info.append(
                    f"  Total: {calories:.0f} kcal | {protein:.1f}g protein | "
                    f"{carbs:.1f}g carbs | {fat:.1f}g fat"
                )
            else:
                meal_info.append(f"  Total: {calories:.0f} kcal, {protein:.1f}g protein")

            context_parts.append("\n".join(meal_info))

        return "\n\n".join(context_parts)

    def get_session_summary(self) -> str:
        """Get a summary of all meals in current session."""
        if not self.session_logs:
            return "No meals analyzed in this session."

        total_calories = sum(
            log.get('total_summary', {}).get('Energy (kcal)', 0)
            for log in self.session_logs
        )

        all_foods = set()
        for log in self.session_logs:
            for key, value in log.items():
                if key.startswith('food_item'):
                    all_foods.add(value.get('name', 'Unknown'))

        return (
            f"Session Summary: {len(self.session_logs)} meals analyzed, "
            f"{total_calories:.0f} total calories, "
            f"Foods: {', '.join(list(all_foods)[:5])}"
        )

    def get_all_ingredients(self) -> List[str]:
        """Extract all unique ingredients from current session."""
        all_ingredients = set()
        for log in self.session_logs:
            for key, value in log.items():
                if key.startswith('food_item'):
                    metadata = value.get('metadata', {})
                    ingredients = metadata.get('ingredients', [])
                    all_ingredients.update(ingredients)
        return sorted(list(all_ingredients))

    def get_health_context(self, condition: str) -> Dict[str, Any]:
        """
        Get health-specific context for conditions like diabetes, allergies, etc.

        Args:
            condition: Health condition (e.g., 'diabetes', 'gluten allergy')

        Returns:
            Dictionary with relevant health information
        """
        if not self.session_logs:
            return {"error": "No meal data available"}

        # Get the most recent log
        latest_log = self.session_logs[-1]
        total = latest_log.get('total_summary', {})

        # Extract all food items with full details
        food_items = []
        all_ingredients = []

        for key, value in latest_log.items():
            if key.startswith('food_item'):
                metadata = value.get('metadata', {})
                ingredients = metadata.get('ingredients', [])
                all_ingredients.extend(ingredients)

                food_items.append({
                    'name': value.get('name', 'Unknown'),
                    'mass_g': value.get('mass_g', 0),
                    'macros': value.get('macros', {}),
                    'ingredients': ingredients,
                    'confidence': metadata.get('confidence', 0)
                })

        return {
            'condition': condition,
            'total_macros': total,
            'food_items': food_items,
            'all_ingredients': list(set(all_ingredients)),
            'total_carbs': total.get('Carbohydrate (g)', 0),
            'total_sugar_estimate': total.get('Carbohydrate (g)', 0) * 0.3,  # Rough estimate
            'meal_count': len([k for k in latest_log.keys() if k.startswith('food_item')])
        }
