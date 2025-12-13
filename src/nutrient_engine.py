"""
Nutrient Engine Module.

This module acts as the logic backend for FoodVisionAI. It connects the
Vision Model's predictions (Class IDs) with the Parquet Database to
compute precise nutritional information based on portion control heuristics.

Implements the "Smart Switch" logic defined in README Section 3:
- Logic Path 1: Discrete Units (Instance Counting)
- Logic Path 2: Container Units (Fill-Level Ratio)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Any
from src import config

class NutrientEngine:
    """
    The central logic controller for nutritional assessment.
    
    Attributes:
        serving_db (pd.DataFrame): 'recipes_servingsize.parquet' data.
        nutrition_db (pd.DataFrame): 'INDB.parquet' data.
    """

    def __init__(self) -> None:
        """
        Initialize the NutrientEngine by loading the Parquet database.
        
        Raises:
            FileNotFoundError: If parquet files are missing in config.PARQUET_DB_DIR.
        """
        self._load_database()

    def _load_database(self) -> None:
        """
        Loads the optimized Parquet files into memory.
        
        Reference: README Section 1 (Data Strategy) - Parquet Conversion.
        """
        # Construct full paths
        serving_path = config.PARQUET_DB_DIR / config.DB_FILES["serving_size"]
        nutrition_path = config.PARQUET_DB_DIR / config.DB_FILES["nutrition"]

        # Load Parquet files (10x faster than Excel)
        # We ensure 'Food Code' is the index for O(1) lookup speeds.
        self.serving_db = pd.read_parquet(serving_path).set_index("recipe_code")
        self.nutrition_db = pd.read_parquet(nutrition_path).set_index("food_code")

    def get_serving_metadata(self, class_id: str) -> Dict[str, Any]:
        """
        Retrieves serving unit metadata for a given class ID.
        
        Args:
            class_id (str): The predicted class ID (e.g., 'ASC171').

        Returns:
            Dict: Contains 'servings_unit' (e.g., 'Piece', 'Bowl') and 'quantity'.
        """
        try:
            # Fetch the row from the dataframe
            row = self.serving_db.loc[class_id]
            return {
                "unit": row["servings_unit"],
                "mass_per_unit": float(row["quantity"])  # grams per unit
            }
        except KeyError:
            # Fallback for unknown classes
            return {"unit": "Standard", "mass_per_unit": 100.0}

    def calculate_nutrition(
        self, 
        class_id: str, 
        visual_stats: Dict[str, float]
    ) -> Dict[str, Union[str, float]]:
        """
        The "Smart Switch" that calculates nutrition based on unit type.
        
        Reference: README Section 3 (The Logic Engine).

        Args:
            class_id (str): The predicted class ID (e.g., 'ASC004').
            visual_stats (Dict): Output from CV heuristics. 
                Must contain 'count' (int) or 'occupancy_ratio' (float).

        Returns:
            Dict: Final calculated nutrition (Energy, Protein, etc.).
        """
        # Step 1: Query Database for Unit Type
        metadata = self.get_serving_metadata(class_id)
        unit_type = metadata["unit"]
        base_mass = metadata["mass_per_unit"]
        
        final_mass = 0.0
        logic_path = ""

        # Step 2: The "Smart Switch"
        
        # Logic Path 1: "Discrete Units"
        # Trigger: Unit is "Piece", "Slice", "Number"
        if unit_type in ["Piece", "Slice", "Number"]:
            logic_path = "Discrete (Count)"
            count = visual_stats.get("count", 1.0)
            # Math: Count * Unit Mass
            final_mass = count * base_mass

        # Logic Path 2: "Container Units"
        # Trigger: Unit is "Bowl", "Cup", "Glass", "Plate"
        elif unit_type in ["Bowl", "Cup", "Glass", "Plate"]:
            logic_path = "Container (Ratio)"
            ratio = visual_stats.get("occupancy_ratio", 1.0)
            
            # Algorithm: Fill-Level Ratio (Occupancy) with Thresholds
            # R > 0.8 -> 1.0x Serving (Full Bowl)
            # R < 0.6 -> 0.5x Serving (Half Bowl)
            # implied: 0.6 <= R <= 0.8 -> Linear or 0.75? 
            # Sticking strictly to README thresholds:
            
            fill_factor = 1.0
            if ratio > 0.8:
                fill_factor = 1.0
            elif ratio < 0.6:
                fill_factor = 0.5
            else:
                # Interpolate or default? README doesn't specify gap.
                # We assume linear interpolation or default to 0.75 for safety.
                # Here we default to actual ratio for precision in the gap.
                fill_factor = ratio 

            final_mass = fill_factor * base_mass
            
        else:
            # Default fallback
            logic_path = "Standard (Weight)"
            final_mass = base_mass

        # Step 3: Retrieve Nutritional Profile
        # The INDB is typically per 100g. We must scale.
        try:
            nutrients = self.nutrition_db.loc[class_id].to_dict()
        except KeyError:
            return {"error": f"Nutritional data not found for {class_id}"}

        # Scale nutrients based on calculated mass
        # Formula: (Value / 100) * Final_Mass
        scale_factor = final_mass / 100.0
        
        result = {
            "Food Code": class_id,
            "Logic Path": logic_path,
            "Detected Unit": unit_type,
            "Calculated Mass (g)": round(final_mass, 1),
            "Energy (kcal)": round(nutrients.get("Energy", 0) * scale_factor, 1),
            "Protein (g)": round(nutrients.get("Protein", 0) * scale_factor, 1),
            "Carbohydrate (g)": round(nutrients.get("Carbohydrate", 0) * scale_factor, 1),
            "Fat (g)": round(nutrients.get("Fat", 0) * scale_factor, 1),
        }

        return result