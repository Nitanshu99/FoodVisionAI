"""
Nutrient Engine Module.

This module acts as the logic backend for FoodVisionAI. It connects the
Vision Model's predictions (Class IDs) with the Parquet Database to
compute precise nutritional information based on portion control heuristics.
"""

import pandas as pd
from typing import Dict, Union, Any
from src import config

class NutrientEngine:
    """
    The central logic controller for nutritional assessment.
    
    Attributes:
        serving_db (pd.DataFrame): 'recipes_servingsize.parquet' data.
        nutrition_db (pd.DataFrame): 'INDB.parquet' data.
        names_db (pd.DataFrame): 'recipes_names.parquet' data.
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
        """
        # Construct full paths
        serving_path = config.PARQUET_DB_DIR / config.DB_FILES["serving_size"]
        nutrition_path = config.PARQUET_DB_DIR / config.DB_FILES["nutrition"]
        
        # We assume recipes_names.parquet exists (from recipes_names.xlsx)
        # You may need to ensure 'recipes_names.parquet' is in DB_FILES in config.py
        names_path = config.PARQUET_DB_DIR / "recipes_names.parquet" 

        # Load Parquet files
        # 1. Serving Size DB: Indexed by 'recipe_code' (e.g., ASC001)
        self.serving_db = pd.read_parquet(serving_path)
        # Ensure index is set if not already
        if "recipe_code" in self.serving_db.columns:
            self.serving_db = self.serving_db.set_index("recipe_code")

        # 2. Nutrition DB (INDB): Indexed by 'food_code' (e.g., ASC001)
        self.nutrition_db = pd.read_parquet(nutrition_path)
        # Ensure index is set
        if "food_code" in self.nutrition_db.columns:
            self.nutrition_db = self.nutrition_db.set_index("food_code")
            
        # 3. Names DB: Indexed by 'recipe_code' (e.g., ASC001)
        if names_path.exists():
            self.names_db = pd.read_parquet(names_path)
            if "recipe_code" in self.names_db.columns:
                self.names_db = self.names_db.set_index("recipe_code")
        else:
            self.names_db = pd.DataFrame() # Fallback empty

    def get_food_name(self, class_id: str) -> str:
        """Retrieves the human-readable name for a given class ID."""
        try:
            # Look up in recipes_names
            return self.names_db.loc[class_id]["recipe_name"]
        except (KeyError, AttributeError):
            # Fallback if name not found
            return class_id

    def get_serving_metadata(self, class_id: str) -> Dict[str, Any]:
        """
        Retrieves serving unit metadata for a given class ID.
        """
        try:
            row = self.serving_db.loc[class_id]
            # Handle duplicates if index is not unique
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
                
            return {
                "unit": row["servings_unit"],
                "mass_per_unit": float(row["quantity"])  # grams per unit
            }
        except KeyError:
            return {"unit": "Standard", "mass_per_unit": 100.0}

    def calculate_nutrition(
        self, 
        class_id: str, 
        visual_stats: Dict[str, float]
    ) -> Dict[str, Union[str, float]]:
        """
        The "Smart Switch" that calculates nutrition based on unit type.
        """
        # Step 1: Query Database for Unit Type
        metadata = self.get_serving_metadata(class_id)
        unit_type = metadata["unit"]
        base_mass = metadata["mass_per_unit"]
        
        final_mass = 0.0
        logic_path = ""

        # Step 2: The "Smart Switch" Logic
        if unit_type in ["Piece", "Slice", "Number", "nos.", "piece"]:
            logic_path = "Discrete (Count)"
            count = visual_stats.get("count", 1.0)
            final_mass = count * base_mass

        elif unit_type in ["Bowl", "Cup", "Glass", "Plate", "cup", "bowl", "glass"]:
            logic_path = "Container (Ratio)"
            ratio = visual_stats.get("occupancy_ratio", 1.0)
            
            fill_factor = 1.0
            if ratio > 0.8:
                fill_factor = 1.0
            elif ratio < 0.6:
                fill_factor = 0.5
            else:
                fill_factor = ratio 

            final_mass = fill_factor * base_mass
            
        else:
            logic_path = "Standard (Weight)"
            final_mass = base_mass

        # Step 3: Retrieve Nutritional Profile
        # Correcting the Key Mismatch here
        # INDB uses: energy_kcal, protein_g, carb_g, fat_g
        try:
            nutrients_row = self.nutrition_db.loc[class_id]
            # Handle potential duplicate rows
            if isinstance(nutrients_row, pd.DataFrame):
                nutrients_row = nutrients_row.iloc[0]
            nutrients = nutrients_row.to_dict()
        except KeyError:
            return {
                "Food Code": class_id,
                "Food Name": "Unknown (Not in DB)",
                "Error": "Data Missing"
            }

        # Scale nutrients (INDB is per 100g)
        scale_factor = final_mass / 100.0
        
        # Look up readable name
        food_name = self.get_food_name(class_id)
        
        # Build Result with Correct Key Mapping
        result = {
            "Food Code": class_id,
            "Food Name": food_name,
            "Logic Path": logic_path,
            "Detected Unit": unit_type,
            "Calculated Mass (g)": round(final_mass, 1),
            
            # Map INDB keys (lowercase) to App keys (Display)
            "Energy (kcal)": round(nutrients.get("energy_kcal", 0) * scale_factor, 1),
            "Protein (g)": round(nutrients.get("protein_g", 0) * scale_factor, 1),
            "Carbohydrate (g)": round(nutrients.get("carb_g", 0) * scale_factor, 1),
            "Fat (g)": round(nutrients.get("fat_g", 0) * scale_factor, 1),
        }

        return result