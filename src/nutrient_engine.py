"""
Nutrient Engine Module.

The logic core that converts Visual Data (Class ID + Geometry) into
Nutritional Data (Mass + Macros + Ingredients) using Density and Volumetry.

Updated to match the specific schema of the provided Parquet files.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Union, Any, List
import config
from src.utils.data_utils import safe_get, clean_string_list

class NutrientEngine:
    """
    Orchestrates the conversion from 'What we see' (Vision) to 'What we eat' (Nutrition).
    """

    def __init__(self) -> None:
        self._load_database()

    def _load_database(self) -> None:
        """
        Loads INDB, Serving Sizes, Units, Recipes, and Links.
        """
        # Paths
        serving_path = config.PARQUET_DB_DIR / config.DB_FILES["serving_size"]
        nutrition_path = config.PARQUET_DB_DIR / config.DB_FILES["nutrition"]
        units_path = config.PARQUET_DB_DIR / config.DB_FILES["units"]
        links_path = config.PARQUET_DB_DIR / config.DB_FILES["links"]
        recipes_path = config.PARQUET_DB_DIR / config.DB_FILES["recipes"]
        names_path = config.PARQUET_DB_DIR / "recipes_names.parquet"

        # 1. Load Serving DB (recipes_servingsize.parquet)
        # Key: 'recipe_code'
        if serving_path.exists():
            self.serving_db = pd.read_parquet(serving_path)
            if "recipe_code" in self.serving_db.columns:
                self.serving_db = self.serving_db.set_index("recipe_code")
        else:
            self.serving_db = pd.DataFrame()

        # 2. Load Nutrition DB (INDB.parquet)
        # Key: 'food_code'
        if nutrition_path.exists():
            self.nutrition_db = pd.read_parquet(nutrition_path)
            if "food_code" in self.nutrition_db.columns:
                self.nutrition_db = self.nutrition_db.set_index("food_code")
        else:
            self.nutrition_db = pd.DataFrame()

        # 3. Load Units DB (Units.parquet)
        # Columns: ['Food items', 'Units', 'Units.1', ...]
        if units_path.exists():
            self.units_db = pd.read_parquet(units_path)
        else:
            self.units_db = pd.DataFrame()

        # 4. Load Links DB (recipe_links.parquet)
        # Columns: ['Food Names', 'Food Code', 'Link']
        if links_path.exists():
            self.links_db = pd.read_parquet(links_path)
            if "Food Code" in self.links_db.columns:
                self.links_db = self.links_db.set_index("Food Code")
        else:
            self.links_db = pd.DataFrame()
            
        # 5. Load Recipes DB (recipes.parquet - Ingredients)
        # Key: 'recipe_code', Col: 'ingredient_name_org'
        if recipes_path.exists():
            self.recipes_db = pd.read_parquet(recipes_path)
        else:
            self.recipes_db = pd.DataFrame()

        # 6. Load Names DB (recipes_names.parquet)
        # Key: 'recipe_code', Col: 'recipe_name'
        if names_path.exists():
            self.names_db = pd.read_parquet(names_path)
            if "recipe_code" in self.names_db.columns:
                self.names_db = self.names_db.set_index("recipe_code")
        else:
            self.names_db = pd.DataFrame()

    def get_food_name(self, class_id: str) -> str:
        """Retrieves recipe name from recipes_names.parquet."""
        try:
            return self.names_db.loc[class_id]["recipe_name"]
        except (KeyError, AttributeError):
            # Fallback to INDB if not in recipes_names
            try:
                return self.nutrition_db.loc[class_id]["food_name"]
            except:
                return class_id

    def get_source_link(self, class_id: str) -> str:
        """Retrieves URL from recipe_links.parquet using 'Food Code'."""
        if self.links_db.empty: return "N/A"
        try:
            # The index is 'Food Code'
            return self.links_db.loc[class_id]["Link"]
        except (KeyError, AttributeError):
            return "Source not available"

    def get_ingredients(self, class_id: str) -> List[str]:
        """Fetches ingredients from recipes.parquet where recipe_code matches."""
        if self.recipes_db.empty: return []
        try:
            mask = self.recipes_db['recipe_code'] == class_id
            ingredients = self.recipes_db.loc[mask, 'ingredient_name_org'].tolist()
            # Clean up: remove NaNs and duplicates using utility function
            return clean_string_list(ingredients)
        except Exception:
            return ["Ingredients data not found"]

    def get_density(self, food_name: str) -> float:
        """
        Estimates density from Units.parquet.
        Schema: 'Food items', 'Units' (Type), 'Units.1' (Weight in g).
        """
        if self.units_db.empty: return 0.85

        try:
            # Fuzzy search on 'Food items'
            # We use column name 'Food items' directly
            matches = self.units_db[self.units_db['Food items'].astype(str).str.contains(food_name, case=False, na=False)]
            
            if not matches.empty:
                # Look for 'cup' or 'bowl' in the 'Units' column to get volume-based weight
                cup_row = matches[matches['Units'].astype(str).str.contains("cup|bowl", case=False, na=False)]
                
                if not cup_row.empty:
                    # 'Units.1' contains the weight (e.g., "135g" or just 135)
                    weight_val = cup_row.iloc[0]['Units.1']
                    weight_str = str(weight_val)
                    
                    # Extract number
                    weight_g = float(re.findall(r"[\d\.]+", weight_str)[0])
                    return weight_g / 240.0 # Standardizing on ~240ml cup
        except Exception:
            pass
            
        return 0.85 # Default 'Food' density

    def calculate_nutrition(
        self, 
        class_id: str, 
        visual_stats: Dict[str, Any]
    ) -> Dict[str, Union[str, float, List[str]]]:
        """
        Calculates full nutritional profile including ingredients and serving metadata.
        """
        food_name = self.get_food_name(class_id)
        source_link = self.get_source_link(class_id)
        ingredients_list = self.get_ingredients(class_id)
        
        # 1. Determine Calculation Strategy & Metadata
        unit_type = "Standard"
        std_serving_size = "N/A"

        try:
            # Lookup in recipes_servingsize
            if class_id in self.serving_db.index:
                row = self.serving_db.loc[class_id]
                unit_type = safe_get(row, "servings_unit", "Standard")
                size_val = safe_get(row, "size_of_servings", "?")
                std_serving_size = f"{size_val} {unit_type}"
        except Exception:
            pass

        final_mass_g = 0.0
        logic_path = ""
        
        # --- STRATEGY A: CONTAINER (LIQUIDS) ---
        if any(x in str(unit_type).lower() for x in ["bowl", "cup", "glass", "katori"]):
            logic_path = "Volumetric (Container)"
            container_vol_ml = 200.0 
            fill_ratio = visual_stats.get("occupancy_ratio", 1.0)
            
            if fill_ratio > 0.8: fill_ratio = 1.0
            elif fill_ratio < 0.4: fill_ratio = 0.5
            
            volume_ml = container_vol_ml * fill_ratio
            final_mass_g = volume_ml * 1.0 # Approx density for liquids
            
        # --- STRATEGY B: DISCRETE SOLIDS (BREADS/PIECES) ---
        elif any(x in str(unit_type).lower() for x in ["piece", "slice", "no.", "number"]):
            logic_path = "Geometric (Cylinder)"
            area_cm2 = visual_stats.get("area_cm2", 0)
            
            if area_cm2 < 10:
                # If detected area is tiny, rely on database serving weight if available
                # But recipes_servingsize usually has size/unit, not gram weight directly.
                # We fallback to a default unless we can link to Units.
                final_mass_g = 80.0 # Default piece weight
                logic_path = "Fallback (Standard Piece)"
            else:
                thickness_cm = 0.5
                volume_ml = area_cm2 * thickness_cm
                density = self.get_density(food_name)
                final_mass_g = volume_ml * density

        # --- STRATEGY C: MOUNDS (RICE/POHA) ---
        else:
            logic_path = "Geometric (Spherical Cap)"
            area_cm2 = visual_stats.get("area_cm2", 0)
            
            if area_cm2 > 10:
                radius_cm = np.sqrt(area_cm2 / np.pi)
                height_cm = radius_cm * 0.6
                volume_ml = (np.pi * height_cm / 6) * (3 * radius_cm**2 + height_cm**2)
                density = self.get_density(food_name)
                final_mass_g = volume_ml * density
            else:
                final_mass_g = 100.0
                logic_path = "Fallback (Default)"

        # 2. Nutrition Lookup (INDB)
        # Columns: energy_kcal, protein_g, carb_g, fat_g
        nutrients = {}
        try:
            if class_id in self.nutrition_db.index:
                nutrients_row = self.nutrition_db.loc[class_id]
                nutrients = {
                    "energy_kcal": float(safe_get(nutrients_row, "energy_kcal", 0)),
                    "protein_g": float(safe_get(nutrients_row, "protein_g", 0)),
                    "carb_g": float(safe_get(nutrients_row, "carb_g", 0)),
                    "fat_g": float(safe_get(nutrients_row, "fat_g", 0))
                }
        except Exception as e:
            # print(f"Nutrient lookup error: {e}")
            pass

        scale = final_mass_g / 100.0
        
        # 3. Construct Final Object
        return {
            "Food Code": class_id,
            "Food Name": food_name,
            "Source": source_link,
            "Ingredients": ingredients_list,
            "Serving Metadata": {
                "Unit": str(unit_type),
                "Standard Size": str(std_serving_size)
            },
            "Logic Path": logic_path,
            "Calculated Mass (g)": round(final_mass_g, 1),
            "Energy (kcal)": round(nutrients.get("energy_kcal", 0) * scale, 1),
            "Protein (g)": round(nutrients.get("protein_g", 0) * scale, 1),
            "Carbohydrate (g)": round(nutrients.get("carb_g", 0) * scale, 1),
            "Fat (g)": round(nutrients.get("fat_g", 0) * scale, 1),
            "Visual Area": f"{visual_stats.get('area_cm2',0):.1f} cm²"
        }