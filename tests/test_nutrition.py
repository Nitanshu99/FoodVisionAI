"""
Unit Tests for Nutrition Module

Tests all nutrition modules:
- engine.py (NutrientEngine class)
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nutrition import engine


class TestNutrientEngine:
    """Test engine.py module"""

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_nutrient_engine_initialization(self, mock_read_parquet):
        """Test NutrientEngine initializes correctly"""
        # Mock empty dataframes
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()

        assert engine_instance is not None
        assert hasattr(engine_instance, 'serving_db')
        assert hasattr(engine_instance, 'nutrition_db')
        assert hasattr(engine_instance, 'units_db')
        assert hasattr(engine_instance, 'links_db')
        assert hasattr(engine_instance, 'recipes_db')
        assert hasattr(engine_instance, 'names_db')

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_food_name_from_names_db(self, mock_read_parquet):
        """Test get_food_name retrieves from names_db"""
        # Mock names database
        mock_names_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'recipe_name': ['Biryani']
        }).set_index('recipe_code')
        
        mock_read_parquet.return_value = mock_names_df

        engine_instance = engine.NutrientEngine()
        engine_instance.names_db = mock_names_df

        food_name = engine_instance.get_food_name('ASC123')

        assert food_name == 'Biryani'

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_food_name_fallback_to_nutrition_db(self, mock_read_parquet):
        """Test get_food_name falls back to nutrition_db"""
        # Mock nutrition database
        mock_nutrition_df = pd.DataFrame({
            'food_code': ['ASC123'],
            'food_name': ['Rice Dish']
        }).set_index('food_code')
        
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.names_db = pd.DataFrame()
        engine_instance.nutrition_db = mock_nutrition_df

        food_name = engine_instance.get_food_name('ASC123')

        assert food_name == 'Rice Dish'

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_food_name_returns_class_id_if_not_found(self, mock_read_parquet):
        """Test get_food_name returns class_id if not found"""
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.names_db = pd.DataFrame()
        engine_instance.nutrition_db = pd.DataFrame()

        food_name = engine_instance.get_food_name('UNKNOWN123')

        assert food_name == 'UNKNOWN123'

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_source_link(self, mock_read_parquet):
        """Test get_source_link retrieves URL"""
        # Mock links database
        mock_links_df = pd.DataFrame({
            'Food Code': ['ASC123'],
            'Link': ['https://example.com/recipe']
        }).set_index('Food Code')
        
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.links_db = mock_links_df

        link = engine_instance.get_source_link('ASC123')

        assert link == 'https://example.com/recipe'

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_source_link_not_found(self, mock_read_parquet):
        """Test get_source_link returns default if not found"""
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.links_db = pd.DataFrame()

        link = engine_instance.get_source_link('UNKNOWN123')

        assert link == 'N/A'

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_ingredients(self, mock_read_parquet):
        """Test get_ingredients retrieves ingredient list"""
        # Mock recipes database
        mock_recipes_df = pd.DataFrame({
            'recipe_code': ['ASC123', 'ASC123', 'ASC123'],
            'ingredient_name_org': ['Rice', 'Chicken', 'Spices']
        })
        
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.recipes_db = mock_recipes_df

        ingredients = engine_instance.get_ingredients('ASC123')

        assert len(ingredients) == 3
        assert 'Rice' in ingredients
        assert 'Chicken' in ingredients
        assert 'Spices' in ingredients

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_ingredients_not_found(self, mock_read_parquet):
        """Test get_ingredients returns empty list if database is empty"""
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.recipes_db = pd.DataFrame()

        ingredients = engine_instance.get_ingredients('UNKNOWN123')

        assert ingredients == []

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_get_density_default(self, mock_read_parquet):
        """Test get_density returns default value"""
        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.units_db = pd.DataFrame()

        density = engine_instance.get_density('Unknown Food')

        assert density == 0.85

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_calculate_nutrition_basic(self, mock_read_parquet):
        """Test calculate_nutrition with basic visual stats"""
        # Mock nutrition database
        mock_nutrition_df = pd.DataFrame({
            'food_code': ['ASC123'],
            'energy_kcal': [150.0],
            'protein_g': [5.0],
            'carb_g': [30.0],
            'fat_g': [2.0]
        }).set_index('food_code')

        mock_names_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'recipe_name': ['Test Food']
        }).set_index('recipe_code')

        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.nutrition_db = mock_nutrition_df
        engine_instance.names_db = mock_names_df
        engine_instance.serving_db = pd.DataFrame()
        engine_instance.links_db = pd.DataFrame()
        engine_instance.recipes_db = pd.DataFrame()

        visual_stats = {'area_cm2': 50.0}
        result = engine_instance.calculate_nutrition('ASC123', visual_stats)

        assert result is not None
        assert result['Food Code'] == 'ASC123'
        assert result['Food Name'] == 'Test Food'
        assert 'Calculated Mass (g)' in result
        assert 'Energy (kcal)' in result
        assert 'Protein (g)' in result
        assert 'Carbohydrate (g)' in result
        assert 'Fat (g)' in result

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_calculate_nutrition_container_strategy(self, mock_read_parquet):
        """Test calculate_nutrition with container (volumetric) strategy"""
        # Mock serving database with bowl unit
        mock_serving_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'servings_unit': ['bowl'],
            'size_of_servings': ['1']
        }).set_index('recipe_code')

        mock_names_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'recipe_name': ['Soup']
        }).set_index('recipe_code')

        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.serving_db = mock_serving_df
        engine_instance.names_db = mock_names_df
        engine_instance.nutrition_db = pd.DataFrame()
        engine_instance.links_db = pd.DataFrame()
        engine_instance.recipes_db = pd.DataFrame()

        visual_stats = {'area_cm2': 50.0, 'occupancy_ratio': 0.9}
        result = engine_instance.calculate_nutrition('ASC123', visual_stats)

        assert result['Logic Path'] == 'Volumetric (Container)'
        assert result['Calculated Mass (g)'] > 0

    @patch('src.nutrition.engine.pd.read_parquet')
    def test_calculate_nutrition_piece_strategy(self, mock_read_parquet):
        """Test calculate_nutrition with piece (discrete solid) strategy"""
        # Mock serving database with piece unit
        mock_serving_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'servings_unit': ['piece'],
            'size_of_servings': ['1']
        }).set_index('recipe_code')

        mock_names_df = pd.DataFrame({
            'recipe_code': ['ASC123'],
            'recipe_name': ['Bread']
        }).set_index('recipe_code')

        mock_read_parquet.return_value = pd.DataFrame()

        engine_instance = engine.NutrientEngine()
        engine_instance.serving_db = mock_serving_df
        engine_instance.names_db = mock_names_df
        engine_instance.nutrition_db = pd.DataFrame()
        engine_instance.links_db = pd.DataFrame()
        engine_instance.recipes_db = pd.DataFrame()
        engine_instance.units_db = pd.DataFrame()

        visual_stats = {'area_cm2': 50.0}
        result = engine_instance.calculate_nutrition('ASC123', visual_stats)

        assert result['Logic Path'] == 'Geometric (Cylinder)'
        assert result['Calculated Mass (g)'] > 0

