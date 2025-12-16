"""
Unit Tests for Chat Module

Tests all chat modules:
- llm.py (QwenLLM class)
- rag.py (SimpleRAG class)
- engine.py (ChatEngine class)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chat import llm, rag, engine


class TestQwenLLM:
    """Test llm.py module"""

    @patch('src.chat.llm.Llama')
    @patch('src.chat.llm.paths.LLM_MODEL_PATH')
    def test_qwen_llm_initialization_success(self, mock_path, mock_llama):
        """Test QwenLLM initializes successfully when model exists"""
        mock_path.exists.return_value = True
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance

        llm_instance = llm.QwenLLM()

        assert llm_instance.model_loaded is True
        assert llm_instance.model is not None

    @patch('src.chat.llm.Llama')
    @patch('src.chat.llm.paths.LLM_MODEL_PATH')
    def test_qwen_llm_initialization_failure(self, mock_path, mock_llama):
        """Test QwenLLM handles missing model gracefully"""
        mock_path.exists.return_value = False

        llm_instance = llm.QwenLLM()

        assert llm_instance.model_loaded is False
        assert llm_instance.model is None

    @patch('src.chat.llm.Llama')
    @patch('src.chat.llm.paths.LLM_MODEL_PATH')
    def test_is_available(self, mock_path, mock_llama):
        """Test is_available returns correct status"""
        mock_path.exists.return_value = True
        mock_llama.return_value = Mock()

        llm_instance = llm.QwenLLM()

        assert llm_instance.is_available() is True

    @patch('src.chat.llm.Llama')
    @patch('src.chat.llm.paths.LLM_MODEL_PATH')
    def test_generate_response_unavailable(self, mock_path, mock_llama):
        """Test generate_response when model is unavailable"""
        mock_path.exists.return_value = False

        llm_instance = llm.QwenLLM()
        response = llm_instance.generate_response("Test prompt")

        assert "unavailable" in response.lower()

    @patch('src.chat.llm.Llama')
    @patch('src.chat.llm.paths.LLM_MODEL_PATH')
    def test_generate_food_trivia_fallback(self, mock_path, mock_llama):
        """Test generate_food_trivia returns fallback when model unavailable"""
        mock_path.exists.return_value = False

        llm_instance = llm.QwenLLM()
        food_items = [{'name': 'Dosa'}]
        trivia = llm_instance.generate_food_trivia(food_items)

        assert len(trivia) > 0
        assert isinstance(trivia, str)


class TestSimpleRAG:
    """Test rag.py module"""

    def test_simple_rag_initialization(self):
        """Test SimpleRAG initializes correctly"""
        rag_instance = rag.SimpleRAG()

        assert rag_instance.session_logs == []

    def test_add_log(self):
        """Test add_log adds logs to session"""
        rag_instance = rag.SimpleRAG()
        log_data = {'food_item_1': {'name': 'Rice', 'mass_g': 100}}

        rag_instance.add_log(log_data)

        assert len(rag_instance.session_logs) == 1
        assert rag_instance.session_logs[0] == log_data

    def test_clear_session(self):
        """Test clear_session clears logs"""
        rag_instance = rag.SimpleRAG()
        rag_instance.add_log({'food_item_1': {'name': 'Rice'}})

        rag_instance.clear_session()

        assert len(rag_instance.session_logs) == 0

    def test_search_context_empty(self):
        """Test search_context with no logs"""
        rag_instance = rag.SimpleRAG()

        context = rag_instance.search_context("What did I eat?")

        assert "No meal data" in context

    def test_search_context_with_logs(self):
        """Test search_context retrieves relevant context"""
        rag_instance = rag.SimpleRAG()
        log_data = {
            'food_item_1': {
                'name': 'Rice',
                'mass_g': 100,
                'macros': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
                'metadata': {'confidence': 0.95, 'ingredients': ['rice', 'water']}
            },
            'total_summary': {
                'Energy (kcal)': 130,
                'Protein (g)': 2.7,
                'Carbohydrate (g)': 28,
                'Fat (g)': 0.3
            }
        }
        rag_instance.add_log(log_data)

        context = rag_instance.search_context("What did I eat?")

        assert "Rice" in context
        assert "100g" in context

    def test_get_session_summary_empty(self):
        """Test get_session_summary with no logs"""
        rag_instance = rag.SimpleRAG()

        summary = rag_instance.get_session_summary()

        assert "No meals" in summary

    def test_get_session_summary_with_logs(self):
        """Test get_session_summary with logs"""
        rag_instance = rag.SimpleRAG()
        log_data = {
            'food_item_1': {'name': 'Rice', 'mass_g': 100},
            'total_summary': {'Energy (kcal)': 130}
        }
        rag_instance.add_log(log_data)

        summary = rag_instance.get_session_summary()

        assert "1 meals" in summary
        assert "130" in summary
        assert "Rice" in summary


class TestChatEngine:
    """Test engine.py module"""

    @patch('src.chat.engine.get_llm')
    def test_chat_engine_initialization(self, mock_get_llm):
        """Test ChatEngine initializes correctly"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = True
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()

        assert engine_instance.llm is not None
        assert engine_instance.rag is not None
        assert engine_instance.chat_enabled is True

    @patch('src.chat.engine.get_llm')
    def test_chat_engine_disabled_when_llm_unavailable(self, mock_get_llm):
        """Test ChatEngine disables chat when LLM unavailable"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = False
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()

        assert engine_instance.chat_enabled is False

    @patch('src.chat.engine.get_llm')
    def test_add_meal_log(self, mock_get_llm):
        """Test add_meal_log adds log to RAG"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = True
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()
        log_data = {'food_item_1': {'name': 'Rice'}}

        engine_instance.add_meal_log(log_data)

        assert len(engine_instance.rag.session_logs) == 1

    @patch('src.chat.engine.get_llm')
    def test_generate_trivia_disabled(self, mock_get_llm):
        """Test generate_trivia returns empty when disabled"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = False
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()
        log_data = {'food_item_1': {'name': 'Rice'}}

        trivia = engine_instance.generate_trivia(log_data)

        assert trivia == ""

    @patch('src.chat.engine.get_llm')
    def test_answer_question_disabled(self, mock_get_llm):
        """Test answer_question returns error when disabled"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = False
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()

        response = engine_instance.answer_question("What did I eat?")

        assert "unavailable" in response.lower()

    @patch('src.chat.engine.get_llm')
    def test_clear_session(self, mock_get_llm):
        """Test clear_session clears RAG logs"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = True
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()
        engine_instance.add_meal_log({'food_item_1': {'name': 'Rice'}})

        engine_instance.clear_session()

        assert len(engine_instance.rag.session_logs) == 0

    @patch('src.chat.engine.get_llm')
    def test_get_session_summary(self, mock_get_llm):
        """Test get_session_summary returns RAG summary"""
        mock_llm = Mock()
        mock_llm.is_available.return_value = True
        mock_get_llm.return_value = mock_llm

        engine_instance = engine.ChatEngine()

        summary = engine_instance.get_session_summary()

        assert "No meals" in summary

