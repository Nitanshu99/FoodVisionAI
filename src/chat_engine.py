"""
LangGraph Chat Engine for Food Analysis.

Simple state machine that:
1. Generates trivia after image analysis
2. Answers questions using current session RAG
3. Gracefully handles LLM failures
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.message import AnyMessage
from src.llm_utils import get_llm
from src.rag_engine import SimpleRAG
from langchain_core.messages import HumanMessage, AIMessage

class ChatState(TypedDict):
    """State for the chat conversation."""
    messages: Annotated[list[AnyMessage], add_messages]
    current_log: Optional[dict]  # Latest inference log
    context: str  # RAG context

class ChatEngine:
    """
    Main chat engine using LangGraph for state management.
    Handles both auto-trivia and user Q&A.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.rag = SimpleRAG()
        self.graph = self._build_graph()
        self.chat_enabled = self.llm.is_available()
        
        if not self.chat_enabled:
            print("⚠️ Chat features disabled (LLM not available)")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(ChatState)
        
        # Add nodes
        graph.add_node("trivia_generator", self._generate_trivia)
        graph.add_node("rag_retriever", self._retrieve_context)
        graph.add_node("chat_responder", self._respond_to_question)
        
        # Set entry point
        graph.set_entry_point("rag_retriever")
        
        # Add edges (simple linear flow)
        graph.add_edge("rag_retriever", "chat_responder")
        graph.add_edge("chat_responder", "__end__")
        graph.add_edge("trivia_generator", "__end__")
        
        return graph.compile()
    
    def _generate_trivia(self, state: ChatState) -> ChatState:
        """Generate food trivia from current log."""
        if not self.chat_enabled or not state.get("current_log"):
            return state
        
        # Extract food items from log
        food_items = []
        log = state["current_log"]
        for key, value in log.items():
            if key.startswith('food_item'):
                food_items.append(value)
        
        trivia = self.llm.generate_food_trivia(food_items)
        
        # Add trivia as a system message
        state["messages"].append({
            "role": "assistant",
            "content": f"🧠 **Did you know?** {trivia}",
            "type": "trivia"
        })
        
        return state
    
    def _retrieve_context(self, state: ChatState) -> ChatState:
        """Retrieve relevant context from session logs."""
        if not state.get("messages"):
            state["context"] = ""
            return state
        
        # Get the last user message
        last_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_message = msg.content
                break
        
        if last_message:
            state["context"] = self.rag.search_context(last_message)
        else:
            state["context"] = ""
        
        return state
    
    def _respond_to_question(self, state: ChatState) -> ChatState:
        """Generate response to user question using context."""
        if not self.chat_enabled:
            error_msg = AIMessage(
                content="Chat features are currently unavailable.",
                additional_kwargs={"type": "error"}
            )
            state["messages"].append(error_msg)
            return state

        # Get the last user message
        user_question = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break

        if not user_question:
            return state

        # Check if this is a health-related question
        health_keywords = ['diabetes', 'diabetic', 'sugar', 'allergy', 'allergic',
                          'gluten', 'lactose', 'vegan', 'keto', 'healthy']
        is_health_query = any(kw in user_question.lower() for kw in health_keywords)

        # Generate response using RAG context
        if state.get("context"):
            if is_health_query:
                # Extract condition from question
                condition = "general health"
                for kw in health_keywords:
                    if kw in user_question.lower():
                        condition = kw
                        break

                # Get health-specific context
                health_context = self.rag.get_health_context(condition)
                response = self.llm.answer_health_question(user_question, health_context)
            else:
                # Regular nutrition question
                prompt = f"""Context from current session:
{state['context']}

User question: {user_question}

Provide a helpful answer based on the meal data above."""

                response = self.llm.generate_response(prompt, max_tokens=120)
        else:
            response = "I don't have any meal data to reference. Please upload a meal photo first!"

        answer_msg = AIMessage(
            content=response,
            additional_kwargs={"type": "answer"}
        )
        state["messages"].append(answer_msg)

        return state
    
    # Public API methods
    
    def add_meal_log(self, log_data: dict):
        """Add a new meal log to the session."""
        self.rag.add_log(log_data)
    
    def generate_trivia(self, log_data: dict) -> str:
        """Generate trivia for a meal (called after image analysis)."""
        if not self.chat_enabled:
            return ""

        # Extract food items from log
        food_items = []
        for key, value in log_data.items():
            if key.startswith('food_item'):
                food_items.append(value)

        # Generate trivia directly (bypass graph for simplicity)
        trivia = self.llm.generate_food_trivia(food_items)

        return trivia
    
    def answer_question(self, question: str) -> str:
        """Answer a user question using current session context (simplified)."""
        if not self.chat_enabled:
            return "Chat features are currently unavailable."

        # Check if this is a health-related question
        health_keywords = ['diabetes', 'diabetic', 'sugar', 'allergy', 'allergic',
                          'gluten', 'lactose', 'vegan', 'keto', 'healthy']
        is_health_query = any(kw in question.lower() for kw in health_keywords)

        # Get context from RAG
        context = self.rag.search_context(question)

        if is_health_query:
            # Extract condition from question
            condition = "general health"
            for kw in health_keywords:
                if kw in question.lower():
                    condition = kw
                    break

            # Get health-specific context
            health_context = self.rag.get_health_context(condition)
            response = self.llm.answer_health_question(question, health_context)
        else:
            # Regular nutrition question
            if context:
                prompt = f"""Context from current session:
{context}

User Question: {question}

Provide a helpful, concise answer based on the context above."""
                response = self.llm.generate_response(prompt, max_tokens=200, temperature=0.6)
            else:
                response = "I don't have any meal data in the current session. Please upload a meal photo first!"

        return response
    
    def clear_session(self):
        """Clear current session data."""
        self.rag.clear_session()
    
    def get_session_summary(self) -> str:
        """Get summary of current session."""
        return self.rag.get_session_summary()
