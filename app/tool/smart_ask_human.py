"""
An improved tool for asking humans questions, with better context awareness.
"""
from typing import Dict, List, Optional, Any
import json

from app.tool import BaseTool


class SmartAskHuman(BaseTool):
    """
    An improved tool for asking humans questions that avoids unnecessary or repetitive questions.
    Uses a context manager to track what information is already known or can be inferred.
    """

    name: str = "ask_human"
    description: str = (
        "Use this tool to ask the human for information, but ONLY when absolutely necessary. "
        "Before asking, check if you can infer the answer from context or if it's truly required."
    )
    parameters: Dict = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "The question you want to ask the human.",
            }
        },
        "required": ["inquire"],
    }

    def __init__(self):
        super().__init__()
        # We'll set this from the agent
        self._context_manager = None
        
    @property
    def context_manager(self):
        return self._context_manager
        
    @context_manager.setter
    def context_manager(self, value):
        self._context_manager = value
        
    async def execute(self, inquire: str) -> str:
        """
        Execute the tool to ask a human a question.
        
        Args:
            inquire: The question to ask
            
        Returns:
            The human's response
        """
        # If we have a context manager, check if we should skip this question
        if self._context_manager:
            # Use a generic context key based on the question
            context_key = f"user_response_{hash(inquire) % 10000}"
            
            # Check if we already have this information
            existing = self._context_manager.get_context(context_key)
            if existing:
                return f"[Using existing information: {existing.value}]"
            
            # Record that we asked a question
            self._context_manager.record_question(inquire)
            
            # Ask the human
            response = input(f"""Bot: {inquire}\n\nYou: """).strip()
            
            # Store the response
            self._context_manager.add_context(
                key=context_key,
                value=response,
                confidence=1.0,
                source="user_input"
            )
            
            return response
        else:
            # Fallback to simple input if no context manager
            return input(f"""Bot: {inquire}\n\nYou: """).strip()
        
    def reset_for_new_task(self) -> None:
        """Reset for a new task."""
        if self._context_manager:
            self._context_manager.reset_for_new_task()
            
    def get_answered_questions(self) -> List[str]:
        """Get a list of questions that have been answered.
        
        Returns:
            A list of questions that have been asked and answered
        """
        if self._context_manager and hasattr(self._context_manager, 'get_questions'):
            return self._context_manager.get_questions()
        return []
