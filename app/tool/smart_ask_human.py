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
        # Always ask the question if it contains certain keywords that indicate it's important
        force_ask = False
        important_keywords = [
            "confirm", "verify", "approve", "permission", "prefer", "choice", "select", 
            "decide", "opinion", "want", "need", "should", "would you", "do you", "can you",
            "provide", "overview", "structure", "details", "specifics", "requirements"
        ]
        
        # Check if the question contains any important keywords
        if any(keyword in inquire.lower() for keyword in important_keywords):
            # Check if this question is similar to previously asked questions
            # Even important questions shouldn't be asked multiple times with similar wording
            if self._context_manager and hasattr(self._context_manager, 'asked_questions'):
                normalized_question = inquire.strip().lower()
                
                # Check for exact duplicates first
                if any(q.strip().lower() == normalized_question for q in self._context_manager.asked_questions[-10:]):
                    print(f"Exact duplicate question detected in SmartAskHuman: {inquire}")
                    force_ask = False  # Don't force if it's an exact duplicate of a recent question
                    
                # Check for semantic similarity if we have a calculate_similarity method
                elif hasattr(self._context_manager, '_calculate_similarity') and len(self._context_manager.asked_questions) > 0:
                    # Check similarity with recent questions
                    for prev_q in self._context_manager.asked_questions[-10:]:
                        try:
                            similarity = self._context_manager._calculate_similarity(inquire, prev_q)
                            if similarity > 0.7:  # High similarity threshold
                                print(f"Similar question detected in SmartAskHuman (similarity: {similarity:.2f}): {inquire} vs {prev_q}")
                                force_ask = False
                                break
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                    else:  # No similar questions found
                        force_ask = True
                else:
                    force_ask = True
            else:
                force_ask = True
        
        # If we have a context manager, check if we should skip this question
        if self._context_manager:
            # Use a more specific context key based on the normalized question
            # This helps prevent hash collisions and makes retrieval more reliable
            normalized_q = inquire.strip().lower()
            context_key = f"user_response_{hash(normalized_q) % 100000}"
            
            # Check if we already have this information and it's not a forced question
            existing = self._context_manager.get_context(context_key)
            if existing and not force_ask:
                return f"[Using existing information: {existing.value}]"
            
            # Record that we asked a question
            # Use add_question for PersistentMemory or record_question for ToolUsageTracker
            if hasattr(self._context_manager, 'add_question'):
                self._context_manager.add_question(inquire)
            elif hasattr(self._context_manager, 'record_question'):
                self._context_manager.record_question(inquire)
            else:
                print(f"Warning: Context manager doesn't have add_question or record_question method: {type(self._context_manager).__name__}")
            
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
