"""
Context manager for tracking agent state and determining when to ask questions.
"""
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    """A piece of context that the agent has gathered."""
    key: str
    value: str
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = Field(default_factory=lambda: __import__("time").time())


class ContextManager:
    """Manages context for an agent to make better decisions about when to ask questions."""
    
    def __init__(self):
        self.context: Dict[str, ContextItem] = {}
        self.required_context: Set[str] = set()
        self.question_history: List[str] = []
        self.max_questions_per_task: int = 3
        
    def add_context(self, key: str, value: str, confidence: float = 1.0, source: str = "unknown") -> None:
        """Add a piece of context."""
        self.context[key] = ContextItem(
            key=key,
            value=value,
            confidence=confidence,
            source=source
        )
        
    def get_context(self, key: str) -> Optional[ContextItem]:
        """Get a piece of context by key."""
        return self.context.get(key)
    
    def set_required_context(self, keys: List[str]) -> None:
        """Set the context keys that are required for the current task."""
        self.required_context = set(keys)
        
    def should_ask_question(self, context_key: str) -> bool:
        """Determine if a question should be asked about a specific context item."""
        # Don't ask if we already have this context with high confidence
        if context_key in self.context and self.context[context_key].confidence > 0.8:
            return False
            
        # Don't ask if we've reached the maximum number of questions
        if len(self.question_history) >= self.max_questions_per_task:
            return False
            
        # Only ask if this is required context
        return context_key in self.required_context
        
    def record_question(self, question: str) -> None:
        """Record that a question was asked."""
        self.question_history.append(question)
        
    def reset_for_new_task(self) -> None:
        """Reset the context manager for a new task."""
        self.required_context = set()
        self.question_history = []
        # Keep the context, as it might be useful for future tasks
        
    def get_missing_required_context(self) -> List[str]:
        """Get a list of required context keys that are missing or have low confidence."""
        missing = []
        for key in self.required_context:
            if key not in self.context or self.context[key].confidence < 0.8:
                missing.append(key)
        return missing
