"""
Memory manager for tracking conversation history and preventing redundant questions.
"""
from typing import Dict, List, Set, Optional, Any
import re
from pydantic import BaseModel, Field


class ConversationMemory(BaseModel):
    """Tracks conversation history to prevent redundant questions and improve context awareness."""
    
    # Track questions that have been asked
    asked_questions: List[str] = Field(default_factory=list)
    
    # Track answers received from the user
    user_responses: Dict[str, str] = Field(default_factory=dict)
    
    # Track topics that have been discussed
    discussed_topics: Set[str] = Field(default_factory=set)
    
    # Track when the agent is getting stuck in loops
    repeated_patterns: Dict[str, int] = Field(default_factory=dict)
    
    # Track required context keys
    required_context: Set[str] = Field(default_factory=set)
    
    # Context storage
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Maximum similarity threshold for considering questions as duplicates
    similarity_threshold: float = 0.8
    
    def add_question(self, question: str) -> None:
        """Record a question that was asked."""
        self.asked_questions.append(question)
        
        # Track potential repeated patterns
        if len(self.asked_questions) >= 2:
            pattern = self.asked_questions[-2] + " -> " + question
            self.repeated_patterns[pattern] = self.repeated_patterns.get(pattern, 0) + 1
    
    def add_response(self, question: str, response: str) -> None:
        """Record a response to a question."""
        self.user_responses[question] = response
        
        # Extract topics from the response
        topics = self._extract_topics(response)
        self.discussed_topics.update(topics)
        
    def add_context(self, key: str, value: str, confidence: float = 1.0, source: str = "unknown") -> None:
        """Add context information."""
        self.user_responses[key] = value
    
    def set_required_context(self, context_keys: List[str]) -> None:
        """Set the required context keys for the current task."""
        self.required_context.update(context_keys)
        
    def should_ask_question(self, context_key: str) -> bool:
        """Determine if a question should be asked based on context."""
        # If we've already asked too many questions, avoid asking more
        if len(self.asked_questions) > 5:
            return False
            
        # If this is required context and we don't have it, ask
        if context_key in self.required_context:
            return True
            
        # Otherwise, avoid redundant questions
        return not self.is_similar_question(context_key)
    
    def is_similar_question(self, question: str) -> bool:
        """Check if a similar question has already been asked."""
        # Simple implementation using string similarity
        # In a real implementation, use embedding similarity or more sophisticated NLP
        for asked in self.asked_questions:
            if self._calculate_similarity(question, asked) > self.similarity_threshold:
                return True
        return False
    
    def get_relevant_context(self, query: str) -> List[str]:
        """Get relevant context from previous conversation based on a query."""
        relevant_responses = []
        
        for question, response in self.user_responses.items():
            if any(topic in query.lower() for topic in self._extract_topics(question)):
                relevant_responses.append(f"Q: {question}\nA: {response}")
                
        return relevant_responses
    
    def is_stuck_in_loop(self) -> bool:
        """Detect if the agent is stuck in a question loop."""
        for pattern, count in self.repeated_patterns.items():
            if count >= 2:  # Same pattern repeated multiple times
                return True
        return False
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity (simple implementation)."""
        # Normalize strings
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()
        
        # Simple word overlap similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple implementation - extract capitalized words and phrases
        topics = []
        words = text.split()
        for word in words:
            if len(word) > 3 and word[0].isupper():
                topics.append(word)
        return topics
        
    def is_in_question_loop(self) -> bool:
        """Detect if the agent is stuck in a question loop."""
        # Check if we have any repeated patterns
        for pattern, count in self.repeated_patterns.items():
            if count >= 2:  # If we've seen the same pattern twice
                return True
                
        return False
        
    def get_context(self, key: str) -> Optional[Any]:
        """Get context value by key if it exists."""
        if key in self.user_responses:
            return type('ContextValue', (), {'value': self.user_responses[key]})
        return None
        
    def reset(self) -> None:
        """Reset the conversation memory."""
        self.asked_questions = []
        self.user_responses = {}
        self.discussed_topics = set()
        self.repeated_patterns = {}
        self.required_context = set()
        
    def reset_for_new_task(self) -> None:
        """Reset the conversation memory for a new task."""
        self.reset()


class ToolUsageTracker(BaseModel):
    """Tracks tool usage to guide better tool selection."""
    
    # Track which tools have been used
    used_tools: Dict[str, int] = Field(default_factory=dict)
    
    # Track tool usage patterns
    tool_patterns: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    # Track tool success/failure
    tool_outcomes: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    
    def record_tool_usage(self, tool_name: str, args: Dict[str, Any], outcome: str = "success") -> None:
        """Record that a tool was used."""
        # Increment usage count
        self.used_tools[tool_name] = self.used_tools.get(tool_name, 0) + 1
        
        # Record outcome
        if tool_name not in self.tool_outcomes:
            self.tool_outcomes[tool_name] = {"success": 0, "failure": 0}
        self.tool_outcomes[tool_name][outcome] += 1
        
        # Record pattern (which tool tends to follow which)
        if len(self.used_tools) >= 2:
            prev_tool = list(self.used_tools.keys())[-2]
            if prev_tool not in self.tool_patterns:
                self.tool_patterns[prev_tool] = {}
            self.tool_patterns[prev_tool][tool_name] = self.tool_patterns[prev_tool].get(tool_name, 0) + 1
    
    def suggest_next_tool(self, current_tool: str) -> Optional[str]:
        """Suggest the next tool to use based on patterns."""
        if current_tool not in self.tool_patterns:
            return None
            
        patterns = self.tool_patterns[current_tool]
        if not patterns:
            return None
            
        # Return the most commonly used next tool
        return max(patterns.items(), key=lambda x: x[1])[0]
    
    def should_avoid_tool(self, tool_name: str) -> bool:
        """Check if a tool should be avoided due to failures."""
        if tool_name not in self.tool_outcomes:
            return False
            
        outcomes = self.tool_outcomes[tool_name]
        total = outcomes.get("success", 0) + outcomes.get("failure", 0)
        
        if total < 3:  # Not enough data
            return False
            
        failure_rate = outcomes.get("failure", 0) / total
        return failure_rate > 0.7  # Avoid tools with high failure rates
    
    def reset(self) -> None:
        """Reset the tool usage tracker."""
        self.used_tools = {}
        self.tool_patterns = {}
        self.tool_outcomes = {}
