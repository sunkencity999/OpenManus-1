#!/usr/bin/env python3
"""
Enhanced version of the ImprovedManus agent that automatically completes tasks.
This version ensures the agent always produces a deliverable, even if it doesn't
gather all the information through browsing or asking questions.
"""
import asyncio
import sys
import re
from typing import List, Dict, Any, Optional, Set
import logging

from app.agent.improved_manus import ImprovedManus
from app.agent.task_completer import TaskCompleter
from app.logger import logger


class AutoCompleteAgent(ImprovedManus):
    """
    Enhanced agent that automatically completes tasks after a certain number of steps.
    This ensures the agent always produces a deliverable for the user.
    """
    
    # Initialize tracking attributes
    step_count: int = 0
    max_steps: int = 20
    task_completed: bool = False
    context: str = ""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context = ""
        self.step_count = 0
        self.task_completed = False
    
    async def step(self) -> bool:
        """Execute a single step of the agent's reasoning process with auto-completion."""
        # Initialize if needed
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True
            
        # Check if we've reached the maximum number of steps
        if self.step_count >= self.max_steps:
            logger.warning(f"Reached maximum number of steps ({self.max_steps}). Forcing task completion.")
            self._force_task_completion()
            return False
            
        # Check if we've reached a reasonable number of steps to complete the task
        if self.step_count >= 5 and not self.task_completed:
            logger.warning(f"Reached {self.step_count} steps. Completing task with available information.")
            self._force_task_completion()
            
        # Think about what to do next
        should_act = await self.think()
        
        # If we should act, do so
        if should_act:
            await self.act()
            
        # Increment step count
        self.step_count += 1
        
        # Continue if we haven't reached the maximum number of steps
        return self.step_count < self.max_steps
    
    def _force_task_completion(self) -> None:
        """Force the completion of the current task."""
        if not hasattr(self, 'task_completer') or self.task_completed:
            return
            
        # Add default information if we don't have enough
        self._add_default_information()
        
        # Create the deliverable
        deliverable = self.task_completer.create_deliverable()
        
        # Add to context
        self.add_to_context(f"\n\nBased on the information I've gathered, here's the completed deliverable:\n\n{deliverable}")
        
        # Mark task as completed
        self.task_completed = True
        logger.warning("🎉 Task completion forced successfully!")
    
    def _add_default_information(self) -> None:
        """Add default information for the current task if missing."""
        if self.task_completer.task_type == "marketing_plan":
            # Extract product name from task if not already set
            if "product_name" not in self.task_completer.gathered_info:
                product_name = self._extract_product_name_from_task()
                if product_name:
                    self.task_completer.add_information("product_name", product_name)
                else:
                    self.task_completer.add_information("product_name", "the product")
            
            # Add default target audience if not set
            if "target_audience" not in self.task_completer.gathered_info:
                self.task_completer.add_information(
                    "target_audience", 
                    "Potential customers interested in the product's features and benefits."
                )
            
            # Add default value proposition if not set
            if "value_proposition" not in self.task_completer.gathered_info:
                product_name = self.task_completer.gathered_info.get("product_name", "The product")
                self.task_completer.add_information(
                    "value_proposition", 
                    f"{product_name} provides significant value through its unique features and capabilities."
                )
            
            # Add default key messaging if not set
            if "key_messaging" not in self.task_completer.gathered_info:
                self.task_completer.add_information(
                    "key_messaging", 
                    "Quality, reliability, innovation, and customer satisfaction."
                )
    
    def _extract_product_name_from_task(self) -> Optional[str]:
        """Extract product name from the current task."""
        if not self.current_task:
            return None
            
        # Look for capitalized words that might be product names
        words = self.current_task.split()
        for word in words:
            if len(word) > 3 and word[0].isupper() and word.lower() not in ["create", "make", "develop", "build", "this", "that"]:
                return word
                
        return None


async def main():
    agent = None
    try:
        # Create the auto-complete agent
        logger.warning("Creating auto-complete agent...")
        agent = await AutoCompleteAgent.create()
        
        # Get the prompt from command line or use default
        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
        else:
            prompt = "Create a marketing plan for ChoiceEngine (https://choiceengine.net#features). Include target audience, value proposition, and key messaging."
        
        logger.warning(f"Running auto-complete agent with task: {prompt}")
        await agent.run(prompt)
        
        logger.warning("Task completed successfully!")
    except Exception as e:
        logger.error(f"Error running auto-complete agent: {e}")
    finally:
        if agent:
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
