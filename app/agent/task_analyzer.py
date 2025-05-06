"""
Task analyzer for breaking down complex tasks and determining information requirements.
"""
from typing import Dict, List, Set, Optional
from pydantic import BaseModel, Field


class TaskStep(BaseModel):
    """A step in a task execution plan."""
    description: str
    required_context: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    completed: bool = False
    
    
class TaskPlan(BaseModel):
    """A plan for executing a task."""
    task_description: str
    steps: List[TaskStep] = Field(default_factory=list)
    current_step_index: int = 0
    
    @property
    def current_step(self) -> Optional[TaskStep]:
        """Get the current step in the plan."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def advance(self) -> None:
        """Mark the current step as completed and advance to the next step."""
        if self.current_step:
            self.current_step.completed = True
            self.current_step_index += 1
            
    def all_required_context(self) -> Set[str]:
        """Get all context keys required for this task."""
        return set().union(*(step.required_context for step in self.steps))


class TaskAnalyzer:
    """
    Analyzes tasks to determine what information is required and creates execution plans.
    This helps the agent avoid asking unnecessary questions by understanding the full task upfront.
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.current_plan: Optional[TaskPlan] = None
        
    async def analyze_task(self, task_description: str) -> TaskPlan:
        """
        Analyze a task to create a step-by-step plan with required information.
        
        Args:
            task_description: Description of the task to analyze
            
        Returns:
            A TaskPlan object with steps and required context
        """
        # In a real implementation, we would use the LLM to analyze the task
        # For now, we'll create a simple placeholder implementation
        
        plan = TaskPlan(task_description=task_description)
        
        # This is where we'd use the LLM to break down the task
        # For now, we'll create a simple plan based on task type
        task_lower = task_description.lower()
        
        # Check for creative content tasks
        if any(keyword in task_lower for keyword in ["poem", "story", "essay", "song", "script"]):
            # This is a creative content task
            creative_type = "content"
            if "poem" in task_lower:
                creative_type = "poem"
            elif "story" in task_lower:
                creative_type = "story"
            elif "essay" in task_lower:
                creative_type = "essay"
            elif "song" in task_lower:
                creative_type = "song"
            elif "script" in task_lower:
                creative_type = "script"
                
            # Determine if there's a file save request
            save_to_file = False
            if any(keyword in task_lower for keyword in ["save", "write to", "create file", ".txt", ".md"]):
                save_to_file = True
                
            # Create a plan for creative content generation
            plan.steps = [
                TaskStep(
                    description="Understand creative content requirements",
                    required_context=["subject", "style", "tone"],
                    tools=["ask_human"]
                ),
                TaskStep(
                    description="Generate creative content",
                    required_context=["subject", "style", "tone"],
                    tools=["str_replace_editor" if save_to_file else "none"]
                )
            ]
        elif "file" in task_lower:
            plan.steps = [
                TaskStep(
                    description="Identify target file",
                    required_context=["file_path"],
                    tools=["find_files"]
                ),
                TaskStep(
                    description="Process file",
                    required_context=["file_path", "processing_method"],
                    tools=["read_file", "write_file"]
                )
            ]
        else:
            # Generic fallback plan
            plan.steps = [
                TaskStep(
                    description="Understand user request",
                    required_context=["user_goal"],
                    tools=["ask_human"]
                ),
                TaskStep(
                    description="Execute the task",
                    required_context=[],
                    tools=[]
                )
            ]
            
        self.current_plan = plan
        return plan
        
    def get_required_context_for_current_step(self) -> List[str]:
        """Get the context keys required for the current step."""
        if not self.current_plan or not self.current_plan.current_step:
            return []
        return self.current_plan.current_step.required_context
