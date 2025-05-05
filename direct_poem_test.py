#!/usr/bin/env python3
"""
Direct test of the poem creation functionality using the TaskCompleter.
This script demonstrates the agent's ability to create a poem about Harriet Tubman
in the style of Stephen King without requiring any agent interaction.
"""
import asyncio
from app.agent.task_completer import TaskCompleter
from app.logger import logger


async def run_direct_poem_test():
    """Run a direct test of the poem creation functionality."""
    logger.warning("Creating TaskCompleter for direct poem test...")
    
    # Create a task completer
    task_completer = TaskCompleter()
    
    # Analyze the poem task
    task_description = "Create a poem about Harriet Tubman in the style of Stephen King."
    task_completer.analyze_task(task_description)
    
    # Add the required information directly
    task_completer.add_information("subject", "harriet tubman")
    task_completer.add_information("style", "stephen king")
    task_completer.add_information("tone", "dark and suspenseful")
    
    # Create the poem
    poem = task_completer.create_deliverable()
    
    # Display the result
    print("\n" + "="*80)
    print("DIRECT POEM TEST RESULT")
    print("="*80)
    print(poem)
    print("="*80)


async def main():
    try:
        await run_direct_poem_test()
    except Exception as e:
        logger.error(f"Error in direct poem test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
