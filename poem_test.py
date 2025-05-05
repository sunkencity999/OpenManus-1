#!/usr/bin/env python3
"""
Test script for the poem creation functionality in the improved OpenManus agent.
This script demonstrates the agent's ability to create a poem about Harriet Tubman
in the style of Stephen King without redundant questioning.
"""
import asyncio
import sys
from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def run_poem_test():
    """Run a test of the poem creation functionality."""
    logger.warning("Creating improved agent for poem test...")
    
    # Create the agent using the factory method
    agent = await ImprovedManus.create()
    
    # Define the poem task
    task = "Create a poem about Harriet Tubman in the style of Stephen King."
    logger.warning(f"Running agent with task: {task}")
    
    # Run the agent with the task
    response = await agent.run(task)
    
    # Display the result
    print("\n" + "="*80)
    print("POEM TEST RESULT")
    print("="*80)
    print(response)
    print("="*80)


async def main():
    try:
        await run_poem_test()
    except Exception as e:
        logger.error(f"Error in poem test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up any resources
        pass


if __name__ == "__main__":
    asyncio.run(main())
