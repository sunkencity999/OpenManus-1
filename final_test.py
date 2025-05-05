#!/usr/bin/env python3
"""
Final test script for the improved OpenManus agent with enhanced reasoning.
This script demonstrates all the improvements:
1. URL detection and browser navigation
2. Task completion with deliverable generation
3. Reduced redundant questioning
"""
import asyncio
import sys
from app.agent.manus import Manus
from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def test_improved_agent():
    """Test the improved agent with a marketing plan task."""
    agent = None
    try:
        # Create the improved agent
        logger.warning("Creating improved agent...")
        agent = await ImprovedManus.create()
        
        # Run the agent with a specific task that includes a URL with fragment
        prompt = "Create a marketing plan for ChoiceEngine (https://choiceengine.net#features). Include target audience, value proposition, and key messaging."
        
        logger.warning(f"Running improved agent with task: {prompt}")
        await agent.run(prompt)
        
        logger.warning("Task completed successfully!")
    except Exception as e:
        logger.error(f"Error testing improved agent: {e}")
    finally:
        if agent:
            await agent.cleanup()


async def test_original_agent():
    """Test the original agent with the same task for comparison."""
    agent = None
    try:
        # Create the original agent
        logger.warning("Creating original agent...")
        agent = await Manus.create()
        
        # Run the agent with the same task
        prompt = "Create a marketing plan for ChoiceEngine (https://choiceengine.net#features). Include target audience, value proposition, and key messaging."
        
        logger.warning(f"Running original agent with task: {prompt}")
        await agent.run(prompt)
        
        logger.warning("Task completed.")
    except Exception as e:
        logger.error(f"Error testing original agent: {e}")
    finally:
        if agent:
            await agent.cleanup()


async def main():
    """Run the tests based on command line arguments."""
    if len(sys.argv) > 1 and sys.argv[1] == "original":
        await test_original_agent()
    else:
        await test_improved_agent()


if __name__ == "__main__":
    asyncio.run(main())
