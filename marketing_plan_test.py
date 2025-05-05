#!/usr/bin/env python3
"""
Test script specifically for creating a marketing plan for ChoiceEngine.
This script uses the improved agent with enhanced URL detection and browser navigation.
"""
import asyncio
import sys
from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def main():
    # Create and initialize the improved agent
    agent = await ImprovedManus.create()
    try:
        # Use a specific prompt that includes the URL with fragment
        prompt = "Create a comprehensive marketing plan for ChoiceEngine (https://choiceengine.net#features). Include target audience, value proposition, marketing channels, and key messaging."
        
        logger.warning(f"Starting marketing plan creation task...")
        await agent.run(prompt)
        logger.info("Task completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
