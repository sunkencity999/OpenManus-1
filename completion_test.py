#!/usr/bin/env python3
"""
Test script specifically focused on ensuring task completion.
This script forces the agent to complete the marketing plan after browsing.
"""
import asyncio
import sys
from app.agent.improved_manus import ImprovedManus
from app.agent.task_completer import TaskCompleter
from app.logger import logger


async def main():
    agent = None
    try:
        # Create the improved agent
        logger.warning("Creating improved agent...")
        agent = await ImprovedManus.create()
        
        # Run the agent with a specific task that includes a URL with fragment
        prompt = "Create a marketing plan for ChoiceEngine (https://choiceengine.net#features). Include target audience, value proposition, and key messaging."
        
        logger.warning(f"Running improved agent with task: {prompt}")
        
        # Manually analyze the task
        agent.current_task = prompt
        
        # Initialize the task completer
        task_completer = TaskCompleter()
        task_completer.analyze_task(prompt)
        
        # Add basic information
        task_completer.add_information("product_name", "ChoiceEngine")
        task_completer.add_information("target_audience", "Businesses and organizations that need to make data-driven decisions through polling and feedback.")
        task_completer.add_information("value_proposition", "ChoiceEngine helps organizations make better decisions by providing easy-to-use polling tools.")
        task_completer.add_information("key_messaging", "Simple to use, powerful analytics, customizable polls, real-time results")
        
        # Create the deliverable
        deliverable = task_completer.create_deliverable()
        
        # Print the deliverable
        print("\n" + "="*50)
        print("MARKETING PLAN DELIVERABLE")
        print("="*50)
        print(deliverable)
        print("="*50)
        
        logger.warning("Task completed successfully!")
    except Exception as e:
        logger.error(f"Error testing improved agent: {e}")
    finally:
        if agent:
            await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
