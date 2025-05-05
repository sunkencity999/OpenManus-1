#!/usr/bin/env python3
"""
Simplified test script to demonstrate the task completion functionality.
This script directly uses the TaskCompleter to create a marketing plan.
"""
import asyncio
import sys
from app.agent.task_completer import TaskCompleter
from app.logger import logger


async def main():
    # Create a task completer
    task_completer = TaskCompleter()
    
    # Analyze a marketing plan task
    task_description = "Create a marketing plan for ChoiceEngine"
    task_completer.analyze_task(task_description)
    
    # Add sample information
    task_completer.add_information("product_name", "ChoiceEngine")
    task_completer.add_information("target_audience", "Business professionals and organizations looking to make better decisions through polling and feedback.")
    task_completer.add_information("value_proposition", "ChoiceEngine helps organizations make better decisions by providing easy-to-use polling tools and real-time feedback analysis.")
    task_completer.add_information("key_messaging", "Simple to use, powerful analytics, customizable polls, real-time results")
    
    # Create the deliverable
    deliverable = task_completer.create_deliverable()
    
    # Print the deliverable
    print("\n" + "="*50)
    print("MARKETING PLAN DELIVERABLE")
    print("="*50)
    print(deliverable)
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
