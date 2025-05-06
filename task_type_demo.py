#!/usr/bin/env python3
"""
Task Type Expansion Demo for LocalManus

This script demonstrates the enhanced task type capabilities of LocalManus,
showing how it can handle various task types including research reports,
technical documentation, content summaries, and more.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.agent.improved_manus import ImprovedManus
from app.agent.task_completer import TaskCompleter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("task_type_demo")

async def demo_task_types():
    """Demonstrate the different task types supported by LocalManus."""
    
    # Create memory directory if it doesn't exist
    memory_dir = os.path.expanduser("~/.localmanus/memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize the agent
    agent = ImprovedManus(
        memory_path=memory_dir,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )
    
    # Display welcome message
    print("\n" + "="*80)
    print("LocalManus Task Type Expansion Demo")
    print("="*80)
    print("\nThis demo showcases LocalManus's ability to handle various task types.\n")
    
    # Define demo tasks
    demo_tasks = {
        "research_report": "Create a research report on renewable energy sources",
        "technical_documentation": "Write technical documentation for a Python API library called DataProcessor",
        "content_summary": "Summarize the key points of the latest climate change report",
        "data_analysis": "Analyze the sales data for Q1 2023 and provide insights",
        "blog_post": "Write a blog post about machine learning applications in healthcare",
        "email_content": "Draft an email to customers about our new product launch",
        "social_media": "Create social media content for Twitter announcing our company anniversary"
    }
    
    # Let the user choose a task type
    print("Available task types:")
    for i, (task_type, task_desc) in enumerate(demo_tasks.items(), 1):
        print(f"{i}. {task_type.replace('_', ' ').title()}: {task_desc}")
    
    choice = input("\nEnter the number of the task type to demo (or 'q' to quit): ")
    if choice.lower() == 'q':
        return
    
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(demo_tasks):
            print("Invalid choice. Please try again.")
            return
        
        selected_task_type = list(demo_tasks.keys())[choice_idx]
        selected_task = demo_tasks[selected_task_type]
        
        print(f"\nDemonstrating task type: {selected_task_type.replace('_', ' ').title()}")
        print(f"Task: {selected_task}\n")
        
        # Create a task completer instance
        task_completer = TaskCompleter()
        
        # Analyze the task
        task_completer.analyze_task(selected_task)
        
        print(f"Task type detected: {task_completer.task_type}")
        print(f"Required information: {', '.join(task_completer.task_requirements)}\n")
        
        # Simulate gathering information
        print("Gathering required information...")
        
        # For demo purposes, we'll provide some sample information
        sample_info = generate_sample_info(task_completer.task_type)
        
        for key, value in sample_info.items():
            task_completer.gathered_info[key] = value
            print(f"- Added information for: {key}")
        
        # Mark task as complete
        task_completer.is_complete = True
        
        # Generate deliverable
        print("\nGenerating deliverable...")
        deliverable = task_completer.create_deliverable()
        
        # Display the deliverable
        print("\n" + "="*80)
        print("GENERATED DELIVERABLE")
        print("="*80 + "\n")
        print(deliverable)
        
        # Demonstrate agent integration
        print("\n" + "="*80)
        print("AGENT INTEGRATION DEMO")
        print("="*80 + "\n")
        
        print("Now demonstrating how this task would be handled by the agent...")
        try:
            response = await agent.run(selected_task)
        except Exception as e:
            import traceback
            print("\nAgent response:")
            print(f"[ERROR] Exception during agent run: {e}")
            print(traceback.format_exc())
            return
        print("\nAgent response:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error in demo: {str(e)}")

def generate_sample_info(task_type):
    """Generate sample information for the selected task type."""
    
    # Common sample data for all task types
    sample_data = {
        "research_report": {
            "topic": "Renewable Energy Sources",
            "background": "The global energy landscape is rapidly evolving with increasing focus on sustainable alternatives to fossil fuels.",
            "methodology": "This report is based on a comprehensive literature review and analysis of recent industry reports.",
            "findings": "Solar and wind energy have seen the most significant growth, with costs decreasing by over 70% in the last decade.",
            "analysis": "The transition to renewable energy is accelerating due to policy support, technological advancements, and economic factors.",
            "conclusion": "Renewable energy sources are becoming increasingly competitive with conventional energy sources and are poised to dominate the future energy landscape.",
            "recommendations": "Increased investment in energy storage technologies and grid modernization will be crucial for wider adoption.",
            "references": "1. International Energy Agency (2023). World Energy Outlook\n2. IRENA (2023). Renewable Capacity Statistics"
        },
        "technical_documentation": {
            "product_name": "DataProcessor",
            "version": "2.1.0",
            "purpose": "A Python library for efficient data processing, transformation, and analysis.",
            "installation": "```bash\npip install dataprocessor\n```",
            "usage_examples": "```python\nfrom dataprocessor import DataProcessor\n\n# Initialize with a dataset\ndp = DataProcessor('data.csv')\n\n# Clean and transform data\nclean_data = dp.clean().transform(['column1', 'column2'])\n\n# Analyze results\nresults = dp.analyze(method='correlation')\nprint(results)\n```",
            "api_endpoints": "### `DataProcessor.clean()`\nRemoves missing values and outliers from the dataset.\n\n### `DataProcessor.transform(columns)`\nApplies transformations to specified columns.",
            "parameters": "- `data_source` (str/DataFrame): The input data source, either a file path or pandas DataFrame.\n- `options` (dict, optional): Configuration options for processing."
        },
        "content_summary": {
            "source_content": "The latest climate change report highlights accelerating global warming trends...",
            "title": "Climate Change Report 2023",
            "key_points": "- Global temperatures have risen by 1.1°C since pre-industrial times\n- Sea levels are rising at an accelerated rate of 3.7mm per year\n- Extreme weather events have increased in frequency and intensity\n- Carbon emissions need to be reduced by 45% by 2030 to limit warming to 1.5°C",
            "length": "medium",
            "audience": "policymakers"
        },
        "data_analysis": {
            "title": "Q1 2023 Sales Analysis",
            "dataset": "Sales data from January to March 2023, including product categories, regions, and customer segments.",
            "variables": "Product category, region, customer segment, sales amount, profit margin",
            "analysis_goal": "Identify top-performing products and regions, and analyze sales trends over the quarter.",
            "methods": "Descriptive statistics, trend analysis, and segment comparison",
            "findings": "1. Electronics category showed 15% growth compared to Q1 2022\n2. Western region outperformed all other regions with 22% of total sales\n3. B2B segment contributed 65% of total profit despite representing only 40% of sales volume",
            "visualizations": "[Sales by Region Pie Chart]\n[Monthly Sales Trend Line Chart]\n[Product Category Comparison Bar Chart]",
            "recommendations": "1. Increase marketing investment in the Electronics category\n2. Develop targeted promotions for underperforming regions\n3. Expand B2B sales team to capitalize on high profit margins"
        },
        "blog_post": {
            "title": "5 Revolutionary Applications of Machine Learning in Healthcare",
            "topic": "Machine Learning in Healthcare",
            "target_audience": "Healthcare professionals and technology enthusiasts",
            "tone": "Informative and forward-looking",
            "key_points": "1. Early disease detection through pattern recognition\n2. Personalized treatment plans based on patient data\n3. Drug discovery acceleration\n4. Medical imaging analysis and diagnostics\n5. Predictive analytics for hospital resource management",
            "call_to_action": "Join our webinar on AI in Healthcare to learn more about implementing these technologies in your practice.",
            "seo_keywords": "machine learning healthcare, AI medical applications, healthcare technology, predictive healthcare"
        },
        "email_content": {
            "subject_line": "Introducing Our Revolutionary New Product: DataSync Pro",
            "recipient": "Valued Customers",
            "purpose": "Announce new product launch and drive initial sales",
            "key_message": "We're excited to announce the launch of DataSync Pro, our most powerful data synchronization tool yet. With 3x faster syncing speeds and enhanced security features, DataSync Pro will transform how you manage your data across platforms.",
            "tone": "Enthusiastic and professional",
            "call_to_action": "Click here to get 25% off your first year when you purchase DataSync Pro before June 30th."
        },
        "social_media": {
            "platform": "Twitter",
            "purpose": "Announce company anniversary",
            "tone": "Celebratory and grateful",
            "key_message": "Today marks 10 amazing years of innovation and growth! Thank you to our incredible customers, partners, and team members who have been part of this journey. We're just getting started! 🎉",
            "hashtags": "#10YearAnniversary #ThankYou #Innovation #CompanyMilestone",
            "call_to_action": "Share your favorite memory with us in the comments!"
        }
    }
    
    # Return the sample data for the selected task type or an empty dict if not found
    return sample_data.get(task_type, {})

if __name__ == "__main__":
    asyncio.run(demo_task_types())
