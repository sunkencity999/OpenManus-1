#!/usr/bin/env python3
"""
Test script for the extract_analyze_save action in BrowserUseTool.

This script demonstrates how to use the new extract_analyze_save action
to extract content from a webpage, analyze it using Ollama, and save the
result to a file.
"""
import asyncio
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_extract_analyze_save")

# Import required modules
from app.tool.browser_use_tool import BrowserUseTool
from app.llm import LLM

async def test_extract_analyze_save():
    """Test the extract_analyze_save action in BrowserUseTool."""
    print("Testing extract_analyze_save action in BrowserUseTool...")
    
    # Initialize LLM
    llm = LLM()
    print(f"LLM initialized with model={llm.model}, api_type={getattr(llm, 'api_type', 'unknown')}")
    print(f"Base URL: {llm.base_url}")
    
    # Fix the base_url if it ends with /v1 (as per memory)
    if llm.base_url.endswith('/v1'):
        original_base_url = llm.base_url
        llm.base_url = llm.base_url[:-3]
        print(f"Removed /v1 suffix from base_url: {original_base_url} -> {llm.base_url}")
    
    # Initialize browser tool
    browser_tool = BrowserUseTool(llm=llm)
    print("Browser tool initialized")
    
    # URL to extract
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    # Goal for extraction
    goal = "Provide a summary of the key applications of artificial intelligence mentioned in the content."
    
    # Output filename
    output_file = f"ai_applications_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    try:
        # Create context
        context = {}
        
        # Navigate to the URL
        print(f"Navigating to {url}...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url=url
        )
        print(f"Navigation result: {result.output}")
        
        # Extract, analyze, and save content
        print(f"Extracting, analyzing, and saving content with goal: {goal}")
        result = await browser_tool.execute(
            context=context,
            action="extract_analyze_save",
            goal=goal,
            filename=output_file,
            format="markdown"
        )
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Success: {result.output}")
            
            # Check if the output file exists
            workspace_dir = os.path.join(os.getcwd(), "workspace")
            output_path = os.path.join(workspace_dir, output_file)
            
            if os.path.exists(output_path):
                print(f"Output file exists at: {output_path}")
                
                # Read the first few lines of the file
                with open(output_path, "r") as f:
                    content = f.read(500)  # Read first 500 characters
                
                print(f"Content preview:\n{content}...")
            else:
                print(f"Output file not found at: {output_path}")
    
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        print(f"Error during test: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up
        await browser_tool.cleanup()
        print("Browser closed")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Run the test
    asyncio.run(test_extract_analyze_save())
