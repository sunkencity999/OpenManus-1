#!/usr/bin/env python3
"""
Test script for browser extraction with fixed Ollama integration.
This script uses the Ollama integration helper to extract and save content from a webpage.
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.llm import LLM
from app.tool.browser_use_tool import BrowserUseTool
from ollama_integration import get_ollama_response

# Monkey patch the BrowserUseTool class to use our Ollama integration helper
async def _get_ollama_response(self, prompt, timeout=300):
    """Get a response from Ollama using our helper module."""
    return await get_ollama_response(self.llm, prompt, timeout)

BrowserUseTool._get_ollama_response = _get_ollama_response

async def test_browser_extraction():
    """
    Test the browser extraction functionality with fixed Ollama integration.
    """
    print("Testing browser extraction with fixed Ollama integration...")
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM()
    
    # Print LLM configuration
    print(f"LLM configuration: model={llm.model}, api_type={getattr(llm, 'api_type', 'unknown')}")
    print(f"Base URL: {llm.base_url}")
    
    # Initialize the browser tool
    print("Initializing browser tool...")
    browser_tool = BrowserUseTool(llm=llm)
    
    # Create a test context
    context = {}
    
    try:
        # Step 1: Go to a website
        print("\nNavigating to Wikipedia...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        print(f"Navigation result: {result.output}")
        
        # Step 2: Extract content using our patched method
        print("\nExtracting content about AI...")
        
        # Get the current page
        page = await context.get_current_page()
        raw_html = await page.content()
        page_url = page.url
        page_title = await page.title()
        
        print(f"Page title: {page_title}")
        print(f"Page URL: {page_url}")
        print(f"HTML content length: {len(raw_html)}")
        
        # Create a simple prompt for extraction
        extraction_goal = "What is artificial intelligence and its main applications?"
        prompt = f"""
        You are an expert content analyzer. Your task is to extract and summarize information from the following webpage.
        
        GOAL: {extraction_goal}
        
        PAGE TITLE: {page_title}
        PAGE URL: {page_url}
        
        Please provide a detailed analysis focusing specifically on this goal. Include all relevant information, facts, and figures.
        Structure your response with appropriate headings and sections. Be comprehensive but concise.
        
        CONTENT:
        {raw_html[:5000]}... [Content truncated due to length]
        """
        
        print("\nSending extraction prompt to Ollama...")
        
        # Use our patched method to get a response
        analysis_response = await browser_tool._get_ollama_response(prompt)
        
        print(f"Response length: {len(analysis_response) if analysis_response else 0}")
        if analysis_response:
            print(f"Response preview: {analysis_response[:200]}...")
            
            # Save the response to a file
            output_dir = os.path.join(os.getcwd(), "workspace")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "browser_extraction_fixed.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(analysis_response)
            
            print(f"Saved response to: {output_path}")
        else:
            print("No response received")
        
        # Step 3: Close the browser
        print("\nClosing browser...")
        await browser_tool.cleanup()
        print("Browser closed successfully")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        await browser_tool.cleanup()
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Run the test
    asyncio.run(test_browser_extraction())
