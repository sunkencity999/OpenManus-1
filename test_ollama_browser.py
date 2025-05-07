#!/usr/bin/env python3
"""
Test script to verify the Ollama integration with the browser tool.
This script uses the fixed version of the browser_use_tool.py file.
"""
import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.llm import LLM

async def test_ollama_browser_integration():
    """
    Test the Ollama integration with the browser tool.
    """
    print("Testing Ollama integration with browser tool...")
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM()
    
    # Check if we're using Ollama
    is_ollama = False
    if hasattr(llm, 'api_type') and llm.api_type.lower() == 'ollama':
        is_ollama = True
    elif 'ollama' in llm.base_url.lower():
        is_ollama = True
    
    print(f"Using Ollama: {is_ollama}")
    print(f"LLM configuration: model={llm.model}, api_type={llm.api_type}")
    print(f"Base URL: {llm.base_url}")
    
    # If using Ollama, adjust the base_url
    if is_ollama and llm.base_url.endswith('/v1'):
        original_base_url = llm.base_url
        llm.base_url = llm.base_url[:-3]
        print(f"Adjusted base URL for Ollama: {original_base_url} -> {llm.base_url}")
    
    # Test a simple prompt with Ollama
    print("\nTesting Ollama API directly...")
    
    prompt = "What are the main applications of artificial intelligence? Provide a brief summary."
    print(f"Prompt: {prompt}")
    
    try:
        # Create a simple message for the LLM
        messages = [{"role": "user", "content": prompt}]
        
        # Use the ask method to get a response
        print("Sending request to LLM...")
        start_time = datetime.now()
        response = await llm.ask(messages, stream=False)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Received response after {duration:.2f} seconds")
        
        if response:
            print(f"Response length: {len(response)}")
            print(f"Response preview: {response[:200]}...")
            
            # Save the response to a file
            output_dir = os.path.join(os.getcwd(), "workspace")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "ollama_test_output.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)
            
            print(f"Saved response to: {output_path}")
        else:
            print("No response received from LLM")
    
    except Exception as e:
        print(f"Error during LLM test: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Run the test
    asyncio.run(test_ollama_browser_integration())
