#!/usr/bin/env python3
"""
Test script for the Ollama integration helper module.
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
from ollama_integration import get_ollama_response

async def test_ollama_helper():
    """
    Test the Ollama integration helper functions.
    """
    print("Testing Ollama integration helper...")
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM()
    
    # Print LLM configuration
    print(f"LLM configuration: model={llm.model}, api_type={getattr(llm, 'api_type', 'unknown')}")
    print(f"Base URL: {llm.base_url}")
    
    # Create a simple prompt
    prompt = "What are the main applications of artificial intelligence? Provide a brief summary."
    print(f"Prompt: {prompt}")
    
    # Test the get_ollama_response function
    print("\nTesting get_ollama_response function...")
    try:
        response = await get_ollama_response(llm, prompt)
        
        if response:
            print(f"Response length: {len(response)}")
            print(f"Response preview: {response[:200]}...")
            
            # Save the response to a file
            output_dir = os.path.join(os.getcwd(), "workspace")
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, "ollama_helper_output.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)
            
            print(f"Saved response to: {output_path}")
        else:
            print("No response received")
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Run the test
    asyncio.run(test_ollama_helper())
