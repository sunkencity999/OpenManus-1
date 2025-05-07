#!/usr/bin/env python3
"""
Test script to directly test the Ollama API integration without relying on the LLM class.
This will help us verify that our approach in the browser_use_tool.py file is correct.
"""
import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import config to get Ollama settings
from app.config import config

async def test_direct_ollama():
    """
    Test direct integration with the Ollama API.
    """
    # Ollama configuration - based on memory about LocalManus using Ollama
    base_url = "http://localhost:11434"  # Default Ollama URL without /v1
    
    # Get model from config if available
    model = "llama3.2:latest"  # Default model
    if hasattr(config, 'llm') and hasattr(config.llm, 'model'):
        model = config.llm.model
    
    print(f"Testing direct Ollama API with model: {model}")
    print(f"Base URL: {base_url}")
    
    # Create a simple prompt
    prompt = "What are the main applications of artificial intelligence? Provide a brief summary."
    print(f"Prompt: {prompt}")
    
    # Test both API endpoints
    await test_chat_api(base_url, model, prompt)
    await test_generate_api(base_url, model, prompt)

async def test_chat_api(base_url, model, prompt):
    """Test the Ollama /api/chat endpoint."""
    print("\n=== TESTING OLLAMA CHAT API ===")
    
    # Format messages for Ollama chat API - use 'user' role instead of 'system'
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    print(f"Sending request to Ollama chat API")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        import aiohttp
        # Create a session with a longer timeout (5 minutes)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes in seconds
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/chat endpoint for chat completions
            chat_url = f"{base_url}/api/chat"
            print(f"Sending request to: {chat_url}")
            
            # Log that we're waiting for the response
            print("Waiting for Ollama to generate response (this may take a while)...")
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(chat_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                print(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Parse the JSON response
                    result = await response.json()
                    print(f"Received response from Ollama with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        print(f"Successfully extracted content from Ollama response")
                        print(f"Content length: {len(content)} characters")
                        
                        # Print a preview of the content
                        preview_length = min(200, len(content))
                        print(f"Content preview:\n{content[:preview_length]}...")
                        
                        # Save the content to a file
                        output_dir = os.path.join(os.getcwd(), "workspace")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_path = os.path.join(output_dir, "ollama_chat_output.md")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        print(f"Saved response to: {output_path}")
                    else:
                        print(f"Unexpected response format from Ollama: {result}")
                else:
                    error_text = await response.text()
                    print(f"Error from Ollama API: {response.status} - {error_text}")
    
    except Exception as e:
        print(f"Exception during Ollama API request: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def test_generate_api(base_url, model, prompt):
    """Test the Ollama /api/generate endpoint."""
    print("\n=== TESTING OLLAMA GENERATE API ===")
    
    # Prepare the request payload for the generate API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to Ollama generate API")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        import aiohttp
        # Create a session with a longer timeout (5 minutes)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes in seconds
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/generate endpoint
            generate_url = f"{base_url}/api/generate"
            print(f"Sending request to: {generate_url}")
            
            # Log that we're waiting for the response
            print("Waiting for Ollama to generate response (this may take a while)...")
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(generate_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                print(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Parse the JSON response
                    result = await response.json()
                    print(f"Received response from Ollama with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'response' in result:
                        content = result['response']
                        print(f"Successfully extracted content from Ollama response")
                        print(f"Content length: {len(content)} characters")
                        
                        # Print a preview of the content
                        preview_length = min(200, len(content))
                        print(f"Content preview:\n{content[:preview_length]}...")
                        
                        # Save the content to a file
                        output_dir = os.path.join(os.getcwd(), "workspace")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_path = os.path.join(output_dir, "ollama_generate_output.md")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        print(f"Saved response to: {output_path}")
                    else:
                        print(f"Unexpected response format from Ollama: {result}")
                else:
                    error_text = await response.text()
                    print(f"Error from Ollama API: {response.status} - {error_text}")
    
    except Exception as e:
        print(f"Exception during Ollama API request: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Run the test
    asyncio.run(test_direct_ollama())
