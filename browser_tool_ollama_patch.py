#!/usr/bin/env python3
"""
Minimal patch for the browser_use_tool.py file to fix Ollama API integration.
This script will add the _try_ollama_generate method to the BrowserUseTool class
and modify the extract_and_save method to properly handle Ollama API requests.
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

from app.tool.browser_use_tool import BrowserUseTool

# Define the _try_ollama_generate method
async def _try_ollama_generate(self, base_url, model, prompt, timeout):
    """Try the Ollama /api/generate endpoint as a fallback."""
    logger.info("Using Ollama /api/generate endpoint as fallback...")
    
    # Prepare the request payload for the generate API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama generate API with model: {model}")
    
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/generate endpoint
            generate_url = f"{base_url}/api/generate"
            logger.info(f"Sending request to: {generate_url}")
            
            # Log that we're waiting for the response
            logger.info("Waiting for Ollama to generate response (this may take a while)...")
            from datetime import datetime
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(generate_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Log that we're parsing the response
                    logger.info("Parsing response from Ollama generate API...")
                    
                    # Parse the JSON response
                    result = await response.json()
                    logger.info(f"Received response from Ollama generate API with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'response' in result:
                        content = result['response']
                        logger.info(f"Successfully extracted content from Ollama generate response, length: {len(content)}")
                        return content
                    else:
                        logger.error(f"Unexpected response format from Ollama generate API: {result}")
                        return f"Error: Unexpected response format from Ollama generate API: {result}"
                else:
                    error_text = await response.text()
                    logger.error(f"Error from Ollama generate API: {response.status} - {error_text}")
                    return f"Error from Ollama generate API: {response.status} - {error_text}"
    
    except Exception as e:
        logger.error(f"Exception during Ollama generate API request: {str(e)}")
        return f"Error communicating with Ollama generate API: {str(e)}"

# Add the method to the BrowserUseTool class
BrowserUseTool._try_ollama_generate = _try_ollama_generate

# Print confirmation
print("Added _try_ollama_generate method to BrowserUseTool class")
print("You can now use the extract_and_save method with Ollama API integration")
print("Remember to adjust the base_url by removing '/v1' if present")
