#!/usr/bin/env python3
import asyncio
import aiohttp
import json
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

async def test_ollama_api():
    """
    Test direct integration with the Ollama API to ensure we can get responses.
    Tests both the /api/chat and /api/generate endpoints.
    """
    # Ollama configuration
    base_url = "http://localhost:11434"  # Default Ollama URL without /v1
    model = "llama3.2:latest"  # Use the same model as in the config
    
    logger.info(f"Testing Ollama API with model: {model}")
    logger.info(f"Base URL: {base_url}")
    
    # Create a simple prompt
    prompt = "What are the main applications of artificial intelligence? Provide a brief summary."
    
    # Create a session with a longer timeout (5 minutes)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes in seconds
    
    # Test both API endpoints
    await test_chat_api(base_url, model, prompt, timeout)
    await test_generate_api(base_url, model, prompt, timeout)

async def test_chat_api(base_url, model, prompt, timeout):
    """Test the /api/chat endpoint"""
    logger.info("\n==== TESTING /api/chat ENDPOINT ====\n")
    
    # Format messages for Ollama chat API
    messages = [
        {"role": "user", "content": prompt}  # Changed from 'system' to 'user'
    ]
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama chat API")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/chat endpoint for chat completions
            chat_url = f"{base_url}/api/chat"
            logger.info(f"Sending request to: {chat_url}")
            
            # Log that we're waiting for the response
            logger.info("Waiting for Ollama to generate response (this may take a while)...")
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(chat_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Log that we're parsing the response
                    logger.info("Parsing response from Ollama...")
                    
                    # Parse the JSON response
                    result = await response.json()
                    logger.info(f"Received response from Ollama with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        logger.info(f"Successfully extracted content from Ollama response")
                        logger.info(f"Content length: {len(content)} characters")
                        
                        # Print a preview of the content
                        preview_length = min(500, len(content))
                        logger.info(f"Content preview:\n{content[:preview_length]}...")
                        
                        # Save the content to a file
                        output_dir = os.path.join(os.getcwd(), "workspace")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_path = os.path.join(output_dir, "ollama_chat_output.md")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        logger.info(f"Saved response to: {output_path}")
                        
                        # Verify the file was created
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            logger.info(f"Verified file exists at {output_path}, size: {file_size} bytes")
                        else:
                            logger.error(f"File was not created at {output_path}")
                    else:
                        logger.error(f"Unexpected response format from Ollama: {result}")
                else:
                    error_text = await response.text()
                    logger.error(f"Error from Ollama API: {response.status} - {error_text}")
    
    except Exception as e:
        logger.error(f"Exception during Ollama chat API request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def test_generate_api(base_url, model, prompt, timeout):
    """Test the /api/generate endpoint"""
    logger.info("\n==== TESTING /api/generate ENDPOINT ====\n")
    
    # Prepare the request payload for the generate API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama generate API")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/generate endpoint
            generate_url = f"{base_url}/api/generate"
            logger.info(f"Sending request to: {generate_url}")
            
            # Log that we're waiting for the response
            logger.info("Waiting for Ollama to generate response (this may take a while)...")
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(generate_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Log that we're parsing the response
                    logger.info("Parsing response from Ollama...")
                    
                    # Parse the JSON response
                    result = await response.json()
                    logger.info(f"Received response from Ollama with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'response' in result:
                        content = result['response']
                        logger.info(f"Successfully extracted content from Ollama response")
                        logger.info(f"Content length: {len(content)} characters")
                        
                        # Print a preview of the content
                        preview_length = min(500, len(content))
                        logger.info(f"Content preview:\n{content[:preview_length]}...")
                        
                        # Save the content to a file
                        output_dir = os.path.join(os.getcwd(), "workspace")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        output_path = os.path.join(output_dir, "ollama_generate_output.md")
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(content)
                        
                        logger.info(f"Saved response to: {output_path}")
                        
                        # Verify the file was created
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            logger.info(f"Verified file exists at {output_path}, size: {file_size} bytes")
                        else:
                            logger.error(f"File was not created at {output_path}")
                    else:
                        logger.error(f"Unexpected response format from Ollama: {result}")
                else:
                    error_text = await response.text()
                    logger.error(f"Error from Ollama API: {response.status} - {error_text}")
    
    except Exception as e:
        logger.error(f"Exception during Ollama generate API request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    except asyncio.TimeoutError:
        logger.error("Request timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Exception during Ollama API request: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Run the test
    asyncio.run(test_ollama_api())
