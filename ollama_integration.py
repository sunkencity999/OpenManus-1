"""
Ollama Integration Helper for Browser Use Tool

This module provides helper functions for integrating with Ollama API
directly, without relying on the OpenAI client.
"""
import logging
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

async def try_ollama_chat(base_url, model, prompt, timeout=300):
    """
    Try to use the Ollama /api/chat endpoint.
    
    Args:
        base_url: Base URL for Ollama API (without /v1)
        model: Model name to use
        prompt: Prompt to send to the model
        timeout: Timeout in seconds
        
    Returns:
        Response content or error message
    """
    # Format messages for Ollama chat API - use 'user' role instead of 'system'
    ollama_messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Prepare the request payload
    chat_payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama chat API with model: {model}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Use the /api/chat endpoint for chat completions
            chat_url = f"{base_url}/api/chat"
            logger.info(f"Sending request to: {chat_url}")
            logger.info("Waiting for Ollama to generate response (this may take a while)...")
            
            start_time = datetime.now()
            async with session.post(chat_url, json=chat_payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Received response from Ollama chat API with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        logger.info(f"Successfully extracted content from Ollama chat response, length: {len(content)}")
                        return content
                    else:
                        logger.warning(f"Unexpected response format from Ollama chat API: {result}")
                        return None
                else:
                    error_text = await response.text()
                    logger.warning(f"Error from Ollama chat API: {response.status} - {error_text}")
                    return None
    
    except Exception as e:
        logger.warning(f"Exception during Ollama chat API request: {str(e)}")
        return None

async def try_ollama_generate(base_url, model, prompt, timeout=300):
    """
    Try the Ollama /api/generate endpoint as a fallback.
    
    Args:
        base_url: Base URL for Ollama API (without /v1)
        model: Model name to use
        prompt: Prompt to send to the model
        timeout: Timeout in seconds
        
    Returns:
        Response content or error message
    """
    logger.info("Using Ollama /api/generate endpoint as fallback...")
    
    # Prepare the request payload for the generate API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama generate API with model: {model}")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
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

async def get_ollama_response(llm, prompt, timeout=300):
    """
    Get a response from Ollama using either the chat or generate API.
    
    Args:
        llm: LLM instance with model and base_url
        prompt: Prompt to send to the model
        timeout: Timeout in seconds
        
    Returns:
        Response content or error message
    """
    # Determine if we're using Ollama
    is_ollama = False
    if hasattr(llm, 'api_type') and llm.api_type.lower() == 'ollama':
        is_ollama = True
    elif hasattr(llm, 'base_url') and 'ollama' in llm.base_url.lower():
        is_ollama = True
    
    if not is_ollama:
        logger.info("Not using Ollama, falling back to standard LLM interface")
        # Use the standard LLM interface for non-Ollama models
        try:
            analysis_messages = [{"role": "system", "content": prompt}]
            return await llm.ask(analysis_messages, stream=False)
        except Exception as e:
            logger.error(f"Error using standard LLM interface: {str(e)}")
            return f"Error using standard LLM interface: {str(e)}"
    
    # Get the base URL without /v1 if present
    base_url = llm.base_url
    if base_url.endswith('/v1'):
        base_url = base_url[:-3]
    logger.info(f"Ollama base URL: {base_url}")
    
    # Try the chat API first
    response = await try_ollama_chat(base_url, llm.model, prompt, timeout)
    
    # If chat API fails, try the generate API
    if response is None:
        logger.info("Chat API failed, trying generate API")
        response = await try_ollama_generate(base_url, llm.model, prompt, timeout)
    
    return response
