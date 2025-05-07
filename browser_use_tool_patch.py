"""
Patch for the browser_use_tool.py file to fix the Ollama API integration.

Instructions:
1. Add this code to the BrowserUseTool class in browser_use_tool.py
2. Replace the existing _try_ollama_generate method with this implementation
3. Update the extract_and_save method to use the ollama_integration helper
"""

# Import the ollama_integration helper
from ollama_integration import get_ollama_response

# Replace the existing _try_ollama_generate method with this implementation
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

# Add this method to the BrowserUseTool class to simplify Ollama integration
async def _get_ollama_response(self, prompt, timeout=300):
    """
    Get a response from Ollama using our helper module.
    
    Args:
        prompt: Prompt to send to the model
        timeout: Timeout in seconds
        
    Returns:
        Response content or error message
    """
    from ollama_integration import get_ollama_response
    return await get_ollama_response(self.llm, prompt, timeout)

"""
In the extract_and_save method, replace the Ollama-specific code with:

try:
    # Get analysis response using our helper
    analysis_response = await self._get_ollama_response(analysis_prompt)
    
    # Log the response details
    logger.info(f"Analysis response type: {type(analysis_response)}")
    if analysis_response:
        logger.info(f"Analysis response length: {len(analysis_response)}")
    else:
        logger.info("Analysis response is None or empty")
    
    logger.info(f"Received analysis response, length: {len(analysis_response) if analysis_response else 0}")
    
except Exception as e:
    logger.error(f"Error during LLM analysis: {str(e)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    analysis_response = None
"""
