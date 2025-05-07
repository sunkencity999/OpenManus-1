import re
import os

def fix_browser_use_tool():
    """Fix the browser_use_tool.py file to properly work with Ollama."""
    file_path = "app/tool/browser_use_tool.py"
    
    # Read the original file
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix 1: Ensure base_url doesn't have /v1 suffix for Ollama
    content = re.sub(
        r'(base_url\s*=\s*self\.llm_config\.get\("base_url"\))',
        r'\1\n        # Remove /v1 suffix for Ollama\n        if base_url and base_url.endswith("/v1"):\n            base_url = base_url[:-3]',
        content
    )
    
    # Fix 2: Add the _try_ollama_generate method if it doesn't exist
    if "_try_ollama_generate" not in content:
        # Find the last method in the class
        match = re.search(r'(    async def [^(]+\([^)]+\):.*?)(\n\n)', content, re.DOTALL)
        if match:
            last_method = match.group(1)
            indent = "    "  # Class method indentation
            
            # Create the _try_ollama_generate method
            ollama_generate_method = f"""
{indent}async def _try_ollama_generate(self, base_url, model, prompt, timeout):
{indent}    \"\"\"Try to use the Ollama generate API as a fallback.\"\"\"
{indent}    self.logger.info(f"Falling back to Ollama generate API")
{indent}    try:
{indent}        async with aiohttp.ClientSession() as session:
{indent}            payload = {{
{indent}                "model": model,
{indent}                "prompt": prompt,
{indent}                "stream": False
{indent}            }}
{indent}            
{indent}            self.logger.info(f"Sending request to: {{base_url}}/api/generate")
{indent}            start_time = datetime.now()
{indent}            self.logger.info("Waiting for Ollama to generate response (this may take a while)...")
{indent}            
{indent}            async with session.post(f"{{base_url}}/api/generate", json=payload, timeout=timeout) as response:
{indent}                elapsed = (datetime.now() - start_time).total_seconds()
{indent}                self.logger.info(f"Received HTTP status: {{response.status}} after {{elapsed:.2f}} seconds")
{indent}                
{indent}                if response.status == 200:
{indent}                    result = await response.json()
{indent}                    self.logger.info(f"Received response from Ollama generate API with keys: {{result.keys()}}")
{indent}                    
{indent}                    if "response" in result:
{indent}                        content = result["response"]
{indent}                        self.logger.info(f"Successfully extracted content from Ollama generate response, length: {{len(content)}}")
{indent}                        return content
{indent}                    else:
{indent}                        self.logger.warning(f"Unexpected response structure from Ollama generate API: {{result.keys()}}")
{indent}                else:
{indent}                    self.logger.error(f"Error from Ollama generate API: {{response.status}}")
{indent}    
{indent}    except Exception as e:
{indent}        self.logger.error(f"Exception during Ollama generate API request: {{str(e)}}")
{indent}    
{indent}    return None"""
            
            # Insert the new method after the last method
            content = content.replace(last_method + "\n\n", last_method + ollama_generate_method + "\n\n")
    
    # Fix 3: Update the extract_and_save method to use both Ollama endpoints
    # Find the LLM request part and update it to try both endpoints
    llm_request_pattern = r'(async with session\.post\(.*?response\.status == 200.*?)else:.*?self\.logger\.error\(f"Error from LLM API: {response\.status}"\)'
    llm_request_replacement = r'\1else:\n                    self.logger.error(f"Error from LLM API: {response.status}")\n                    \n                    # Try fallback to Ollama generate endpoint if using Ollama\n                    if self.llm_config.get("api_type") == "ollama" and base_url:\n                        fallback_response = await self._try_ollama_generate(base_url, model, prompt, timeout)\n                        if fallback_response:\n                            return fallback_response'
    
    content = re.sub(llm_request_pattern, llm_request_replacement, content, flags=re.DOTALL)
    
    # Write the updated content back to the file
    with open(file_path, "w") as f:
        f.write(content)
    
    print("Successfully updated browser_use_tool.py to work with Ollama")
    print("Key changes:")
    print("1. Removed /v1 suffix from base_url for Ollama")
    print("2. Added _try_ollama_generate method as a fallback")
    print("3. Updated extract_and_save to use both Ollama endpoints")

if __name__ == "__main__":
    fix_browser_use_tool()
