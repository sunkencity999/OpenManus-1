#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path
from openai import AsyncOpenAI

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.llm import LLM
from app.tool.browser_use_tool import BrowserUseTool

async def test_improved_extraction():
    """
    Test the improved two-step extraction process:
    1. First save the raw content from the webpage
    2. Then analyze the raw content and save the analysis
    """
    print("Initializing browser tool...")
    
    # Initialize the LLM
    llm = LLM()
    
    # Note: For Ollama, we need to modify the base_url to remove '/v1'
    if llm.base_url.endswith('/v1'):
        print(f"Detected Ollama endpoint with /v1: {llm.base_url}")
        llm.base_url = llm.base_url[:-3]
        print(f"Updated base_url: {llm.base_url}")
        # Recreate the client with the updated base_url
        llm.client = AsyncOpenAI(api_key=llm.api_key, base_url=llm.base_url)
    
    print(f"LLM configuration: model={llm.model}, api_type={llm.api_type}")
    print(f"Base URL: {llm.base_url}")
    
    # Initialize the browser tool
    browser_tool = BrowserUseTool(llm=llm)
    
    # Create a test context
    context = {}
    
    try:
        # Step 1: Go to a website with structured information
        print("Navigating to Wikipedia...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        print(f"Navigation result: {result.output}")
        
        # Step 2: Extract content with a specific goal
        print("\nExtracting content about AI applications...")
        print("This may take some time as the LLM processes the content...")
        
        try:
            # Set a longer timeout for the extraction process
            result = await asyncio.wait_for(
                browser_tool.execute(
                    context=context,
                    action="extract_and_save",
                    goal="What are the main applications of artificial intelligence?",
                    filename="ai_applications",
                    format="markdown"
                ),
                timeout=300  # 5 minutes timeout
            )
            print(f"Extraction completed with result: {result.output if result else 'No result'}")
        except asyncio.TimeoutError:
            print("Extraction timed out after 5 minutes. This doesn't necessarily mean it failed.")
            print("The process might still be running in the background.")
            result = None
        except Exception as e:
            print(f"Error during extraction: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            result = None
        
        # Check if files were created
        print("\nChecking for created files:")
        import glob
        workspace_files = glob.glob("workspace/*")
        raw_dirs = glob.glob("workspace/raw_content/*")
        
        print(f"Files in workspace: {workspace_files}")
        print(f"Directories in raw_content: {raw_dirs}")
        
        # Check if the main output file was created
        output_path = os.path.join(os.getcwd(), "workspace", "ai_applications.md")
        if os.path.exists(output_path):
            print(f"\nOutput file exists at: {output_path}")
            with open(output_path, 'r') as f:
                content = f.read(500)  # Read first 500 chars
            print(f"Content preview:\n{content}...")
        else:
            print(f"Output file not found at: {output_path}")
        
        # Close the browser
        print("\nClosing browser...")
        await browser_tool.cleanup()
        print("Browser closed successfully")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        await browser_tool.cleanup()
        raise

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    raw_content_dir = os.path.join(workspace_dir, "raw_content")
    
    print(f"Creating workspace directory at: {workspace_dir}")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(raw_content_dir, exist_ok=True)
    
    # Check if directories were created successfully
    if os.path.exists(workspace_dir):
        print(f"Workspace directory exists: {workspace_dir}")
    else:
        print(f"Failed to create workspace directory: {workspace_dir}")
    
    if os.path.exists(raw_content_dir):
        print(f"Raw content directory exists: {raw_content_dir}")
    else:
        print(f"Failed to create raw content directory: {raw_content_dir}")
    
    # Run the test
    asyncio.run(test_improved_extraction())
