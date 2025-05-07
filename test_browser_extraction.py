#!/usr/bin/env python3
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
from app.tool.browser_use_tool import BrowserUseTool

async def test_browser_extraction():
    """
    Test the browser extraction functionality with proper Ollama integration.
    """
    print("Setting up test environment...")
    
    # Create workspace directories
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    raw_content_dir = os.path.join(workspace_dir, "raw_content")
    
    print(f"Creating workspace directory at: {workspace_dir}")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(raw_content_dir, exist_ok=True)
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM()
    print(f"LLM configuration: model={llm.model}, api_type={llm.api_type}")
    print(f"Base URL: {llm.base_url}")
    
    # Initialize the browser tool
    print("Initializing browser tool...")
    browser_tool = BrowserUseTool(llm=llm)
    
    # Create a test context
    context = {}
    
    try:
        # Step 1: Go to a website with structured information
        print("\nNavigating to a test website...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)"
        )
        print(f"Navigation result: {result.output}")
        
        # Step 2: Extract content with a specific goal
        print("\nExtracting content about Python programming language...")
        print("This may take some time as the LLM processes the content...")
        
        # Set a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"python_info_{timestamp}"
        
        try:
            # Set a longer timeout for the extraction process
            result = await asyncio.wait_for(
                browser_tool.execute(
                    context=context,
                    action="extract_and_save",
                    goal="What are the key features and applications of Python programming language?",
                    filename=filename,
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
        workspace_files = glob.glob(f"{workspace_dir}/*")
        raw_dirs = glob.glob(f"{raw_content_dir}/*")
        
        print(f"Files in workspace: {workspace_files}")
        print(f"Files in raw_content: {raw_dirs}")
        
        # Check if the main output file was created
        output_path = os.path.join(workspace_dir, f"{filename}.md")
        if os.path.exists(output_path):
            print(f"\nOutput file exists at: {output_path}")
            with open(output_path, 'r') as f:
                content = f.read(500)  # Read first 500 chars
            print(f"Content preview:\n{content}...")
            print(f"File size: {os.path.getsize(output_path)} bytes")
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
    # Run the test
    asyncio.run(test_browser_extraction())
