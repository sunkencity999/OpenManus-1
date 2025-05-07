import asyncio
import os
import logging
from datetime import datetime
from app.tool.browser_use_tool import BrowserUseTool
from app.llm import LLM

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_browser_ollama")

async def test_browser_extraction():
    """Test the browser extraction with Ollama integration."""
    print("Testing browser extraction with Ollama integration...")
    
    # Initialize LLM
    llm = LLM()
    print(f"LLM initialized with model={llm.model}, api_type={getattr(llm, 'api_type', 'unknown')}")
    print(f"Base URL: {llm.base_url}")
    
    # Initialize browser tool
    browser_tool = BrowserUseTool(llm=llm)
    print("Browser tool initialized with Ollama config")
    
    # URL to extract
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    # Goal for extraction
    goal = "Provide a summary of the key applications of artificial intelligence mentioned in the content."
    
    # Output filename
    output_file = "workspace/ai_applications_summary.md"
    
    try:
        # Create context
        context = {}
        
        # Navigate to the URL
        print(f"Navigating to {url}...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url=url
        )
        print(f"Navigation result: {result.output}")
        
        # Extract and save content
        print(f"Extracting content with goal: {goal}")
        result = await browser_tool.execute(
            context=context,
            action="extract_and_save",
            goal=goal,
            filename=output_file,
            format="markdown"
        )
        
        if result:
            print(f"Extraction successful!")
            print(f"Result output: {result.output}")
            
            # Check all possible locations for the output file
            possible_paths = [
                output_file,  # Absolute path
                os.path.join(os.getcwd(), output_file),  # Relative to current directory
                os.path.join(os.getcwd(), "workspace", output_file),  # In workspace directory
                os.path.join(os.getcwd(), "workspace", os.path.basename(output_file))  # Just the filename in workspace
            ]
            
            # List all files in workspace
            print("\nChecking workspace directory:")
            import glob
            workspace_files = glob.glob(os.path.join(os.getcwd(), "workspace", "*"))
            print(f"Files in workspace: {workspace_files}")
            
            # Check if raw_content directory exists
            raw_content_dir = os.path.join(os.getcwd(), "workspace", "raw_content")
            if os.path.exists(raw_content_dir):
                raw_files = glob.glob(os.path.join(raw_content_dir, "*"))
                print(f"Files in raw_content: {raw_files}")
            
            # Check each possible path
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"\nFound output file at: {path}")
                    with open(path, "r") as f:
                        content = f.read()
                    print(f"Content preview (first 200 chars):\n{content[:200]}...")
                    found = True
                    break
            
            if not found:
                print(f"\nWarning: Output file not found in any expected location")
        else:
            print("Extraction failed or returned no content")
    
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        print(f"Error during test: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up
        await browser_tool.cleanup()
        print("Browser closed")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Run the test
    asyncio.run(test_browser_extraction())
