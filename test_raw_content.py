#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_raw_content_save():
    """
    Test the raw content saving functionality directly without using the LLM.
    This will help us understand if the file saving part works correctly.
    """
    print("Testing raw content saving...")
    
    # Create the workspace and raw_content directories
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    raw_content_dir = os.path.join(workspace_dir, "raw_content")
    
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(raw_content_dir, exist_ok=True)
    
    print(f"Created directories: {workspace_dir} and {raw_content_dir}")
    
    # Generate a timestamp for the raw content file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sample HTML content
    html_content = """<!DOCTYPE html>
    <html>
    <head>
        <title>Test HTML Content</title>
    </head>
    <body>
        <h1>Artificial Intelligence</h1>
        <p>This is a test HTML page about artificial intelligence.</p>
        <p>AI is the simulation of human intelligence by machines.</p>
    </body>
    </html>"""
    
    # Sample markdown content
    markdown_content = """# Artificial Intelligence
    
    This is a test markdown document about artificial intelligence.
    
    AI is the simulation of human intelligence by machines.
    """
    
    # Save the raw HTML
    raw_filename = f"raw_{timestamp}.html"
    raw_file_path = os.path.join(raw_content_dir, raw_filename)
    
    try:
        with open(raw_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully saved raw HTML to {raw_file_path}")
    except Exception as e:
        print(f"Error saving raw HTML: {str(e)}")
    
    # Save the raw markdown
    raw_md_filename = f"raw_{timestamp}.md"
    raw_md_path = os.path.join(raw_content_dir, raw_md_filename)
    
    try:
        with open(raw_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Successfully saved raw markdown to {raw_md_path}")
    except Exception as e:
        print(f"Error saving raw markdown: {str(e)}")
    
    # Create an extracted content file
    extracted_content = {
        "text": "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        "metadata": {
            "source": "Test Source",
            "extraction_method": "Direct Test",
            "extraction_time": datetime.now().isoformat()
        }
    }
    
    # Format as markdown
    formatted_content = f"""# What is AI?

{extracted_content['text']}

*Source: {extracted_content['metadata']['source']}*

*Extraction method: {extracted_content['metadata']['extraction_method']}*
"""
    
    # Save the extracted content
    extracted_file_path = os.path.join(workspace_dir, "test_extraction.md")
    
    try:
        with open(extracted_file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        print(f"Successfully saved extracted content to {extracted_file_path}")
        
        # Read back the content to verify
        with open(extracted_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Extracted content preview: {content[:100]}...")
    except Exception as e:
        print(f"Error saving extracted content: {str(e)}")
    
    # Check if all files were created
    print("\nChecking created files:")
    print(f"Raw HTML exists: {os.path.exists(raw_file_path)}")
    print(f"Raw markdown exists: {os.path.exists(raw_md_path)}")
    print(f"Extracted content exists: {os.path.exists(extracted_file_path)}")
    
    return True

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_raw_content_save())
