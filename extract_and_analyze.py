#!/usr/bin/env python3
"""
Extract and analyze content from a webpage using Ollama.

This script demonstrates the core workflow:
1. Pull raw data from a webpage and save it
2. Pass the data to the LLM via the Ollama endpoint
3. Receive the analysis and store it
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

from app.config import config
from app.llm import LLM
from app.tool.browser_use_tool import BrowserUseTool
import ollama_integration

async def extract_and_analyze(url, goal, output_filename, format="markdown"):
    """
    Extract and analyze content from a webpage using Ollama.
    
    Args:
        url: URL of the webpage to extract content from
        goal: Goal of the extraction
        output_filename: Filename to save the analysis to
        format: Format of the output file (markdown, json, html, text)
    """
    print(f"Extracting and analyzing content from {url}")
    print(f"Goal: {goal}")
    
    # Initialize the LLM
    print("Initializing LLM...")
    llm = LLM()
    
    # Print LLM configuration
    print(f"LLM configuration: model={llm.model}, api_type={getattr(llm, 'api_type', 'unknown')}")
    print(f"Base URL: {llm.base_url}")
    
    # Adjust base_url for Ollama if needed
    if llm.base_url.endswith('/v1'):
        original_base_url = llm.base_url
        llm.base_url = llm.base_url[:-3]
        print(f"Adjusted base URL for Ollama: {original_base_url} -> {llm.base_url}")
    
    # Initialize the browser tool
    print("Initializing browser tool...")
    browser_tool = BrowserUseTool(llm=llm)
    
    # Create a test context
    context = {}
    
    try:
        # Step 1: Go to the website
        print(f"\nNavigating to {url}...")
        result = await browser_tool.execute(
            context=context,
            action="go_to_url",
            url=url
        )
        print(f"Navigation result: {result.output}")
        
        # Step 2: Extract raw content from the webpage
        print("\nExtracting raw content...")
        
        # Get the current page
        page = await browser_tool.context.get_current_page()
        raw_html = await page.content()
        page_url = page.url
        page_title = await page.title()
        
        print(f"Page title: {page_title}")
        print(f"Page URL: {page_url}")
        print(f"HTML content length: {len(raw_html)}")
        
        # Create workspace directory structure
        workspace_dir = os.path.join(os.getcwd(), "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Create a raw content directory
        raw_content_dir = os.path.join(workspace_dir, "raw_content")
        os.makedirs(raw_content_dir, exist_ok=True)
        
        # Generate a timestamp for the raw content file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a directory for this specific extraction
        extraction_dir = os.path.join(raw_content_dir, f"extraction_{timestamp}")
        os.makedirs(extraction_dir, exist_ok=True)
        
        # Save the raw HTML
        raw_html_path = os.path.join(extraction_dir, "raw.html")
        with open(raw_html_path, 'w', encoding='utf-8') as f:
            f.write(raw_html)
        print(f"Saved raw HTML content to {raw_html_path}")
        
        # Convert HTML to markdown for better readability
        import markdownify
        content = markdownify.markdownify(raw_html)
        
        # Save the raw markdown
        raw_md_path = os.path.join(extraction_dir, "raw.md")
        with open(raw_md_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved raw markdown content to {raw_md_path}")
        
        # Save metadata about the extraction
        metadata = {
            "url": page_url,
            "title": page_title,
            "timestamp": timestamp,
            "goal": goal,
            "extraction_time": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(extraction_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved extraction metadata to {metadata_path}")
        
        # Step 3: Analyze the raw content using Ollama
        print("\nAnalyzing content using Ollama...")
        
        # Create a prompt for analysis
        analysis_prompt = f"""
        You are an expert content analyzer. Your task is to extract and summarize information from the following webpage content.
        
        GOAL: {goal}
        
        PAGE TITLE: {page_title}
        PAGE URL: {page_url}
        
        Please provide a detailed analysis focusing specifically on this goal. Include all relevant information, facts, figures, and quotes.
        Structure your response with appropriate headings and sections. Be comprehensive but concise.
        
        CONTENT:
        {content[:5000] if len(content) > 5000 else content}
        
        {"[Content truncated due to length]" if len(content) > 5000 else ""}
        """
        
        # Get a response from Ollama
        analysis_response = await ollama_integration.get_ollama_response(llm, analysis_prompt)
        
        print(f"Analysis response length: {len(analysis_response) if analysis_response else 0}")
        if analysis_response:
            print(f"Analysis response preview: {analysis_response[:200]}...")
            
            # Save the raw analysis
            analysis_path = os.path.join(extraction_dir, "analysis.md")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write(analysis_response)
            print(f"Saved raw analysis to {analysis_path}")
            
            # Step 4: Format the analysis based on the requested format
            print(f"\nFormatting analysis in {format} format...")
            
            # Ensure filename has the correct extension based on format
            file_extension = {
                "json": ".json",
                "html": ".html",
                "text": ".txt",
                "markdown": ".md"
            }.get(format, ".md")
            
            if not output_filename.endswith(file_extension):
                output_filename = output_filename + file_extension
            
            # Format the content based on the requested format
            if format == "json":
                # Create a JSON structure with the analysis and metadata
                formatted_data = {
                    "goal": goal,
                    "url": page_url,
                    "title": page_title,
                    "analysis": analysis_response,
                    "metadata": {
                        "extraction_time": datetime.now().isoformat(),
                        "source": page_url,
                        "raw_content_path": extraction_dir
                    }
                }
                formatted_content = json.dumps(formatted_data, indent=2)
                
            elif format == "html":
                # Create a nicely formatted HTML document
                formatted_content = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>{goal} - {page_title}</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 800px; margin: 0 auto; }}
                        h1 {{ color: #333; }}
                        .content {{ margin: 20px 0; }}
                        .source {{ color: #666; font-style: italic; }}
                        .extraction-info {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8em; color: #999; }}
                    </style>
                </head>
                <body>
                    <h1>{goal}</h1>
                    <h2>{page_title}</h2>
                    <div class="content">
                        {analysis_response}
                    </div>
                    <p class="source">Source: <a href="{page_url}">{page_url}</a></p>
                    <div class="extraction-info">
                        Extraction time: {datetime.now().isoformat()}<br>
                        Raw content available at: {extraction_dir}
                    </div>
                </body>
                </html>"""
                
            elif format == "text":
                # Create a simple text format
                formatted_content = f"""GOAL: {goal}

TITLE: {page_title}

SOURCE: {page_url}

{analysis_response}

Extraction time: {datetime.now().isoformat()}
Raw content available at: {extraction_dir}
"""
                
            else:  # Default to markdown
                # Create a nicely formatted markdown document
                formatted_content = f"""# {goal}

## {page_title}

{analysis_response}

---

*Source: [{page_url}]({page_url})*

*Extraction time: {datetime.now().isoformat()}*

*Raw content available at: `{extraction_dir}`*
"""
            
            # Write the formatted content to the output file
            output_path = os.path.join(workspace_dir, output_filename)
            
            # Create the workspace directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the content to the file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)
            
            print(f"Saved formatted content to {output_path}")
            
            # Verify the file was created
            if os.path.exists(output_path):
                print(f"Verified file exists at {output_path}")
                file_size = os.path.getsize(output_path)
                print(f"File size: {file_size} bytes")
            else:
                print(f"File was not created at {output_path}")
        else:
            print("No analysis response received")
        
        # Step 5: Close the browser
        print("\nClosing browser...")
        await browser_tool.cleanup()
        print("Browser closed successfully")
        
        print("\nExtraction and analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during extraction and analysis: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        await browser_tool.cleanup()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract and analyze content from a webpage using Ollama")
    parser.add_argument("url", help="URL of the webpage to extract content from")
    parser.add_argument("goal", help="Goal of the extraction")
    parser.add_argument("output", help="Filename to save the analysis to")
    parser.add_argument("--format", choices=["markdown", "json", "html", "text"], default="markdown", help="Format of the output file")
    args = parser.parse_args()
    
    # Create workspace directory if it doesn't exist
    os.makedirs("workspace", exist_ok=True)
    
    # Run the extraction and analysis
    asyncio.run(extract_and_analyze(args.url, args.goal, args.output, args.format))
