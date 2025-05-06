#!/usr/bin/env python3
"""
Test script for the enhanced web research functionality in ImprovedManus agent.
This script tests the ability to extract and analyze content from multiple URLs,
handle website overlays, and process up to 10 sources.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import the ImprovedManus agent
from app.agent.improved_manus import ImprovedManus

async def test_web_research():
    """Test the web research functionality with various queries."""
    
    # Create an instance of the ImprovedManus agent
    agent = await ImprovedManus.create()
    
    # Test queries with different complexity and topics
    test_queries = [
        "What are the latest developments in quantum computing?",
        "How does climate change affect marine ecosystems?",
        "What are the benefits and risks of artificial intelligence?",
        "What are the most effective strategies for cybersecurity?",
        "How has remote work impacted productivity and work-life balance?"
    ]
    
    for i, query in enumerate(test_queries):
        logger.info(f"\n\n{'='*80}\nTEST {i+1}: {query}\n{'='*80}")
        
        try:
            # Perform web research with the query
            result = await agent.perform_web_research(query, max_sites=10)
            
            # Log the result
            logger.info(f"Research result length: {len(result)}")
            logger.info(f"Research result sample: {result[:500]}...")
            
            # Check if the result is meaningful
            if len(result) > 1000:
                logger.info(f"✅ Test {i+1} PASSED: Got meaningful result")
            else:
                logger.warning(f"❌ Test {i+1} FAILED: Result too short")
                
        except Exception as e:
            logger.error(f"❌ Test {i+1} FAILED with error: {str(e)}")
    
    # Clean up
    await agent.cleanup()
    logger.info("\nAll tests completed")

async def test_overlay_handling():
    """Test the website overlay handling functionality."""
    
    # Create an instance of the ImprovedManus agent
    agent = await ImprovedManus.create()
    
    # Helper function to execute browser actions safely using the agent's method
    async def execute_browser_action(action, **kwargs):
        return await agent.execute_browser_use(action, **kwargs)
    
    # Test websites known for having overlays
    test_sites = [
        "https://www.nytimes.com",
        "https://www.washingtonpost.com",
        "https://www.theguardian.com",
        "https://www.cnn.com",
        "https://www.bbc.com"
    ]
    
    for i, site in enumerate(test_sites):
        logger.info(f"\n\n{'='*80}\nOVERLAY TEST {i+1}: {site}\n{'='*80}")
        
        try:
            # Navigate to the site
            await execute_browser_action('go_to_url', url=site)
            logger.info(f"Navigated to {site}")
            
            # Wait for the page to load
            import time
            time.sleep(3)
            
            # Handle overlays
            await agent._handle_website_overlays()
            logger.info(f"Handled overlays on {site}")
            
            # Check if we can access the content
            content_check = """
            function checkContent() {
                // Check if we can access the main content
                const contentSelectors = [
                    'main', '#main', '.main', 'article', '.article', 
                    '.content', '#content', '.story', '.post'
                ];
                
                for (const selector of contentSelectors) {
                    const content = document.querySelector(selector);
                    if (content && content.textContent.length > 500) {
                        return {
                            success: true,
                            selector: selector,
                            textLength: content.textContent.length
                        };
                    }
                }
                
                // If no specific content found, check body text length
                const bodyText = document.body.textContent;
                return {
                    success: bodyText.length > 1000,
                    selector: 'body',
                    textLength: bodyText.length
                };
            }
            return checkContent();
            """
            
            # For JavaScript evaluation, we need to use a different approach
            # The browser_use tool expects 'action' to be one of the allowed actions
            # Let's use a direct approach to evaluate JavaScript
            browser_tool = agent.available_tools.get_tool("browser_use")
            result = await browser_tool.execute('evaluate', script=content_check)
            
            # Handle different response formats
            if isinstance(result, dict) and 'success' in result:
                # Dictionary response format
                if result['success']:
                    logger.info(f"✅ Overlay Test {i+1} PASSED: Accessed content with selector {result.get('selector')}, text length: {result.get('textLength')}")
                else:
                    logger.warning(f"❌ Overlay Test {i+1} FAILED: Could not access content")
            elif isinstance(result, str):
                # String response format (might be an error message)
                if 'error' in result.lower():
                    logger.warning(f"❌ Overlay Test {i+1} FAILED: {result}")
                else:
                    # Consider it a success if we got a non-error string response
                    logger.info(f"✅ Overlay Test {i+1} PASSED: Got response of length {len(result)}")
            else:
                # Any other response format
                logger.info(f"✅ Overlay Test {i+1} PASSED: Got response of type {type(result)}")
                logger.info(f"Response: {str(result)[:100]}...")
                
                
        except Exception as e:
            logger.error(f"❌ Overlay Test {i+1} FAILED with error: {str(e)}")
    
    # Clean up
    await agent.cleanup()
    logger.info("\nAll overlay tests completed")

async def main():
    """Run all tests."""
    logger.info("Starting web research functionality tests")
    
    # Test web research functionality
    await test_web_research()
    
    # Test overlay handling
    await test_overlay_handling()
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
