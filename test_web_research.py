from app.agent.improved_manus import ImprovedManus
import asyncio
import logging
import time
import re

# Configure more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the logger for our test script
logger = logging.getLogger('web_research_test')

async def test():
    print("\n===== TESTING WEB RESEARCH FUNCTIONALITY =====\n")
    
    # Initialize the agent properly
    logger.info("Initializing ImprovedManus agent...")
    manus = await ImprovedManus.create()
    logger.info("Agent initialized successfully.")
    
    # Test query - using a query about quantum computing to test our changes
    query = 'What are the latest developments in quantum computing?'
    logger.info(f"Performing web research for query: '{query}'")
    logger.info(f"Using max_sites=10 to process more URLs")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Perform web research with 10 URLs
        logger.info("Starting web research...")
        result = await manus.perform_web_research(query, max_sites=10)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Count the number of sources processed
        source_count = len(re.findall(r'--- SOURCE', result)) if result else 0
        
        logger.info(f"Web research completed in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {source_count} sources")
        logger.info(f"Result length: {len(result)} characters")
        
        print("\n===== RESEARCH RESULT =====\n")
        print(result[:1000] + "..." if len(result) > 1000 else result)  # Show first 1000 chars to avoid flooding console
        print(f"\n===== PROCESSED {source_count} SOURCES =====\n")
    except Exception as e:
        logger.error(f"Error during web research: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up the agent
        logger.info("Cleaning up agent resources...")
        await manus.cleanup()
        logger.info("Test completed.")
        print("\n===== END OF TEST =====\n")

if __name__ == "__main__":
    asyncio.run(test())
