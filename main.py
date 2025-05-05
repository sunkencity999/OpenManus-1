import asyncio

from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def main():
    # Create and initialize ImprovedManus agent with enhanced reasoning
    logger.warning("Creating ImprovedManus agent with enhanced reasoning...")
    agent = await ImprovedManus.create()
    
    try:
        # Get user prompt
        prompt = input("Enter your task (e.g., 'Create a poem about Harriet Tubman in the style of Stephen King'): ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        # Process the request
        logger.warning(f"Processing your request: {prompt}")
        response = await agent.run(prompt)
        
        # Extract the actual deliverable from the response if it contains step outputs
        if response.startswith("Step 1:"):
            # The response contains step outputs, extract the final deliverable
            if hasattr(agent, 'task_completer') and hasattr(agent.task_completer, 'deliverable_content'):
                final_result = agent.task_completer.deliverable_content
            else:
                final_result = "No deliverable content found in the response."
        else:
            # The response is already the deliverable
            final_result = response
        
        # Display the result
        print("\n" + "="*80)
        print("TASK RESULT")
        print("="*80)
        print(final_result)
        print("="*80)
        
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
