import asyncio

from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def main(prompt=None, agent=None, continue_conversation=False):
    # Create and initialize ImprovedManus agent with enhanced reasoning
    logger.warning("Creating ImprovedManus agent with enhanced reasoning...")
    if agent is None:
        agent = await ImprovedManus.create()
    
    try:
        # Get user prompt
        if prompt is None:
            prompt = input("Enter your task (e.g., 'Create a poem about Harriet Tubman in the style of Stephen King'): ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        # Process the request
        logger.warning(f"Processing your request: {prompt}")
        response = await agent.run(prompt)
        
        # Extract the actual deliverable from the response
        # First try to use the get_task_result method if available
        if hasattr(agent, 'get_task_result') and callable(agent.get_task_result):
            final_result = agent.get_task_result()
            if not final_result:
                # Fall back to checking if the response contains step outputs
                if response.startswith("Step 1:"):
                    # The response contains step outputs, extract the final deliverable
                    if hasattr(agent, 'task_completer') and hasattr(agent.task_completer, 'deliverable_content'):
                        final_result = agent.task_completer.deliverable_content
                    else:
                        final_result = "No deliverable content found in the response."
                else:
                    # The response is already the deliverable
                    final_result = response
        else:
            # Fall back to the original approach
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
        
        # Ask for next task instead of closing
        next_prompt = input("\nTask completed! Enter your next task (or press Ctrl+C to exit): ")
        if next_prompt.strip():
            # Process next task recursively without cleaning up the agent
            await main(next_prompt, agent)
            return
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        if not continue_conversation:
            # Only clean up when we're really exiting
            await agent.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting LocalManus. Goodbye!")
