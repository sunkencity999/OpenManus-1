#!/usr/bin/env python3
"""
Quick fix script to test the URL detector integration with the original Manus agent.
This provides a more direct way to test the URL detection functionality.
"""
import asyncio
import sys

from app.agent.manus import Manus
from app.logger import logger


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        if len(sys.argv) > 1:
            prompt = sys.argv[1]
        else:
            prompt = input("Enter your prompt: ")
            
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
