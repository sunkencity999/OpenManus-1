#!/usr/bin/env python3
"""
Test script to compare the original Manus agent with the improved version.
This allows testing both agents with the same prompt to see the difference in behavior.
"""
import asyncio
import argparse

from app.agent.manus import Manus
from app.agent.improved_manus import ImprovedManus
from app.logger import logger


async def run_agent(agent_type, prompt):
    """Run the specified agent with the given prompt."""
    if agent_type == "original":
        agent = await Manus.create()
    elif agent_type == "improved":
        agent = await ImprovedManus.create()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    try:
        logger.warning(f"Running {agent_type} agent with prompt: {prompt}")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


async def main():
    parser = argparse.ArgumentParser(description="Test different OpenManus agents")
    parser.add_argument(
        "--agent", 
        choices=["original", "improved", "both"], 
        default="improved",
        help="Which agent to test (original, improved, or both)"
    )
    parser.add_argument(
        "prompt", 
        nargs="?", 
        default="",
        help="The prompt to test with"
    )
    
    args = parser.parse_args()
    
    if not args.prompt:
        args.prompt = input("Enter your prompt: ")
        
    if not args.prompt.strip():
        logger.warning("Empty prompt provided.")
        return
        
    if args.agent == "both":
        logger.warning("Testing both agents sequentially...")
        logger.warning("-" * 50)
        logger.warning("ORIGINAL AGENT")
        logger.warning("-" * 50)
        await run_agent("original", args.prompt)
        
        logger.warning("\n\n")
        logger.warning("-" * 50)
        logger.warning("IMPROVED AGENT")
        logger.warning("-" * 50)
        await run_agent("improved", args.prompt)
    else:
        await run_agent(args.agent, args.prompt)


if __name__ == "__main__":
    asyncio.run(main())
