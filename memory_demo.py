#!/usr/bin/env python3
"""
Memory System Demo for LocalManus

This script demonstrates the capabilities of the enhanced persistent memory system
in LocalManus. It shows how to store, retrieve, and search memories using semantic
similarity.
"""
import asyncio
import os
import time
from typing import List, Dict, Any, Tuple

from app.agent.persistent_memory import PersistentMemory, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW
from app.config import config
from app.logger import logger


async def demo_memory_system():
    """Run a demonstration of the persistent memory system."""
    print("\n" + "="*80)
    print("LocalManus Persistent Memory System Demo".center(80))
    print("="*80 + "\n")
    
    # Initialize the memory system using the default path
    memory = PersistentMemory()
    
    print(f"Initialized persistent memory at: {memory.db_path}")
    
    # Demo 1: Storing memories
    print("\n[DEMO 1] Storing different types of memories")
    print("-" * 50)
    
    # Store some example memories
    memory_ids = []
    memory_ids.append(memory.store_memory(
        text="The user prefers concise responses with bullet points",
        source="user_preference",
        priority=PRIORITY_HIGH,
        tags=["preference", "style"]
    ))
    
    memory_ids.append(memory.store_memory(
        text="The user is working on a marketing project for a fitness app",
        source="conversation",
        priority=PRIORITY_MEDIUM,
        tags=["project", "context"]
    ))
    
    memory_ids.append(memory.store_memory(
        text="The fitness app targets adults aged 25-40 who want to exercise at home",
        source="conversation",
        priority=PRIORITY_MEDIUM,
        tags=["project", "target_audience"]
    ))
    
    memory_ids.append(memory.store_memory(
        text="The user mentioned they prefer blue color schemes in their marketing materials",
        source="conversation",
        priority=PRIORITY_MEDIUM,
        tags=["preference", "design"]
    ))
    
    memory_ids.append(memory.store_memory(
        text="The app is called 'FitHome' and launches next month",
        source="conversation",
        priority=PRIORITY_HIGH,
        tags=["project", "timeline"]
    ))
    
    print(f"Stored {len(memory_ids)} memories in the database")
    
    # Demo 2: Basic memory retrieval
    print("\n[DEMO 2] Basic memory retrieval")
    print("-" * 50)
    
    # Retrieve a specific memory
    memory_item = memory.get_memory(memory_ids[0])
    print(f"Retrieved memory: {memory_item.text}")
    print(f"Priority: {memory_item.priority}, Source: {memory_item.source}")
    print(f"Tags: {', '.join(memory_item.tags)}")
    
    # Retrieve all memories
    all_memories = memory.get_all_memories()
    print(f"\nRetrieved {len(all_memories)} total memories")
    
    # Filter memories by source
    project_memories = memory.get_all_memories(source="conversation")
    print(f"Retrieved {len(project_memories)} conversation memories")
    
    # Demo 3: Simple text search
    print("\n[DEMO 3] Simple text search")
    print("-" * 50)
    
    # Search for memories containing "fitness"
    fitness_results = memory.search_memories("fitness")
    print("Search results for 'fitness':")
    for memory_item, score in fitness_results:
        print(f"- {memory_item.text} (relevance: {score:.2f})")
    
    # Demo 4: Semantic search
    print("\n[DEMO 4] Semantic search")
    print("-" * 50)
    
    # Generate embeddings for all memories
    print("Generating embeddings for all memories...")
    for memory_id in memory_ids:
        memory_item = memory.get_memory(memory_id)
        embedding = await memory.get_embedding(memory_item.text)
        memory.update_memory_embedding(memory_id, embedding)
        print(f"Generated embedding for memory {memory_id}")
    
    # Perform semantic searches
    search_queries = [
        "What does the user prefer in responses?",
        "Who is the target audience?",
        "When is the product launching?",
        "What color should I use in designs?"
    ]
    
    for query in search_queries:
        print(f"\nSemantic search for: '{query}'")
        results = await memory.search_memories_semantic(query)
        for memory_item, score in results:
            if score > 0.7:  # Only show high relevance results
                print(f"- {memory_item.text} (relevance: {score:.2f})")
    
    # Demo 5: Memory prioritization
    print("\n[DEMO 5] Memory prioritization")
    print("-" * 50)
    
    # Update priority of a memory
    memory.update_memory_priority(memory_ids[1], PRIORITY_HIGH)
    print(f"Updated priority of memory {memory_ids[1]} to HIGH")
    
    # Get high priority memories
    high_priority = memory.get_all_memories(priority=PRIORITY_HIGH)
    print(f"Retrieved {len(high_priority)} high priority memories:")
    for mem in high_priority:
        print(f"- {mem.text}")
    
    # Demo 6: Task tracking
    print("\n[DEMO 6] Task tracking")
    print("-" * 50)
    
    # Store a task
    task_id = memory.store_task(
        task_description="Create a marketing plan for FitHome app",
        completed=False
    )
    print(f"Stored task with ID: {task_id}")
    
    # Update task status
    memory.update_task(
        task_id=task_id,
        completed=True,
        outcome="Created comprehensive marketing plan with social media strategy"
    )
    print("Updated task status to completed")
    
    # Get recent tasks
    recent_tasks = memory.get_recent_tasks()
    print(f"Retrieved {len(recent_tasks)} recent tasks:")
    for task in recent_tasks:
        status = "Completed" if task["completed"] else "In Progress"
        print(f"- {task['task_description']} ({status})")
        if task["outcome"]:
            print(f"  Outcome: {task['outcome']}")
    
    # Demo 7: User preferences
    print("\n[DEMO 7] User preferences")
    print("-" * 50)
    
    # Store preferences
    memory.store_preference("response_style", "concise with bullet points")
    memory.store_preference("color_theme", "blue")
    memory.store_preference("language", "English")
    print("Stored user preferences")
    
    # Retrieve preferences
    all_prefs = memory.get_all_preferences()
    print("User preferences:")
    for key, value in all_prefs.items():
        print(f"- {key}: {value}")
    
    # Clean up
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_memory_system())
