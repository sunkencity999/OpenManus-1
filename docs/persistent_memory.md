# LocalManus Persistent Memory System

The LocalManus Persistent Memory System provides long-term storage and semantic retrieval capabilities for the agent. This document explains how the system works and how to use it effectively.

## Overview

The persistent memory system enhances the agent's capabilities by:

1. **Storing information across sessions** - The agent remembers important information even after being restarted
2. **Semantic memory retrieval** - Finding relevant information based on meaning, not just keywords
3. **Memory prioritization** - Important information is preserved longer and retrieved more readily
4. **Task tracking** - Keeping a record of completed tasks and their outcomes
5. **User preference storage** - Remembering user preferences across sessions

## Architecture

The system consists of:

- **SQLite database** - Stores memories, tasks, and preferences
- **Embedding-based retrieval** - Uses OpenAI's text-embedding-3-small model for semantic search
- **Memory prioritization system** - Assigns and manages priority levels for memories
- **Caching mechanisms** - Improves performance by caching embeddings

## Key Components

### PersistentMemory Class

This class extends the original `ConversationMemory` with persistent storage capabilities:

```python
from app.agent.persistent_memory import PersistentMemory

# Initialize with a database path
memory = PersistentMemory(db_path="./memory.db")

# Store a memory
memory_id = memory.store_memory(
    text="The user prefers concise responses",
    source="conversation",
    priority="high",
    tags=["preference", "style"]
)

# Retrieve memories semantically
results = await memory.search_memories_semantic("What does the user like?")
for memory_item, score in results:
    print(f"{memory_item.text} (relevance: {score:.2f})")
```

### Memory Items

Each memory item contains:

- **Text content** - The actual information
- **Source** - Where the information came from (conversation, browser, etc.)
- **Timestamp** - When the memory was created
- **Embedding** - Vector representation for semantic search
- **Priority** - Importance level (high, medium, low)
- **Tags** - Categories for filtering
- **Metadata** - Additional structured information

### Priority Levels

The system uses three priority levels:

- **High** - Critical information that should be preserved long-term
- **Medium** - Important information that is relevant to current tasks
- **Low** - General information that may be cleaned up over time

## Integration with ImprovedManus

The `ImprovedManus` agent has been updated to use the persistent memory system:

1. **Initialization** - The agent creates a PersistentMemory instance on startup
2. **Context Storage** - Information is stored in both in-memory context and persistent storage
3. **Web Content** - Content from web browsing is stored with high priority
4. **Task Completion** - The agent uses semantic search to find relevant memories for task completion
5. **Task Tracking** - Completed tasks are recorded for future reference

## Usage Examples

### Storing Information

```python
# Store user preferences
memory.store_memory(
    text="The user prefers blue color schemes",
    source="conversation",
    priority="high",
    tags=["preference", "design"]
)

# Store web content
memory.store_memory(
    text=f"Content from {url}: {content}",
    source="browser",
    priority="medium",
    tags=["web_content"],
    metadata={"url": url}
)
```

### Retrieving Information

```python
# Simple text search
results = memory.search_memories("blue color")

# Semantic search
results = await memory.search_memories_semantic("What color does the user like?")

# Get all high priority memories
high_priority = memory.get_all_memories(priority="high")
```

### Managing Tasks

```python
# Store a task
task_id = memory.store_task(
    task_description="Create a marketing plan",
    completed=False
)

# Update task status
memory.update_task(
    task_id=task_id,
    completed=True,
    outcome="Created comprehensive marketing plan"
)

# Get recent tasks
recent_tasks = memory.get_recent_tasks()
```

### User Preferences

```python
# Store preferences
memory.store_preference("response_style", "concise")
memory.store_preference("language", "English")

# Retrieve preferences
style = memory.get_preference("response_style")
all_prefs = memory.get_all_preferences()
```

## Performance Considerations

- **Database Size** - The database can grow over time, especially with many stored memories
- **Embedding Generation** - Generating embeddings requires API calls to OpenAI
- **Caching** - Embeddings are cached to improve performance
- **Memory Cleanup** - Use `cleanup_old_memories()` periodically to remove old, low-priority memories

## Future Improvements

- **Local embedding models** - Replace OpenAI API with local models
- **Vector database** - Upgrade to a dedicated vector database for better scaling
- **Memory consolidation** - Combine related memories to reduce redundancy
- **Forgetting mechanisms** - More sophisticated approaches to memory decay
