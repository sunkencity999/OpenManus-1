"""
Persistent memory manager for LocalManus that provides long-term storage
and semantic retrieval capabilities.
"""
import os
import json
import time
import sqlite3
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
from pydantic import BaseModel, Field, model_validator

from app.agent.memory_manager import ConversationMemory
from app.config import config

# Set up logging
logger = logging.getLogger(__name__)

# Priority levels for memory items
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"


class MemoryItem(BaseModel):
    """A single memory item with metadata."""
    
    id: Optional[int] = None
    text: str
    source: str = "conversation"  # conversation, browser, file, etc.
    timestamp: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    priority: str = PRIORITY_MEDIUM
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def age_days(self) -> float:
        """Get the age of this memory in days."""
        return (time.time() - self.timestamp) / (60 * 60 * 24)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp,
            "embedding": json.dumps(self.embedding) if self.embedding else None,
            "priority": self.priority,
            "tags": json.dumps(self.tags),
            "metadata": json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create a MemoryItem from a dictionary."""
        item_data = data.copy()
        if isinstance(item_data.get("embedding"), str) and item_data["embedding"]:
            item_data["embedding"] = json.loads(item_data["embedding"])
        if isinstance(item_data.get("tags"), str):
            item_data["tags"] = json.loads(item_data["tags"])
        if isinstance(item_data.get("metadata"), str):
            item_data["metadata"] = json.loads(item_data["metadata"])
        return cls(**item_data)


class PersistentMemory(ConversationMemory):
    """
    Enhanced memory manager that provides persistent storage and semantic retrieval.
    Extends the ConversationMemory class with database storage and embedding-based retrieval.
    """
    
    # Database configuration
    db_path: str = Field(default=None)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # Dimension of OpenAI's text-embedding-3-small
    
    # Default memory directory paths by platform
    _default_memory_dirs = {
        "darwin": os.path.expanduser("~/.localmanus/memory"),  # macOS
        "linux": os.path.expanduser("~/.localmanus/memory"),   # Linux
        "win32": os.path.expanduser("~/.localmanus/memory")    # Windows
    }
    
    # Memory management settings
    max_memories: int = 1000
    memory_decay_rate: float = 0.1  # Rate at which memory importance decays per day
    
    # Caching settings
    _embedding_cache: Dict[str, List[float]] = Field(default_factory=dict)
    _cache_size: int = 100
    
    # Connection management
    _conn: Optional[sqlite3.Connection] = None
    
    @model_validator(mode="after")
    def initialize_database(self) -> "PersistentMemory":
        """Initialize the database connection and tables."""
        # Set default db_path if none is provided
        if self.db_path is None:
            import platform
            system = platform.system().lower()
            if system == "darwin":
                platform_key = "darwin"  # macOS
            elif system == "linux":
                platform_key = "linux"   # Linux
            elif system == "windows":
                platform_key = "win32"   # Windows
            else:
                platform_key = "linux"   # Default to Linux for unknown systems
                
            # Use the default directory for the current platform
            default_dir = self._default_memory_dirs.get(platform_key, os.path.expanduser("~/.localmanus/memory"))
            
            # Create a unique database file name based on timestamp
            import uuid
            db_name = f"memory_{uuid.uuid4().hex[:8]}.db"
            self.db_path = os.path.join(default_dir, db_name)
            logger.info(f"Using default memory database path: {self.db_path}")
            
        self._ensure_db_directory()
        self._connect_db()
        self._create_tables()
        return self
    
    def _ensure_db_directory(self) -> None:
        """Ensure the directory for the database exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _connect_db(self) -> None:
        """Connect to the SQLite database."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
    
    def _create_tables(self) -> None:
        """Create the necessary database tables if they don't exist."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        # Create memories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp REAL NOT NULL,
            embedding TEXT,
            priority TEXT NOT NULL,
            tags TEXT NOT NULL,
            metadata TEXT NOT NULL
        )
        ''')
        
        # Create tasks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_description TEXT NOT NULL,
            completed BOOLEAN NOT NULL,
            timestamp REAL NOT NULL,
            outcome TEXT,
            metadata TEXT NOT NULL
        )
        ''')
        
        # Create preferences table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        ''')
        
        # Create index for faster similarity search
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority)')
        
        self._conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self) -> None:
        """Ensure the database connection is closed when the object is deleted."""
        self.close()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text string using OpenAI's API."""
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            from openai import AsyncOpenAI
            
            # Initialize OpenAI client
            client = AsyncOpenAI(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url
            )
            
            # Get embedding from OpenAI
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Update cache
            if len(self._embedding_cache) >= self._cache_size:
                # Remove a random item if cache is full
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            
            self._embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dimension
    
    def store_memory(self, text: str, source: str = "conversation", 
                    priority: str = PRIORITY_MEDIUM, tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> int:
        """
        Store a new memory in the database.
        
        Args:
            text: The text content of the memory
            source: Where this memory came from
            priority: Importance level (high, medium, low)
            tags: List of tags for categorization
            metadata: Additional structured data
            
        Returns:
            The ID of the stored memory
        """
        self._connect_db()
        
        # Default values
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
            
        # Create memory item
        memory = MemoryItem(
            text=text,
            source=source,
            timestamp=time.time(),
            priority=priority,
            tags=tags,
            metadata=metadata
        )
        
        # Insert into database
        cursor = self._conn.cursor()
        cursor.execute('''
        INSERT INTO memories (text, source, timestamp, embedding, priority, tags, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.text,
            memory.source,
            memory.timestamp,
            None,  # Embedding will be added asynchronously
            memory.priority,
            json.dumps(memory.tags),
            json.dumps(memory.metadata)
        ))
        
        memory_id = cursor.lastrowid
        self._conn.commit()
        
        # Schedule embedding computation (this would be done asynchronously in production)
        # For now, we'll just log that it needs to be done
        logger.info(f"Memory {memory_id} stored, embedding will be computed asynchronously")
        
        return memory_id
    
    def update_memory_embedding(self, memory_id: int, embedding: List[float]) -> None:
        """Update the embedding for a stored memory."""
        self._connect_db()
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (json.dumps(embedding), memory_id)
        )
        self._conn.commit()
    
    def get_memory(self, memory_id: int) -> Optional[MemoryItem]:
        """Retrieve a specific memory by ID."""
        self._connect_db()
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if row:
            return MemoryItem.from_dict(dict(row))
        return None
    
    def get_all_memories(self, limit: int = 100, 
                        source: Optional[str] = None,
                        priority: Optional[str] = None) -> List[MemoryItem]:
        """
        Retrieve all memories with optional filtering.
        
        Args:
            limit: Maximum number of memories to retrieve
            source: Filter by source
            priority: Filter by priority level
            
        Returns:
            List of memory items
        """
        self._connect_db()
        cursor = self._conn.cursor()
        
        query = "SELECT * FROM memories"
        params = []
        
        # Add filters
        conditions = []
        if source:
            conditions.append("source = ?")
            params.append(source)
        if priority:
            conditions.append("priority = ?")
            params.append(priority)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [MemoryItem.from_dict(dict(row)) for row in rows]
    
    def search_memories(self, query: str, limit: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        Search memories using text similarity.
        This is a simple implementation that doesn't use embeddings yet.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of (memory, relevance_score) tuples
        """
        self._connect_db()
        cursor = self._conn.cursor()
        
        # Simple text search
        cursor.execute(
            "SELECT * FROM memories WHERE text LIKE ? ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", limit)
        )
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            memory = MemoryItem.from_dict(dict(row))
            # Calculate simple relevance score based on substring match
            relevance = 1.0 if query.lower() in memory.text.lower() else 0.5
            results.append((memory, relevance))
            
        return results
    
    async def search_memories_semantic(self, query: str, limit: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        Search memories using semantic similarity with embeddings.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of (memory, relevance_score) tuples
        """
        self._connect_db()
        
        # Get embedding for the query
        query_embedding = await self.get_embedding(query)
        
        # Get all memories with embeddings
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            memory = MemoryItem.from_dict(dict(row))
            
            if memory.embedding:
                # Calculate cosine similarity
                similarity = 1 - cosine(query_embedding, memory.embedding)
                results.append((memory, similarity))
        
        # Sort by similarity (highest first) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def update_memory_priority(self, memory_id: int, priority: str) -> None:
        """Update the priority of a memory."""
        self._connect_db()
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE memories SET priority = ? WHERE id = ?",
            (priority, memory_id)
        )
        self._conn.commit()
    
    def delete_memory(self, memory_id: int) -> None:
        """Delete a memory from the database."""
        self._connect_db()
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
    
    def cleanup_old_memories(self, days_threshold: int = 30, 
                           keep_high_priority: bool = True) -> int:
        """
        Remove old, low-priority memories to prevent database bloat.
        
        Args:
            days_threshold: Age threshold in days
            keep_high_priority: Whether to keep high priority memories regardless of age
            
        Returns:
            Number of memories deleted
        """
        self._connect_db()
        cursor = self._conn.cursor()
        
        timestamp_threshold = time.time() - (days_threshold * 24 * 60 * 60)
        
        if keep_high_priority:
            cursor.execute(
                "DELETE FROM memories WHERE timestamp < ? AND priority != ?",
                (timestamp_threshold, PRIORITY_HIGH)
            )
        else:
            cursor.execute(
                "DELETE FROM memories WHERE timestamp < ?",
                (timestamp_threshold,)
            )
            
        deleted_count = cursor.rowcount
        self._conn.commit()
        
        return deleted_count
    
    def store_task(self, task_description: str, completed: bool = False,
                 outcome: Optional[str] = None, metadata: Dict[str, Any] = None) -> int:
        """
        Store information about a task.
        
        Args:
            task_description: Description of the task
            completed: Whether the task was completed
            outcome: Result of the task
            metadata: Additional data about the task
            
        Returns:
            The ID of the stored task
        """
        self._connect_db()
        
        if metadata is None:
            metadata = {}
            
        cursor = self._conn.cursor()
        cursor.execute('''
        INSERT INTO tasks (task_description, completed, timestamp, outcome, metadata)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            task_description,
            completed,
            time.time(),
            outcome,
            json.dumps(metadata)
        ))
        
        task_id = cursor.lastrowid
        self._conn.commit()
        
        return task_id
    
    def update_task(self, task_id: int, completed: bool, outcome: Optional[str] = None) -> None:
        """Update the status of a task."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        if outcome is not None:
            cursor.execute(
                "UPDATE tasks SET completed = ?, outcome = ? WHERE id = ?",
                (completed, outcome, task_id)
            )
        else:
            cursor.execute(
                "UPDATE tasks SET completed = ? WHERE id = ?",
                (completed, task_id)
            )
            
        self._conn.commit()
    
    def get_recent_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent tasks."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        cursor.execute(
            "SELECT * FROM tasks ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        
        rows = cursor.fetchall()
        
        tasks = []
        for row in rows:
            task = dict(row)
            task['metadata'] = json.loads(task['metadata'])
            tasks.append(task)
            
        return tasks
    
    def store_preference(self, key: str, value: str) -> None:
        """Store a user preference."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        # Check if preference already exists
        cursor.execute("SELECT id FROM preferences WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row:
            # Update existing preference
            cursor.execute(
                "UPDATE preferences SET value = ?, timestamp = ? WHERE key = ?",
                (value, time.time(), key)
            )
        else:
            # Insert new preference
            cursor.execute(
                "INSERT INTO preferences (key, value, timestamp) VALUES (?, ?, ?)",
                (key, value, time.time())
            )
            
        self._conn.commit()
    
    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a stored user preference."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        cursor.execute("SELECT value FROM preferences WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row:
            return row['value']
        return default
    
    def get_all_preferences(self) -> Dict[str, str]:
        """Get all stored user preferences."""
        self._connect_db()
        cursor = self._conn.cursor()
        
        cursor.execute("SELECT key, value FROM preferences")
        rows = cursor.fetchall()
        
        return {row['key']: row['value'] for row in rows}
    
    # Override ConversationMemory methods to use persistent storage
    
    def add_question(self, question: str) -> None:
        """Record a question that was asked."""
        super().add_question(question)
        self.store_memory(
            text=f"Question asked: {question}",
            source="conversation",
            priority=PRIORITY_MEDIUM,
            tags=["question"]
        )
    
    def add_response(self, question: str, response: str) -> None:
        """Record a response to a question."""
        super().add_response(question, response)
        self.store_memory(
            text=f"Q: {question}\nA: {response}",
            source="conversation",
            priority=PRIORITY_MEDIUM,
            tags=["response"],
            metadata={"question": question}
        )
    
    def add_context(self, key: str, value: str, confidence: float = 1.0, source: str = "unknown") -> None:
        """Add context information."""
        super().add_context(key, value)
        
        # Determine priority based on confidence
        if confidence > 0.8:
            priority = PRIORITY_HIGH
        elif confidence > 0.5:
            priority = PRIORITY_MEDIUM
        else:
            priority = PRIORITY_LOW
            
        self.store_memory(
            text=f"{key}: {value}",
            source=source,
            priority=priority,
            tags=["context", key],
            metadata={"confidence": confidence}
        )
    
    async def get_relevant_context(self, query: str) -> List[str]:
        """Get relevant context from previous conversation based on a query."""
        # Use semantic search if available
        try:
            results = await self.search_memories_semantic(query, limit=5)
            return [f"{mem.text}" for mem, score in results if score > 0.7]
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            # Fall back to basic search
            results = self.search_memories(query, limit=5)
            return [f"{mem.text}" for mem, score in results]
    
    def reset(self) -> None:
        """Reset the conversation memory."""
        super().reset()
        # We don't delete persistent memories, just mark the reset
        self.store_memory(
            text="Conversation memory was reset",
            source="system",
            priority=PRIORITY_LOW,
            tags=["system", "reset"]
        )
