import asyncio
import json
import logging
import os
import random
import re
import time
import traceback  # For detailed error logging in python_execute
from datetime import datetime  # Added for timestamps
from pathlib import Path  # Added for path manipulation
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urljoin, urlparse  # Added for URL parsing

import requests  # Added for Ollama
from bs4 import BeautifulSoup  # Added for parsing
from pydantic import Field, model_validator

# Assuming these imports point to the correct modules in your project structure
from app.agent.browser import BrowserContextHelper
from app.agent.browser_navigator import BrowserNavigator
from app.agent.memory_manager import (
    ToolUsageTracker,
)  # ConversationMemory removed if PersistentMemory replaces its direct use
from app.agent.persistent_memory import PersistentMemory  # Using PersistentMemory
from app.agent.task_analyzer import TaskAnalyzer
from app.agent.task_completer import TaskCompleter
from app.agent.toolcall import ToolCallAgent
from app.agent.url_detector import URLDetector
from app.config import config  # Import config
from app.logger import logger
from app.prompt.improved_manus import IMPROVED_NEXT_STEP_PROMPT, IMPROVED_SYSTEM_PROMPT
from app.schema import Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.python_execute import PythonExecute
from app.tool.smart_ask_human import SmartAskHuman
from app.tool.str_replace_editor import StrReplaceEditor


class ImprovedManus(ToolCallAgent):
    """
    An improved version of the Manus agent with better reasoning capabilities.
    Focuses on minimizing unnecessary questions and making better decisions about
    when user input is truly required.
    """

    name: str = "ImprovedManus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools including MCP-based tools, with improved reasoning"
    )

    system_prompt: str = IMPROVED_SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = IMPROVED_NEXT_STEP_PROMPT

    # Properties to store task results
    _task_result: str = ""
    _deliverable_content: str = ""
    _deliverable_path: str = ""

    max_observe: int = 15000  # Increased observation length for web content
    max_steps: int = 10  # Default to a reasonable number of steps
    dynamic_steps: bool = (
        True  # Enable dynamic step adjustment based on task complexity
    )

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection, using SmartAskHuman instead of AskHuman
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            SmartAskHuman(),  # SmartAskHuman should ideally use PersistentMemory
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # Track connected MCP servers
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command
    _initialized: bool = False

    # New components for improved reasoning
    conversation_memory: PersistentMemory = None  # Using PersistentMemory
    browser_navigator: BrowserNavigator = None
    url_detector: URLDetector = None
    task_completer: TaskCompleter = None
    task_analyzer: Optional[TaskAnalyzer] = None

    # Memory database path - if None, will use the default platform-specific path
    memory_db_path: Optional[str] = Field(default=None)

    # Tracking attributes
    browser_used: bool = False
    visited_urls: Set[str] = Field(default_factory=set)
    current_task: str = ""
    task_completed: bool = False
    step_count: int = 0  # Initialize step counter

    # Memory management components
    tool_tracker: ToolUsageTracker = Field(default_factory=ToolUsageTracker)

    # Track URLs mentioned in the conversation
    mentioned_urls: Set[str] = Field(default_factory=set)

    # Flag to prevent repeated tool calls
    _prevent_repeated_questions: bool = True

    @model_validator(mode="after")
    def initialize_helper(self) -> "ImprovedManus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)

        # Use PersistentMemory
        if self.conversation_memory is None:  # Initialize only if not already set
            if self.memory_db_path is not None:
                # If a specific path is provided, use it, relative to workspace
                memory_path = Path(config.workspace_root) / self.memory_db_path
                memory_path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure directory exists
                self.conversation_memory = PersistentMemory(db_path=str(memory_path))
                logger.info(
                    f"Initialized PersistentMemory with custom path: {memory_path}"
                )
            else:
                # Otherwise use the default platform-specific path
                self.conversation_memory = (
                    PersistentMemory()
                )  # Default path handled by PersistentMemory class
                logger.info(
                    f"Initialized PersistentMemory with default path: {self.conversation_memory.db_path}"
                )

        self.browser_navigator = BrowserNavigator()
        self.url_detector = URLDetector()
        self.task_completer = TaskCompleter()
        self.task_analyzer = TaskAnalyzer(llm=self.llm)

        # Pass PersistentMemory instance to SmartAskHuman as context_manager
        smart_ask_tool = self.available_tools.get_tool("ask_human")
        if isinstance(smart_ask_tool, SmartAskHuman):
            smart_ask_tool.context_manager = (
                self.conversation_memory
            )  # SmartAskHuman uses context_manager, not memory

        return self

    @classmethod
    async def create(cls, **kwargs) -> "ImprovedManus":
        """Factory method to create and properly initialize an ImprovedManus instance."""
        instance = cls(**kwargs)
        # Initialization including MCP servers is now handled after Pydantic validation
        # We ensure _initialized is True after successful post-validation setup
        # await instance.initialize_mcp_servers() # Called from think if not initialized
        instance._initialized = False  # Will be set true in think or after MCP init
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        if self._initialized:  # Avoid re-initialization
            return

        cfg = config  # Use config for configuration settings
        if (
            not hasattr(cfg, "mcp_config")
            or not cfg.mcp_config
            or not hasattr(cfg.mcp_config, "servers")
        ):
            logger.warning(
                "MCP configuration not found or empty. Skipping MCP server initialization."
            )
            self._initialized = True  # Mark as initialized even if no MCP servers
            return

        logger.info("Initializing MCP server connections...")
        connect_tasks = []
        for server_id, server_config in cfg.mcp_config.servers.items():
            try:
                if server_config.type == "sse" and server_config.url:
                    connect_tasks.append(
                        self.connect_mcp_server(server_config.url, server_id)
                    )
                elif server_config.type == "stdio" and server_config.command:
                    connect_tasks.append(
                        self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                    )
            except Exception as e:
                logger.error(
                    f"Failed to prepare connection for MCP server {server_id}: {e}"
                )

        if connect_tasks:
            await asyncio.gather(*connect_tasks)  # Connect concurrently
            logger.info(f"Attempted connection to {len(connect_tasks)} MCP servers.")
        else:
            logger.info("No MCP servers configured to connect.")

        self._initialized = True

    async def connect_mcp_server(
        self,
        server_endpoint: str,  # Renamed from server_url to endpoint (can be command or url)
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        server_id = server_id or server_endpoint  # Ensure server_id has a value
        if server_id in self.connected_servers:
            logger.warning(f"Already connected to MCP server {server_id}. Skipping.")
            return

        logger.info(
            f"Connecting to MCP server {server_id} ({'stdio' if use_stdio else 'sse'})..."
        )
        try:
            if use_stdio:
                await self.mcp_clients.connect_stdio(
                    server_endpoint, stdio_args or [], server_id
                )
                self.connected_servers[server_id] = f"stdio:{server_endpoint}"
                logger.info(
                    f"Connected to MCP server {server_id} using command {server_endpoint}"
                )
            else:
                await self.mcp_clients.connect_sse(server_endpoint, server_id)
                self.connected_servers[server_id] = f"sse:{server_endpoint}"
                logger.info(f"Connected to MCP server {server_id} at {server_endpoint}")

            # Update available tools with only the new tools from this server
            new_tools = [
                tool
                for tool in self.mcp_clients.tools
                if getattr(tool, "server_id", None) == server_id
            ]
            if new_tools:
                self.available_tools.add_tools(*new_tools)
                logger.info(
                    f"Added {len(new_tools)} tools from MCP server {server_id}: {[t.name for t in new_tools]}"
                )
            else:
                logger.warning(
                    f"No new tools found for MCP server {server_id} after connection."
                )

        except Exception as e:
            logger.error(
                f"Failed to connect to MCP server {server_id} ({server_endpoint}): {e}",
                exc_info=True,
            )
            # Clean up if connection failed partially
            self.connected_servers.pop(server_id, None)

    async def disconnect_mcp_server(self, server_id: str) -> None:
        """Disconnect from an MCP server and remove its tools."""
        if not server_id or server_id not in self.connected_servers:
            logger.warning(
                f"Disconnect MCP: Server ID '{server_id}' not found or not connected."
            )
            return

        logger.info(f"Disconnecting from MCP server {server_id}...")
        try:
            await self.mcp_clients.disconnect(server_id)
            self.connected_servers.pop(server_id, None)

            # Rebuild available tools list more carefully
            current_tools = self.available_tools.tools
            updated_tools = []
            removed_count = 0
            for tool in current_tools:
                if (
                    isinstance(tool, MCPClientTool)
                    and getattr(tool, "server_id", None) == server_id
                ):
                    removed_count += 1
                    continue  # Skip tools from the disconnected server
                updated_tools.append(tool)

            self.available_tools = ToolCollection(
                *updated_tools
            )  # Recreate with filtered list
            logger.info(
                f"Disconnected from MCP server {server_id}. Removed {removed_count} tools."
            )

        except Exception as e:
            logger.error(
                f"Error disconnecting from MCP server {server_id}: {e}", exc_info=True
            )

    async def cleanup(self):
        """Clean up ImprovedManus agent resources."""
        logger.info("Cleaning up ImprovedManus agent...")

        # Clean up browser context if it exists
        if self.browser_context_helper:
            logger.debug("Cleaning up browser context...")
            await self.browser_context_helper.cleanup_browser()  # Use the correct cleanup method
            logger.info("Browser context cleaned up.")

        # Disconnect from all MCP servers
        if self.connected_servers:

            server_ids_to_disconnect = list(self.connected_servers.keys())
            logger.info(
                f"Disconnecting from {len(server_ids_to_disconnect)} MCP servers..."
            )
            disconnect_tasks = [
                self.disconnect_mcp_server(sid) for sid in server_ids_to_disconnect
            ]
            await asyncio.gather(*disconnect_tasks)
            logger.info("MCP servers disconnected.")

        # Close persistent memory connection
        if self.conversation_memory:
            try:
                # Check if close method exists and is awaitable
                if hasattr(self.conversation_memory, "close"):
                    close_method = getattr(self.conversation_memory, "close")
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        # Call non-async close method
                        close_method()
                    logger.info("Persistent memory connection closed.")
            except Exception as e:
                logger.error(f"Error closing persistent memory: {e}")

        self._initialized = False
        logger.info("ImprovedManus cleanup complete.")

    def _parse_task_keyword(self, task: str) -> Tuple[Optional[str], str]:
        """Extracts a routing keyword (e.g., 'research:', 'create:') from the start of the prompt, returns (keyword, prompt)."""
        task = task.lstrip()
        for kw in ["research:", "create:", "summarize:", "analyze:", "plan:"]:
            if task.lower().startswith(kw):
                return kw[:-1], task[len(kw) :].lstrip()
        return None, task

    def _list_keywords(self):
        """Return a string listing all available routing keywords and their descriptions."""
        return (
            "Available routing keywords:\n"
            "  create:    Generate creative content (poems, essays, stories, etc.)\n"
            "  research:  Perform research, use browser/tools, or comparative analysis\n"
            "  summarize: Summarize content or documents\n"
            "  analyze:   Analyze information or data\n"
            "  plan:      Create plans or outlines\n"
            "\nExample usage: create: Write a poem about the sea.\n"
            "You can also use natural language without a keyword—agent will infer intent."
        )

    async def analyze_new_task(self, task: str) -> None:
        """Analyze the new task and set up required information."""
        if task.strip() == "/keywords":
            self._explicit_task_type = "keywords"
            self._explicit_task_prompt = self._list_keywords()
            logger.info("[KEYWORD LIST] User requested list of routing keywords.")
            return
        keyword, prompt = self._parse_task_keyword(task)
        if keyword:
            logger.info(
                f"[KEYWORD ROUTING] Detected keyword: '{keyword}'. Routing explicitly."
            )
            self.conversation_memory.reset_for_new_task()

            # Special handling for 'create:' keyword - directly generate creative content
            if keyword == "create":
                logger.info(
                    f"[CREATE KEYWORD] Directly generating creative content for: {prompt}"
                )
                try:
                    # Check for filename in the prompt BEFORE generating content
                    filename_match = re.search(
                        r'save (?:to|as)\s*["''"]([^"''"]+)\s*["''"]',
                        prompt,
                        re.IGNORECASE,
                    )
                    if not filename_match:
                        # Try without quotes
                        filename_match = re.search(
                            r"save (?:to|as)\s+([\w\-./\\]+)", prompt, re.IGNORECASE
                        )
                    
                    # Extract target filename if found
                    target_filename = filename_match.group(1).strip() if filename_match else None
                    if target_filename:
                        logger.info(f"[CREATE FILENAME] Will save to: {target_filename}")
                    
                    # Generate the content with target filename for direct saving
                    content = await self._generate_creative_content(prompt, target_filename=target_filename)
                    # Set the current task name for proper task completion
                    self.current_task = (
                        f"Create: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
                    )

                    # Store the content in memory for display in task result
                    result_summary = f"\n---\n\n### Creative Content Generated\n\n```\n{content[:500]}\n```\n\n{'...[content truncated]...' if len(content) > 500 else ''}\n\n"

                    if filename_match:
                        filename = filename_match.group(1).strip("\"'")
                        # Full path already created in the _generate_creative_content method
                        full_path = os.path.join(config.workspace_root, filename)

                        # Set these properties for proper task result display
                        self.task_completed = True
                        self._task_result = f"Creative content generated and saved to {filename}\n{result_summary}"
                        self._deliverable_content = content
                        self._deliverable_path = full_path

                        # Update memory with task completion message
                        completion_message = f"✅ I've created your content and saved it to '{filename}'.\n\n{content[:200]}... [content truncated]\n\nTask completed successfully! What would you like me to help you with next?"
                        self.update_memory("assistant", completion_message)
                    else:
                        # No filename found, add to memory directly
                        self.task_completed = True
                        self._task_result = f"Creative content generated:\n{result_summary}"
                        self._deliverable_content = content
                        # Update memory with task completion message
                        completion_message = f"✅ Here's the creative content you requested:\n\n{content}\n\nTask completed successfully! What would you like me to help you with next?"
                        self.update_memory("assistant", completion_message)
                except Exception as e:
                    logger.error(
                        f"[CREATE ERROR] Failed to generate creative content: {e}",
                        exc_info=True,
                    )
                    self.update_memory(
                        "assistant",
                        f"I tried to create the content, but encountered an error: {str(e)}\n\nWhat would you like me to help you with instead?",
                    )
                    self._task_result = f"Error generating content: {str(e)}"

                # Mark task as explicitly handled to prevent task_completer from running
                self._explicit_task_type = "completed"

                # Create a dummy TaskCompleter but mark it as complete to prevent regeneration
                self.task_completer = TaskCompleter(task=prompt)
                self.task_completer._is_complete = True

                return

            # Regular keyword handling for other keywords
            self.task_completer = TaskCompleter(task=prompt)
            self._explicit_task_type = keyword
            self._explicit_task_prompt = prompt
            return
        logger.info(f"Analyzing new task: '{task[:100]}...'")
        # Reset components for new task
        if self.conversation_memory:
            self.conversation_memory.reset_for_new_task()
        self.task_completer = TaskCompleter(task=task)
        self._explicit_task_type = None
        self._explicit_task_prompt = None

        # Reset persistent memory context for new task
        if self.conversation_memory:
            self.conversation_memory.reset()  # Reset persistent memory context for new task
            logger.info("Persistent memory reset for new task.")
        else:
            logger.warning(
                "Conversation memory not initialized during analyze_new_task."
            )

        self.tool_tracker.reset()
        self.url_detector.extract_urls(task)
        self.mentioned_urls = set(
            self.url_detector.mentioned_urls
        )  # Initialize from task

        # Initialize or reset task completer
        self.task_completer = TaskCompleter()  # Re-initialize
        self.task_completer.analyze_task(task)
        logger.info(
            f"Task analyzed. Type: {self.task_completer.task_type}, Missing info: {self.task_completer.get_missing_info()}"
        )

        # Store the current task description
        self.current_task = task
        self.task_completed = False
        self.browser_used = False
        self.visited_urls = set()
        self.step_count = 0  # Reset step counter

        # Reset SmartAskHuman state
        smart_ask_tool = self.available_tools.get_tool("ask_human")
        if isinstance(smart_ask_tool, SmartAskHuman):
            smart_ask_tool.reset_for_new_task()

        # Perform task analysis using TaskAnalyzer
        if self.task_analyzer:
            logger.info("Running TaskAnalyzer...")
            try:
                plan = await self.task_analyzer.analyze_task(task)
                if plan and plan.steps:
                    required_context = plan.all_required_context()
                    if required_context:
                        self.context_manager.set_required_context(
                            list(required_context)
                        )
                        logger.info(
                            f"Task analysis set required context: {required_context}"
                        )

                    # Adjust max_steps based on task complexity if dynamic_steps is enabled
                    if self.dynamic_steps:
                        # Base the number of steps on the plan complexity
                        num_steps = len(plan.steps)
                        # Simple tasks (1-2 steps) need fewer iterations
                        if num_steps <= 2:
                            self.max_steps = max(5, num_steps * 3)  # At least 5 steps
                        # Medium complexity tasks (3-5 steps)
                        elif num_steps <= 5:
                            self.max_steps = max(10, num_steps * 3)  # At least 10 steps
                        # Complex tasks (more than 5 steps)
                        else:
                            # Allow up to 25 steps for complex tasks
                            self.max_steps = min(25, max(15, num_steps * 3))

                        logger.info(
                            f"Dynamically adjusted max_steps to {self.max_steps} based on task complexity"
                        )

                    logger.info(f"📋 Task plan created with {len(plan.steps)} steps:")
                    for i, step in enumerate(plan.steps):
                        log_msg = f"  Step {i+1}: {step.description}"
                        if step.required_context:
                            log_msg += (
                                f" (Requires: {', '.join(step.required_context)})"
                            )
                        logger.info(log_msg)
                else:
                    logger.info("Task analysis did not produce a plan.")

                # Create navigation plan if URLs are present
                if self.mentioned_urls:
                    base_url = list(self.mentioned_urls)[0]  # Use first mentioned URL
                    navigation_plan = self.browser_navigator.create_navigation_plan(
                        base_url, task
                    )
                    if navigation_plan:
                        logger.info(
                            f"🌐 Created website navigation plan ({len(navigation_plan)} steps): {navigation_plan}"
                        )
                    else:
                        logger.info("🌐 No specific navigation plan created.")
            except Exception as e:
                logger.error(
                    f"Error during task analysis or planning: {e}", exc_info=True
                )
        else:
            logger.warning("TaskAnalyzer not available.")

    async def execute_tool(self, command: ToolCall) -> str:
        """Override execute_tool to add memory tracking, suggestions, and robust error handling."""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format provided."

        name = command.function.name
        args_str = command.function.arguments or "{}"
        args = {}
        observation = ""

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON arguments for tool {name}: {args_str}"
            logger.error(error_msg)
            args_with_message = {
                "arguments": args_str,
                "message": "Invalid JSON arguments",
            }
            self.tool_tracker.record_tool_usage(name, args_with_message, "failure")
            return f"Error: {error_msg}"

        if name not in self.available_tools.tool_map:
            error_msg = f"Unknown tool '{name}' called."
            logger.error(error_msg)
            args_with_message = (
                args.copy() if isinstance(args, dict) else {"arguments": args}
            )
            args_with_message["message"] = "Unknown tool"
            self.tool_tracker.record_tool_usage(name, args_with_message, "failure")
            return f"Error: {error_msg}"

        logger.info(f"Executing Tool: {name} with args: {args}")

        # --- Special Handling Logic ---
        if name == "str_replace_editor":
            return await self._handle_str_replace_editor(
                command
            )  # Uses updated command if modified
        if name == "python_execute":
            return await self._handle_python_execute(
                command
            )  # Uses updated command if modified
        if name == "browser_use":
            action = args.get("action") or args.get(
                "url"
            )  # Get action or URL if action is missing
            if not action:
                return "Error: browser_use requires an 'action' or 'url'."
            # Use dedicated handler which returns formatted observation
            return await self.execute_browser_use(action, **args)

        # --- Suggestion Logic for ask_human ---
        if name == "ask_human":
            question = args.get("inquire")
            if question:
                # Check for important keywords that indicate the question should always be asked
                important_keywords = [
                    "confirm",
                    "verify",
                    "approve",
                    "permission",
                    "prefer",
                    "choice",
                    "select",
                    "decide",
                    "opinion",
                    "want",
                    "need",
                    "should",
                    "would you",
                    "do you",
                    "can you",
                    "provide",
                    "overview",
                    "structure",
                    "details",
                    "specifics",
                    "requirements",
                ]

                # Check if the question is a duplicate or semantically similar to recently asked questions
                question_norm = question.strip().lower()
                duplicate_or_similar = False

                # First check for exact duplicates
                for prev_q in self.tool_tracker.asked_questions[
                    -10:
                ]:  # Check the last 10 questions
                    prev_norm = prev_q.strip().lower()
                    if question_norm == prev_norm:
                        logger.warning(f"Exact duplicate question detected: {question}")
                        duplicate_or_similar = True
                        break

                # If not an exact duplicate, check for semantic similarity
                if not duplicate_or_similar:
                    for prev_q in self.tool_tracker.asked_questions[-10:]:
                        similarity = self._calculate_similarity(question, prev_q)
                        if similarity > 0.7:  # High similarity threshold
                            logger.warning(
                                f"Similar question detected (similarity: {similarity:.2f}): {question} vs {prev_q}"
                            )
                            duplicate_or_similar = True
                            break

                # Force asking the question if it contains important keywords AND is not a duplicate/similar
                force_ask = (
                    any(keyword in question.lower() for keyword in important_keywords)
                    and not duplicate_or_similar
                )

                # Check for redundancy
                is_redundant = self._is_redundant_question(question)

                # Record the question in the tool tracker if we're going to ask it
                if not is_redundant or force_ask:
                    self.tool_tracker.record_question(question)

                # Skip redundant questions unless we're forcing them
                if is_redundant and not force_ask:
                    # Check if we've been asking the same question repeatedly
                    # If so, we should just ask it anyway to break the loop
                    repeated_count = sum(
                        1
                        for q in self.tool_tracker.asked_questions[-10:]
                        if self._calculate_similarity(q, question) > 0.8
                    )

                    if repeated_count >= 3:
                        logger.warning(
                            f"Question '{question}' has been filtered multiple times. Forcing it to break potential loop."
                        )
                    else:
                        observation = f"Observation: Question '{question}' seems redundant. Skipping."
                        logger.warning(f"Redundant question skipped: {question}")
                        # Return observation directly, no tool execution needed
                        return observation

                # Only suggest better tools if we're not forcing the question
                if not force_ask:
                    better_tool_suggestion = await self._suggest_better_tool(name, args)
                    if better_tool_suggestion:
                        suggested_tool_name = better_tool_suggestion.get("tool")
                        suggested_args = better_tool_suggestion.get("args", {})
                        logger.info(
                            f"🔄 Intercepted '{name}'. Using better tool: '{suggested_tool_name}' for question: '{question}'"
                        )

                        # Create a new ToolCall for the suggested tool
                        suggested_command = ToolCall(
                            id=f"suggested_{command.id}",
                            type="function",
                            function={
                                "name": suggested_tool_name,
                                "arguments": json.dumps(suggested_args),
                            },
                        )
                        # Execute the suggested tool using this same execute_tool flow (recursive but safe due to checks)
                        return await self.execute_tool(suggested_command)

                if force_ask:
                    logger.info(f"🔑 Forcing important question: {question}")

            # If no suggestion or not a question, proceed with SmartAskHuman via super().execute_tool

        # --- Standard Tool Execution via Base Class ---
        try:
            # Let ToolCallAgent handle the actual execution and basic observation formatting
            observation = await super().execute_tool(command)
            # Record success (assuming super().execute_tool raises exception on failure)
            self.tool_tracker.record_tool_usage(name, args, "success")
            logger.info(f"Tool '{name}' executed successfully.")
            logger.debug(f"Tool '{name}' observation: {observation[:500]}...")

        except Exception as e:
            error_msg = f"Error executing tool '{name}' with args {args_str}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            args_with_message = (
                args.copy() if isinstance(args, dict) else {"arguments": args}
            )
            args_with_message["message"] = str(e)
            self.tool_tracker.record_tool_usage(name, args_with_message, "failure")
            # Return a formatted error observation for the LLM
            observation = f"Error: Tool `{name}` failed with error: {str(e)}"

        return observation  # Return formatted observation (success or error)

    # List of realistic user agents for rotation
    REALISTIC_USER_AGENTS: ClassVar[List[str]] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/112.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.42",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.58",
    ]

    def _get_random_user_agent(self):
        """Return a random user agent from the list of realistic user agents."""
        import random

        return random.choice(self.REALISTIC_USER_AGENTS)

    async def _apply_stealth_techniques(self, **kwargs):
        """Apply various stealth techniques to avoid bot detection."""
        url = kwargs.get("url", "")
        stealth_script = """
        () => {
            // Override navigator properties to make detection harder
            const newProto = navigator.__proto__;
            delete newProto.webdriver;

            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === "notifications" ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );

            // Prevent automation detection
            Object.defineProperty(navigator, "plugins", {
                get: () => [
                    {
                        0: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "PDF Viewer",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "PDF Viewer"
                    },
                    {
                        0: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "Chrome PDF Viewer",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chrome PDF Viewer"
                    },
                    {
                        0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "Chrome PDF Plugin",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chrome PDF Plugin"
                    }
                ]
            });

            // Add some randomization to make fingerprinting harder
            const randomFactor = Math.floor(Math.random() * 10) / 100;
            Object.defineProperty(window, "devicePixelRatio", {
                get: function() { return Math.floor((window.screen.width / window.screen.availWidth + randomFactor) * 100) / 100; }
            });

            // Simulate human-like behavior
            const randomScrollAmount = Math.floor(Math.random() * 100) + 50;
            setTimeout(() => { window.scrollTo(0, randomScrollAmount); }, Math.random() * 1000 + 500);
            setTimeout(() => { window.scrollTo(0, 0); }, Math.random() * 1000 + 1500);

            return "Stealth techniques applied";
        }
        """

        try:
            # Apply stealth techniques by using a combination of browser actions
            # 1. Randomize scrolling to simulate human behavior
            scroll_amount = random.randint(100, 300)
            await self.execute_browser_use("scroll_down", scroll_amount=scroll_amount)
            await asyncio.sleep(0.3)
            await self.execute_browser_use(
                "scroll_up", scroll_amount=scroll_amount // 2
            )

            # 2. Add a slight delay to simulate human reading
            await asyncio.sleep(0.5)
            logger.info(
                f"Applied stealth techniques for {url if url else 'current page'}"
            )
        except Exception as e:
            logger.warning(f"Error applying stealth techniques: {str(e)}")

    async def execute_browser_use(self, action_or_url: str, **kwargs):
        """Execute browser use tool with enhanced navigation, overlay handling, and context storage."""
        # Apply user agent rotation
        random_user_agent = self._get_random_user_agent()
        kwargs["headers"] = kwargs.get("headers", {})
        kwargs["headers"]["User-Agent"] = random_user_agent
        action = action_or_url
        url = kwargs.get("url", "")
        final_content = ""  # Initialize content string

        try:
            self.browser_used = True

            # Normalize action/url if called simply with a URL
            if action.startswith("http"):
                url = action
                action = "go_to_url"
                kwargs["url"] = url
            elif (
                not url
                and action != "get_html"
                and "selector" not in kwargs
                and "script" not in kwargs
                and "key" not in kwargs
            ):
                raise ValueError(
                    f"Invalid browser_use call: action='{action}' needs url, selector, script, or key."
                )

            logger.info(
                f"Executing browser action: '{action}' | URL: '{url}' | Other args: { {k:v for k,v in kwargs.items() if k != 'url'} }"
            )

            browser_tool = self.available_tools.get_tool("browser_use")
            if not browser_tool:
                raise RuntimeError("BrowserUseTool not found.")

            # --- Core Browser Action Execution ---
            raw_result = await browser_tool.execute(action, **kwargs)
            raw_content_str = str(raw_result) if raw_result is not None else ""
            logger.debug(
                f"Browser action '{action}' raw result type: {type(raw_result)}, length: {len(raw_content_str)}"
            )

            # --- Post-Action Processing (Navigation, Content Extraction, etc.) ---
            if action == "go_to_url" and url:
                self.visited_urls.add(url)
                logger.info(
                    f"Navigation to {url} successful (initial). Applying stealth techniques and handling overlays..."
                )

                # Apply stealth techniques to avoid bot detection
                await self._apply_stealth_techniques(url=url)

                # Handle Overlays AFTER Navigation attempt
                await self._handle_website_overlays()

                # Get Content AFTER Overlays Handled
                extracted_content = ""
                try:
                    # Scroll first to trigger lazy loading
                    await browser_tool.execute("scroll_down", scroll_amount=500)
                    await asyncio.sleep(0.5)
                    await browser_tool.execute("scroll_up", scroll_amount=500)
                    time.sleep(0.5)

                    # Try extracting main content via JS (more lightweight)
                    js_extract_script = """
                     function extractMainText() {
                         const selectors = ['main', 'article', '.content', '#content', '.main', '#main', '[role="main"]', '.post-content', '.entry-content'];
                         for (const selector of selectors) {
                             const element = document.querySelector(selector);
                             if (element && element.innerText.trim().length > 200) return element.innerText;
                         }
                         return document.body.innerText; // Fallback
                     }
                     extractMainText();
                     """
                    js_content = await browser_tool.execute(
                        "extract_content",
                        goal="Extract main textual content from the page",
                    )
                    if js_content and len(str(js_content)) > 100:
                        extracted_content = str(js_content).strip()
                        logger.info(
                            f"Extracted main text via JS for {url}, length: {len(extracted_content)}"
                        )
                    else:
                        logger.info(
                            "JS text extraction failed or too short, getting full HTML."
                        )
                        html = await browser_tool.execute(
                            "extract_content", goal="Get the full HTML of the page"
                        )
                        if html and not str(html).startswith("Error:"):
                            extracted_content = self._extract_main_content(str(html))
                            logger.info(
                                f"Extracted main content via HTML parsing for {url}, length: {len(extracted_content)}"
                            )
                        else:
                            logger.warning(
                                f"Failed to get HTML for {url}. Using raw result if available."
                            )
                            extracted_content = (
                                raw_content_str if len(raw_content_str) > 50 else ""
                            )

                except Exception as e:
                    logger.error(
                        f"Error extracting content after navigation to {url}: {e}",
                        exc_info=True,
                    )
                    extracted_content = f"Error extracting content from {url}: {e}"  # Report error in content

                # Store & Contextualize meaningful content
                if (
                    extracted_content
                    and not extracted_content.startswith("Error:")
                    and len(extracted_content) > 50
                ):
                    memory_text = (
                        f"Content from {url}:\n{extracted_content[:self.max_observe]}"
                        + ("..." if len(extracted_content) > self.max_observe else "")
                    )
                    self.add_to_context(
                        memory_text,
                        source="browser",
                        tags=["web_content", "visited_url", url],
                    )

                    # Process for task relevance - enhanced for creative tasks
                    if self.task_completer and hasattr(
                        self.task_completer, "task_type"
                    ):
                        task_type = self.task_completer.task_type or ""
                        if task_type.lower() in ["poem", "story", "creative"]:
                            logger.info(
                                f"Processing content for creative task: {task_type}"
                            )
                            # Extract key facts and details
                            key_facts = []
                            soup = BeautifulSoup(extracted_content, "html.parser")

                            # Extract title if available
                            title = soup.title.string if soup.title else ""
                            if title and len(title) < 150:  # Reasonable title length
                                key_facts.append(f"Title: {title.strip()}")

                            # Extract key sections
                            section_headers = [
                                "early life",
                                "biography",
                                "achievements",
                                "legacy",
                                "personal life",
                            ]
                            for header in section_headers:
                                section = self._extract_section(
                                    extracted_content, header
                                )
                                if (
                                    section and len(section) > 50
                                ):  # Only include meaningful sections
                                    key_facts.append(f"{header.title()}: {section}")

                            # Extract important dates
                            years = re.findall(r"\b(?:19|20)\d{2}\b", extracted_content)
                            if years:
                                key_facts.append(
                                    f"Key years: {', '.join(sorted(set(years)))}"
                                )

                            # Add to task completer if we found useful information
                            if key_facts:
                                facts_text = "\n".join(key_facts)
                                self.task_completer.add_information(
                                    "key_facts", facts_text
                                )
                                logger.info(
                                    f"Extracted {len(key_facts)} key facts for creative content"
                                )

                    # Process for general task relevance
                    self._process_content_for_task(url, extracted_content)
                    final_content = (
                        extracted_content  # Use extracted content for observation
                    )

                else:  # Handle cases where extraction failed or yielded little
                    logger.warning(f"No meaningful content extracted from {url}.")
                    fail_msg = (
                        f"Visited {url} but could not extract significant content."
                    )
                    self.add_to_context(
                        fail_msg,
                        source="browser",
                        tags=["web_content", "visited_url", url, "extraction_failed"],
                    )
                    final_content = fail_msg  # Use this message for observation

            else:
                # For other actions (click, evaluate, get_html), use the raw result
                final_content = raw_content_str
                # Optionally store results of evaluations etc. in memory if needed
                if (
                    action == "evaluate" and final_content and len(final_content) < 2000
                ):  # Store non-huge eval results
                    self.add_to_context(
                        f"Result of evaluate script: {final_content[:300]}...",
                        source="browser_evaluate",
                        priority="low",
                        tags=["evaluate_result"],
                    )

            # --- Prepare Observation ---
            observation_content = final_content[: self.max_observe] + (
                "..." if len(final_content) > self.max_observe else ""
            )
            observation = f"Observed output of cmd `browser_use` ({action}) executed:\n{observation_content}"
            return observation

        except Exception as e:
            error_msg = (
                f"Error executing browser action '{action}' with {kwargs}: {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            self.add_to_context(
                f"Failed browser action: {action}. Error: {str(e)}",
                source="browser_error",
                priority="high",
                tags=["error", "browser"],
            )
            return f"Error: Browser action `{action}` failed: {str(e)}"

    async def enhanced_web_research(
        self,
        query: str,
        max_sites: int = 3,
        min_content_length: int = 500,
        max_content_length: int = 10000,
    ) -> Dict[str, Any]:
        """
        Enhanced web research with better content processing and storage.

        Args:
            query: The search query
            max_sites: Maximum number of sites to visit
            min_content_length: Minimum content length to consider
            max_content_length: Maximum content length to process

        Returns:
            Dict containing research results and metadata
        """
        logger.info(f"🔍 Starting enhanced web research for: {query}")
        research_results = {
            "query": query,
            "sources": [],
            "total_content_length": 0,
            "start_time": time.time(),
            "status": "in_progress",
        }

        try:
            # Step 1: Perform initial search
            logger.info(f"🔎 Searching for: {query}")
            search_results = await self.perform_web_research(query, max_sites)

            if not search_results or "sources" not in search_results:
                research_results["status"] = "no_results"
                logger.warning("No search results found")
                return research_results

            # Step 2: Process each search result
            for i, source in enumerate(search_results["sources"][:max_sites], 1):
                try:
                    url = source.get("url", "")
                    if not url or url in [
                        s.get("url") for s in research_results["sources"]
                    ]:
                        continue

                    logger.info(
                        f"🌐 Visiting source {i}/{min(max_sites, len(search_results['sources']))}: {url}"
                    )

                    # Navigate to the URL
                    await self.execute_browser_use("go_to_url", url=url)
                    await asyncio.sleep(2)  # Allow page to load

                    # Extract content
                    browser_tool = self.available_tools.get_tool("browser_use")
                    if not browser_tool:
                        continue

                    # Use the LLM to generate the content
                    try:
                        response = await self.llm.ask(
                            messages, stream=False, temperature=0.8
                        )
                        content = (
                            response["content"]
                            if isinstance(response, dict) and "content" in response
                            else response
                        )
                        logger.info(f"Creative content generated successfully.")
                    except Exception as e:
                        logger.error(f"Error generating creative content: {e}")
                        content = "[Error generating content]"

                    # Save to file if filename was specified
                    if filename and content:
                        try:
                            import os

                            workspace_dir = os.path.join(os.getcwd(), "workspace")
                            file_path = (
                                filename
                                if os.path.isabs(filename)
                                else os.path.join(workspace_dir, filename)
                            )
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            logger.info(
                                f"[FILE SAVE] Creative content saved to: {file_path}"
                            )
                        except Exception as file_err:
                            logger.error(
                                f"[FILE SAVE ERROR] Could not save file '{filename}': {file_err}"
                            )

                        query = query

                    # Process for task relevance
                    if self.task_completer:
                        self._process_content_for_task(url, content)

                    # Add to context memory
                    self.add_to_context(
                        f"Content from {url} (truncated):\n{content[:1000]}...",
                        source="web_research",
                        tags=["web_content", "research", url],
                    )

                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}", exc_info=True)
                    continue

            research_results["status"] = "completed"
            research_results["end_time"] = time.time()
            research_results["duration"] = (
                research_results["end_time"] - research_results["start_time"]
            )

            logger.info(
                f"✅ Research completed. Processed {len(research_results['sources'])} sources, "
                f"total {research_results['total_content_length']} characters"
            )

            return research_results

        except Exception as e:
            research_results["status"] = "error"
            research_results["error"] = str(e)
            logger.error(f"Error in enhanced_web_research: {str(e)}", exc_info=True)
            return research_results

    async def perform_web_research(
        self, query: str, max_sites: int = 4
    ) -> Dict[str, Any]:
        """Perform web research: Search (DDG HTML) -> Extract URLs -> Visit -> Extract Content -> Analyze.

        Returns:
            Dict containing search results and sources
        """
        urls_to_visit = []
        search_engine_url = ""
        logger.info(f"🔬 Starting web research for query: '{query}'")

        # --- Step 1: Search using DuckDuckGo HTML ---
        try:
            search_query_encoded = requests.utils.quote(query)
            search_engine_url = (
                f"https://html.duckduckgo.com/html/?q={search_query_encoded}"
            )
            logger.info(f"Searching DuckDuckGo HTML: {search_engine_url}")

            # Use execute_browser_use to navigate and get initial HTML (it handles overlays)
            # We need the HTML content here, not just confirmation
            await self.execute_browser_use("go_to_url", url=search_engine_url)
            # Handle potential overlays on the search results page itself
            await self._handle_website_overlays()
            # Get the page content directly from the browser tool
            browser_tool = self.available_tools.get_tool("browser_use")
            if not browser_tool:
                logger.error("BrowserUseTool not found.")
                return "Error: Browser tool not available."

            # Get the current page content
            try:
                # First try to get the page state which includes the HTML
                state_result = await browser_tool.get_current_state()

                # Save the state output to a debug file
                try:
                    debug_dir = Path("debug_output")
                    debug_dir.mkdir(exist_ok=True)
                    with open(
                        debug_dir / "browser_state.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(
                            str(state_result.output)
                            if state_result and state_result.output
                            else "No output"
                        )
                    logger.info("Saved browser state to debug_output/browser_state.txt")
                except Exception as e:
                    logger.error(f"Failed to save debug state: {e}")

                if state_result and not state_result.error:
                    # Extract URLs directly from the state output using regex
                    state_text = str(state_result.output)
                    # First, try to extract URLs from the interactive_elements section
                    # This section contains links with their text and URLs
                    interactive_elements = ""
                    if "interactive_elements" in state_text:
                        # Extract the interactive_elements section
                        match = re.search(
                            r'"interactive_elements":\s*"(.+?)"', state_text, re.DOTALL
                        )
                        if match:
                            interactive_elements = match.group(1).replace("\\n", "\n")

                    # Extract URLs from the interactive elements section
                    extracted_urls = []
                    if interactive_elements:
                        # Look for URLs in the interactive elements section
                        # Pattern for URLs like www.example.com/path/ or https://example.com/path/
                        url_patterns = [
                            r"\[\d+\]<a\s+([\w.-]+\.[\w]{2,}/[^\s>]+)/>",  # Domain with path
                            r"\[\d+\]<a\s+([\w.-]+\.[\w]{2,})/>",  # Just domain
                            r"\[\d+\]<a\s+https?://([^\s>]+)/>",  # Full URL
                            r'https?://([\w.-]+\.[\w]{2,}/[^\s>"]+)',  # URLs in text
                            r'([\w.-]+\.[\w]{2,}/[^\s>"]+)',  # Domain with path in text
                        ]

                        for pattern in url_patterns:
                            for match in re.finditer(pattern, interactive_elements):
                                url = match.group(1).strip()
                                if not url.startswith("http"):
                                    url = f"https://{url}"
                                extracted_urls.append(url)

                        # Also look for URLs in the numbered links [X]<a url>
                        numbered_links = re.finditer(
                            r"\[(\d+)\]<a\s+([^/]+)/>", interactive_elements
                        )
                        for match in numbered_links:
                            index = match.group(1)
                            link_text = match.group(2).strip()

                            # Look for the corresponding URL that might appear after this link
                            # Often in format [X]<a www.example.com/>
                            if re.match(r"^[\w.-]+\.[\w]{2,}", link_text):
                                if not link_text.startswith("http"):
                                    extracted_urls.append(f"https://{link_text}")
                                else:
                                    extracted_urls.append(link_text)

                    # If we didn't find URLs in the interactive elements, try general regex
                    if not extracted_urls:
                        all_urls = self._extract_urls(state_text)
                        extracted_urls = all_urls

                    # Filter out DuckDuckGo internal URLs and image/document files
                    urls_to_visit = [
                        url
                        for url in extracted_urls
                        if "duckduckgo.com" not in url
                        and not url.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf")
                        )
                    ]

                    logger.info(
                        f"Extracted {len(urls_to_visit)} URLs directly from browser state."
                    )
                    search_page_html = (
                        state_text  # Keep the state text for fallback parsing
                    )
                else:
                    logger.warning(
                        f"Failed to get page state: {state_result.error if state_result else 'No result'}"
                    )
                    search_page_html = ""  # Empty string will trigger fallback
                    urls_to_visit = []
            except Exception as e:
                logger.error(f"Error getting page content: {e}")
                search_page_html = ""  # Empty string will trigger fallback

            # If we don't have URLs from the direct extraction, try fallback methods
            if not urls_to_visit:
                if (
                    not search_page_html
                    or str(search_page_html).startswith("Error:")
                    or len(str(search_page_html)) < 1500
                ):
                    logger.error(
                        f"Failed to retrieve valid HTML from DDG search: {str(search_page_html)[:200]}"
                    )
                    # Try Google as a fallback
                    logger.info("Trying Google as a fallback search engine")
                    try:
                        google_search_url = (
                            f"https://www.google.com/search?q={search_query_encoded}"
                        )
                        logger.info(f"Searching Google: {google_search_url}")

                        # Navigate to Google search with our user agent and stealth techniques
                        await self.execute_browser_use(
                            "go_to_url", url=google_search_url
                        )

                        # Apply stealth techniques
                        await self._apply_stealth_techniques(url=google_search_url)

                        # Handle potential overlays
                        await self._handle_website_overlays()

                        # Wait a bit for dynamic content to load
                        time.sleep(2)

                        # Get the page state which should include links
                        google_state = await browser_tool.get_current_state()
                        if google_state and not google_state.error:
                            # Extract URLs directly from the state output using regex
                            google_state_text = str(google_state.output)
                            all_urls = self._extract_urls(google_state_text)
                            # Filter out Google internal URLs and document files
                            google_urls = [
                                url
                                for url in all_urls
                                if not any(
                                    x in url.lower()
                                    for x in [
                                        "google.com/search",
                                        "google.com/imgres",
                                        "accounts.google",
                                        "/amp/",
                                        "webcache.googleusercontent",
                                        "translate.google",
                                    ]
                                )
                                and not url.lower().endswith(
                                    (
                                        ".pdf",
                                        ".doc",
                                        ".docx",
                                        ".ppt",
                                        ".pptx",
                                        ".xls",
                                        ".xlsx",
                                    )
                                )
                            ]
                            logger.info(
                                f"Extracted {len(google_urls)} URLs from Google state"
                            )
                            urls_to_visit = google_urls[:max_sites]
                        else:
                            # Try Bing as a last resort
                            logger.info(
                                "Google fallback failed, trying Bing as a last resort"
                            )
                            bing_urls = await self._fallback_bing_search(query)

                            if bing_urls:
                                urls_to_visit = bing_urls[:max_sites]
                                logger.info(
                                    f"Successfully extracted {len(urls_to_visit)} URLs from Bing fallback search"
                                )
                            else:
                                # If all search engines fail, return an error
                                logger.error(
                                    "All search engines failed to provide usable URLs"
                                )
                                return "Error: Could not extract any URLs from multiple search engines. Please try a different query or try again later."
                    except Exception as e:
                        logger.error(f"Error during fallback search: {str(e)}")
                        return f"Error: Failed to retrieve search results from both DuckDuckGo and Google. {str(e)}"
                else:
                    logger.info(
                        f"Successfully retrieved DDG HTML results page (length: {len(str(search_page_html))})."
                    )
                    # Try the traditional HTML parsing as a fallback
                    parsed_urls = self._extract_ddg_search_results(
                        str(search_page_html)
                    )
                    if parsed_urls:
                        urls_to_visit = parsed_urls
                        logger.info(
                            f"Extracted {len(urls_to_visit)} URLs from DDG HTML parsing."
                        )
                    else:
                        # If still no URLs, try a more aggressive approach with regex
                        logger.info(
                            "No URLs from HTML parsing, trying aggressive regex approach"
                        )
                        all_urls = self._extract_urls(str(search_page_html))
                        urls_to_visit = [
                            url
                            for url in all_urls
                            if "duckduckgo.com" not in url
                            and not url.lower().endswith(
                                (".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf")
                            )
                        ]
                        logger.info(
                            f"Extracted {len(urls_to_visit)} URLs from aggressive regex approach."
                        )

            if urls_to_visit:
                logger.info(f"Extracted {len(urls_to_visit)} potential URLs from DDG.")
                urls_to_visit = [
                    u
                    for u in urls_to_visit
                    if not u.lower().endswith((".pdf", ".xml", ".zip"))
                ][:max_sites]
                logger.info(
                    f"Selected {len(urls_to_visit)} URLs to visit: {urls_to_visit}"
                )
            else:
                logger.warning(
                    f"No valid URLs extracted from DDG results for: '{query}'"
                )
                return f"Web research for '{query}' did not yield any result URLs to visit."

        except Exception as e:
            logger.error(
                f"Error during search engine interaction or URL extraction: {e}",
                exc_info=True,
            )
            return f"Error: Failed during web search phase for '{query}'. {str(e)}"

        # --- Step 2: Visit URLs and Extract Content ---
        processed_content = []  # Store {'url': url, 'content': text}
        visited_count = 0
        if not urls_to_visit:
            return "No search result URLs found to visit."

        for i, url in enumerate(urls_to_visit):
            if visited_count >= max_sites:
                break
            logger.info(f"Visiting URL {i+1}/{len(urls_to_visit)}: {url}")
            max_retries = 2  # Number of retries for each URL
            retry_count = 0
            success = False

            while retry_count <= max_retries and not success:
                try:
                    if retry_count > 0:
                        logger.info(f"Retry attempt {retry_count} for URL: {url}")

                    # execute_browser_use handles navigation, overlays, extraction
                    # It returns the observation string which includes the content or error
                    observation = await self.execute_browser_use("go_to_url", url=url)

                    # Check if we got a timeout or connection error that might benefit from a retry
                    if (
                        "timeout" in observation.lower()
                        or "connection" in observation.lower()
                        or "failed to load" in observation.lower()
                    ):
                        if retry_count < max_retries:
                            logger.warning(
                                f"Timeout or connection issue for {url}, will retry. Error: {observation[:100]}..."
                            )
                            retry_count += 1
                            time.sleep(2)  # Wait before retry
                            continue

                    # Extract content from the observation string if successful
                    if observation.startswith(
                        "Observed output"
                    ) and not observation.startswith("Error:"):
                        # Extract content after the preamble
                        content_part = observation.split("executed:\n", 1)[-1]

                        # Check if we got meaningful content
                        if content_part and len(content_part.strip()) > 100:
                            if (
                                not content_part.startswith("Visited")
                                and "extract significant content" not in content_part
                            ):
                                logger.info(
                                    f"Successfully retrieved content for {url}, length: {len(content_part)}"
                                )

                                # Store the content with the URL
                                processed_content.append(
                                    {"url": url, "content": content_part}
                                )
                                visited_count += 1
                                success = True
                            else:
                                # Content extraction issue, might benefit from a retry with a different approach
                                if retry_count < max_retries:
                                    logger.warning(
                                        f"Content extraction failed for {url}, will retry with different approach"
                                    )
                                    # Try to extract content directly using a more aggressive approach
                                    try:
                                        # Get the current page state which should include the HTML
                                        browser_tool = self.available_tools.get_tool(
                                            "browser_use"
                                        )
                                        state_result = (
                                            await browser_tool.get_current_state()
                                        )

                                        if state_result and not state_result.error:
                                            # Extract the HTML and process it directly
                                            html_content = str(state_result.output)
                                            extracted_content = (
                                                self._extract_main_content(html_content)
                                            )

                                            if (
                                                extracted_content
                                                and len(extracted_content) > 200
                                            ):
                                                logger.info(
                                                    f"Successfully extracted content directly from HTML for {url}"
                                                )
                                                processed_content.append(
                                                    {
                                                        "url": url,
                                                        "content": extracted_content,
                                                    }
                                                )
                                                visited_count += 1
                                                success = True
                                            else:
                                                logger.warning(
                                                    f"Direct HTML extraction failed for {url}"
                                                )
                                                retry_count += 1
                                        else:
                                            logger.warning(
                                                f"Failed to get page state for {url}"
                                            )
                                            retry_count += 1
                                    except Exception as e:
                                        logger.error(
                                            f"Error during direct HTML extraction for {url}: {e}"
                                        )
                                        retry_count += 1
                                else:
                                    logger.warning(
                                        f"Skipping URL {url} after {max_retries} failed extraction attempts"
                                    )
                                    success = True  # Mark as done to move on
                        else:
                            # Empty or very short content, might be a loading issue
                            if retry_count < max_retries:
                                logger.warning(
                                    f"Empty or very short content for {url}, will retry"
                                )
                                retry_count += 1
                                time.sleep(2)  # Wait before retry
                            else:
                                logger.warning(
                                    f"Skipping URL {url} due to persistent empty content issue"
                                )
                                success = True  # Mark as done to move on
                    else:
                        # Error in observation
                        if retry_count < max_retries:
                            logger.warning(
                                f"Error for {url}, will retry. Error: {observation[:100]}..."
                            )
                            retry_count += 1
                            time.sleep(2)  # Wait before retry
                        else:
                            logger.warning(
                                f"Skipping URL {url} after {max_retries} failed attempts. Last error: {observation[:200]}"
                            )
                            success = True  # Mark as done to move on

                except Exception as e:
                    if retry_count < max_retries:
                        logger.error(f"Exception processing URL {url}: {e}, will retry")
                        retry_count += 1
                        time.sleep(2)  # Wait before retry
                    else:
                        logger.error(
                            f"Critical error processing URL {url} after {max_retries} retries: {e}"
                        )
                        success = True  # Mark as done to move on

            # Add a delay between URLs to avoid rate limiting
            if i < len(urls_to_visit) - 1:
                time.sleep(2)  # Slightly longer delay between URLs

        # --- Step 3: Analyze Collected Content ---
        logger.info(
            f"Finished visiting URLs. Collected content from {visited_count} sites."
        )

        if not processed_content:
            logger.warning(f"No content collected for query: '{query}'")
            return "Web research completed, but no content was successfully extracted from the result URLs."

        combined_content_for_analysis = ""
        sources_metadata = []
        for idx, item in enumerate(processed_content):
            combined_content_for_analysis += (
                f"\n\n--- SOURCE {idx+1}: {item['url']} ---\n\n{item['content']}"
            )
            sources_metadata.append({"id": idx + 1, "url": item["url"]})

        logger.info(
            f"Combined content length for analysis: {len(combined_content_for_analysis)}"
        )

        analysis_result = await self._analyze_combined_content(
            combined_content_for_analysis, query, sources_metadata
        )
        return f"Web research result for query '{query}':\n{analysis_result}"  # Add context

    async def _fallback_bing_search(self, query: str) -> List[str]:
        """Use Bing as a fallback search engine when other methods fail."""
        try:
            logger.info(f"Using Bing fallback search for query: '{query}'")
            search_query_encoded = requests.utils.quote(query)
            bing_url = f"https://www.bing.com/search?q={search_query_encoded}"

            # Navigate to Bing with our stealth techniques (user agent rotation is handled by execute_browser_use)
            await self.execute_browser_use("go_to_url", url=bing_url)

            # Apply stealth techniques
            await self._apply_stealth_techniques(url=bing_url)

            # Handle potential overlays
            await self._handle_website_overlays()

            # Add a delay to let the page load
            time.sleep(2)

            # Get the browser state which should include the page content
            browser_tool = self.available_tools.get_tool("browser_use")
            if not browser_tool:
                logger.error("BrowserUseTool not found.")
                return []

            # Get the current page state
            state_result = await browser_tool.get_current_state()
            if state_result and not state_result.error:
                # Extract URLs directly from the state output using regex
                state_text = str(state_result.output)
                all_urls = self._extract_urls(state_text)
                # Filter out Bing internal URLs and document files
                urls = [
                    url
                    for url in all_urls
                    if not any(
                        x in url.lower()
                        for x in [
                            "bing.com/search",
                            "bing.com/images",
                            "microsoft.com",
                            "msn.com",
                        ]
                    )
                    and not url.lower().endswith(
                        (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")
                    )
                ]
                logger.info(f"Extracted {len(urls)} URLs from Bing state")
                return urls
            else:
                logger.warning("Failed to get Bing page state")
                return []
        except Exception as e:
            logger.error(f"Error during Bing fallback search: {e}")
            return []

    def _extract_google_search_results(self, html: str) -> List[str]:
        """Extracts organic search result URLs from Google HTML content."""
        urls = set()
        try:
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/url?q="):
                    try:
                        parsed_url = urlparse(href)
                        query_params = parse_qs(parsed_url.query)
                        if "q" in query_params:
                            actual_url = query_params["q"][0]
                            if (
                                actual_url.startswith("http")
                                and "google.com" not in urlparse(actual_url).netloc
                            ):
                                urls.add(unquote(actual_url))
                    except Exception as e:
                        logger.debug(f"Error parsing Google redirect {href}: {e}")
                elif (
                    href.startswith("http")
                    and "google.com" not in urlparse(href).netloc
                ):
                    # Check parent structure for direct links (less reliable)
                    parent_h3 = link.find_parent("h3")
                    if parent_h3:
                        result_block = parent_h3.find_parent(
                            ["div", "li"], class_=re.compile(r"\bg\b|rc|kvH3mc|VtXmk")
                        )
                        if result_block:
                            urls.add(unquote(href))

            logger.info(f"Extracted {len(urls)} unique URLs from Google HTML.")
            return list(urls)
        except Exception as e:
            logger.error(
                f"Error parsing Google search results HTML: {e}", exc_info=True
            )
            return []

    def _extract_ddg_search_results(self, html: str) -> List[str]:
        """Extracts organic search result URLs from DuckDuckGo HTML version."""
        urls = set()
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Try multiple selector approaches to be more robust
            selectors = [
                "div.result a.result__a",  # Traditional DDG HTML results
                "div.result a[href]",  # Any links in result divs
                "div.links_main a[href]",  # Alternative structure
                "h2 a[href]",  # Headers with links
                "a.result__url",  # URL class
                "a[data-testid='result-title-a']",  # Newer test IDs
                "a[href^='/l/?']",  # DDG redirect links
                "a[href^='https://']",  # Direct https links
                "a[href^='http://']",  # Direct http links
            ]

            # Try each selector approach
            for selector in selectors:
                for link_tag in soup.select(selector):
                    if link_tag and link_tag.has_attr("href"):
                        href = link_tag["href"]
                        try:
                            # Handle DDG redirect links
                            if href.startswith("/l/?"):
                                parsed_href = urlparse(href)
                                query_params = parse_qs(parsed_href.query)
                                if "uddg" in query_params:
                                    actual_url = query_params["uddg"][0]
                                    if actual_url.startswith("http"):
                                        urls.add(unquote(actual_url))
                                elif "ud" in query_params:
                                    actual_url = query_params["ud"][0]
                                    if actual_url.startswith("http"):
                                        urls.add(unquote(actual_url))
                            # Handle direct links
                            elif href.startswith("http"):
                                # Skip DuckDuckGo internal links
                                if "duckduckgo.com" not in urlparse(href).netloc:
                                    urls.add(unquote(href))
                        except Exception as e:
                            logger.debug(f"Error parsing DDG link {href}: {e}")

            # If we still don't have any URLs, try a more aggressive approach
            if not urls:
                logger.info(
                    "No URLs found with standard selectors, trying aggressive approach"
                )
                # Get all links and filter out non-http and DuckDuckGo internal links
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    try:
                        if href.startswith("http") and "duckduckgo.com" not in href:
                            urls.add(unquote(href))
                        elif href.startswith("/l/?"):
                            parsed_href = urlparse(href)
                            query_params = parse_qs(parsed_href.query)
                            for param in ["uddg", "ud"]:
                                if param in query_params:
                                    actual_url = query_params[param][0]
                                    if actual_url.startswith("http"):
                                        urls.add(unquote(actual_url))
                    except Exception as e:
                        logger.debug(f"Error in aggressive parsing for {href}: {e}")

            logger.info(f"Extracted {len(urls)} unique URLs from DuckDuckGo HTML.")
            return list(urls)
        except Exception as e:
            logger.error(
                f"Error parsing DuckDuckGo search results HTML: {e}", exc_info=True
            )
            return []

    def _extract_main_content(self, html: str) -> str:
        """Extract the main textual content from an HTML page using multiple strategies."""
        if not html:
            return ""
        try:
            # Try to parse the HTML with the best available parser
            try:
                soup = BeautifulSoup(html, "lxml")  # Prefer lxml for speed/robustness
            except ImportError:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                except Exception as parse_err:
                    logger.error(
                        f"HTML parsing failed: {parse_err}. Falling back to regex."
                    )
                    text = re.sub(
                        r"<script.*?</script>|<style.*?</style>",
                        "",
                        html,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text[: self.max_observe]

            # Remove noise elements that typically don't contain main content
            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "form",
                    "button",
                    "iframe",
                    "noscript",
                    "figure",
                    "img",
                    "svg",
                    "path",
                    "meta",
                    "link",
                    "head",
                    "[class*=cookie]",
                    "[class*=banner]",
                    "[class*=menu]",
                    "[class*=sidebar]",
                    "[class*=ad-]",
                    "[class*=popup]",
                    "[id*=cookie]",
                    "[id*=banner]",
                    "[id*=menu]",
                    "[id*=sidebar]",
                    "[id*=ad-]",
                    "[id*=popup]",
                ]
            ):
                try:
                    element.decompose()
                except Exception as e:
                    logger.debug(f"Failed to remove noise element: {e}")

            # Strategy 1: Look for common content containers using CSS selectors
            main_content_element = None
            selectors = [
                "article",
                "main",
                ".main-content",
                "#main-content",
                ".post-content",
                ".entry-content",
                ".page-content",
                ".content",
                "#content",
                ".article-body",
                "#bodyContent",
                ".story-content",
                ".article",
                ".post",
                ".entry",
                "[role=main]",
                "[itemprop=articleBody]",
                "[itemprop=mainContentOfPage]",
                ".blog-post",
                ".news-article",
                ".text-content",
                ".main-text",
                ".document-content",
                ".cms-content",
                ".rich-text",
                ".markdown-body",
            ]

            # Try each selector and pick the one with substantial content
            for selector in selectors:
                try:
                    elements = soup.select(selector)
                    for element in elements:
                        text_len = len(element.get_text(strip=True))
                        if text_len > 300:
                            # Check if this element has a good text-to-link ratio
                            links_text = sum(
                                len(a.get_text(strip=True))
                                for a in element.find_all("a")
                            )
                            if (
                                links_text < text_len * 0.7
                            ):  # Less than 70% of text is in links
                                main_content_element = element
                                logger.info(
                                    f"Found main content using selector: '{selector}' (length: {text_len})"
                                )
                                break
                    if main_content_element:
                        break
                except Exception as e:
                    logger.debug(f"Selector '{selector}' failed: {e}")

            # Strategy 2: Find the largest text block with good text-to-link ratio
            if not main_content_element:
                largest_container = None
                max_len = 0
                min_content_length = 500  # Minimum text length to consider

                # Check div and section elements for large text blocks
                for tag in soup.find_all(["div", "section", "article", "main"]):
                    try:
                        # Skip elements that are likely navigation or sidebars
                        skip_classes = [
                            "nav",
                            "menu",
                            "sidebar",
                            "footer",
                            "header",
                            "comment",
                        ]
                        if any(
                            cls in (tag.get("class", []) or []) for cls in skip_classes
                        ):
                            continue

                        text = tag.get_text(strip=True)
                        text_len = len(text)

                        if text_len > max_len and text_len > min_content_length:
                            # Calculate text in links to avoid navigation-heavy sections
                            links_text_len = sum(
                                len(a.get_text(strip=True)) for a in tag.find_all("a")
                            )

                            # Good content should have a low percentage of text in links
                            if links_text_len < text_len * 0.5:
                                max_len = text_len
                                largest_container = tag
                    except Exception as e:
                        logger.debug(f"Error processing tag {tag.name}: {e}")

                if largest_container:
                    main_content_element = largest_container
                    logger.info(
                        f"Found main content using largest text container (length: {max_len})"
                    )

            # Strategy 3: Extract paragraphs with substantial content
            if not main_content_element:
                paragraphs = []
                for p in soup.find_all("p"):
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Only paragraphs with substantial content
                        paragraphs.append(text)

                if paragraphs:
                    logger.info(
                        f"Extracted {len(paragraphs)} paragraphs with substantial content"
                    )
                    return "\n\n".join(paragraphs)[: self.max_observe]

            # Strategy 4: Fallback to body or entire document
            if not main_content_element:
                main_content_element = soup.body
                if main_content_element:
                    logger.info("Using <body> as main content container.")
                else:
                    main_content_element = soup
                    logger.warning("Could not find <body>, using entire soup.")

            # Extract and clean the text from the identified content element
            if main_content_element:
                # Get text with newlines preserved for paragraph structure
                text = main_content_element.get_text(separator="\n", strip=True)

                # Clean up the text
                text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize multiple newlines
                text = re.sub(r"\s{2,}", " ", text)  # Normalize multiple spaces
                text = re.sub(
                    r"(\n\s*){3,}", "\n\n", text
                )  # Limit consecutive newlines

                # Remove common noise patterns
                noise_patterns = [
                    r"Cookie Policy",
                    r"Privacy Policy",
                    r"Terms of Service",
                    r"All Rights Reserved",
                    r"Copyright \d{4}",
                    r"Accept Cookies",
                    r"newsletter signup",
                    r"sign up for our newsletter",
                ]
                for pattern in noise_patterns:
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

                logger.info(f"Extracted text length: {len(text)}")
                return text[: self.max_observe]  # Truncate before returning
            else:
                logger.warning("Failed to find any suitable content container.")
                return ""

        except Exception as e:
            logger.error(f"Error in _extract_main_content: {e}", exc_info=True)
            # Fallback to basic regex-based extraction
            try:
                text = re.sub(
                    r"<script.*?</script>|<style.*?</style>",
                    "",
                    html,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[: self.max_observe]
            except Exception as e2:
                logger.error(f"Even fallback extraction failed: {e2}")
                return "Failed to extract content from this page."

    async def _analyze_combined_content(
        self, combined_content: str, query: str, sources: List[Dict]
    ) -> str:
        """Analyze combined content using Ollama API, with saving results."""
        logger.info(
            f"Analyzing combined content (length: {len(combined_content)}) for query: '{query}'"
        )
        if not combined_content:
            return "No content was available for analysis."

        try:
            cfg = config

            # Get LLM settings from config
            llm_settings = None
            if hasattr(cfg, "llm") and cfg.llm:
                # If config.llm is a dictionary of LLM settings
                if isinstance(cfg.llm, dict):
                    # Try to get the default LLM settings
                    if "default" in cfg.llm:
                        llm_settings = cfg.llm["default"]
                    else:
                        # Just use the first LLM settings we find
                        for key, value in cfg.llm.items():
                            if isinstance(value, dict) and "api_type" in value:
                                llm_settings = value
                                break
                else:
                    # config.llm is already the LLM settings object
                    llm_settings = cfg.llm

            # Check if we have valid LLM settings and if it's Ollama
            api_type = None
            api_base = None
            model_name = None
            temperature = 0.0
            max_tokens = 4096

            if isinstance(llm_settings, dict):
                api_type = llm_settings.get("api_type", "").lower()
                api_base = llm_settings.get("base_url")
                model_name = llm_settings.get("model")
                temperature = llm_settings.get("temperature", 0.0)
                max_tokens = llm_settings.get("max_tokens", 4096)
            else:
                api_type = getattr(llm_settings, "api_type", "").lower()
                api_base = getattr(llm_settings, "base_url", None)
                model_name = getattr(llm_settings, "model", None)
                temperature = getattr(llm_settings, "temperature", 0.0)
                max_tokens = getattr(llm_settings, "max_tokens", 4096)

            # Verify it's Ollama with all required settings
            if api_type != "ollama" or not api_base or not model_name:
                logger.error(
                    f"Ollama configuration missing or incomplete: api_type={api_type}, base_url={api_base}, model={model_name}"
                )
                return "Error: Ollama analysis configuration is missing or incomplete."

            # Adjust the API base URL if it contains '/v1' (OpenAI-style versioning)
            if "/v1" in api_base:
                api_base = api_base.replace("/v1", "")
                logger.info(f"Adjusted API base URL for Ollama: {api_base}")

            api_base = api_base.rstrip("/")
            ollama_url = f"{api_base}/api/generate"
            logger.info(
                f"Using Ollama: URL={ollama_url}, Model={model_name}, Temp={temperature}, MaxTokens={max_tokens}"
            )

            max_input_len = 32000  # Safety limit
            if len(combined_content) > max_input_len:
                logger.warning(
                    f"Combined content ({len(combined_content)} chars) exceeds max input guess ({max_input_len}). Truncating."
                )
                combined_content = (
                    combined_content[:max_input_len] + "\n\n[Input Content Truncated]"
                )

            prompt = f"""You are a research analyst. Analyze the following web content collected for the query: "{query}"

--- Collected Content (Sources are marked as '--- SOURCE [Number]: [URL] ---') ---
{combined_content}
--- End Collected Content ---

Based *only* on the text provided above:
1. Answer the query: "{query}"
2. Summarize the key information relevant to the query from all sources.
3. Identify any significant conflicting information or differing perspectives.
4. Highlight the most relevant facts and insights.
5. Cite sources using [Source N] format where N matches the number in the content.

Provide a concise, well-structured response. If the content doesn't answer the query, state that. Do not add external knowledge.
"""

            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }

            logger.info("Sending request to Ollama API for analysis...")
            start_time = time.time()
            response = requests.post(
                ollama_url, json=payload, timeout=180
            )  # Increased timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            end_time = time.time()
            logger.info(
                f"Ollama API request completed in {end_time - start_time:.2f} seconds."
            )

            result = response.json()
            analysis = result.get("response", "").strip()

            if analysis:
                logger.info(
                    f"Successfully received analysis from Ollama API ({len(analysis)} chars)"
                )
                # Save Research Results
                try:
                    results_dir = Path(config.workspace_root) / "research_results"
                    results_dir.mkdir(parents=True, exist_ok=True)
                    query_slug = _slugify(query)[:50]
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"research_{query_slug}_{ts}.json"
                    filepath = results_dir / filename
                    research_data = {
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "sources": sources,
                        "analysis": analysis,
                        "raw_combined_content_length": len(combined_content),
                    }
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved research results to {filepath}")
                except Exception as save_err:
                    logger.error(f"Failed to save research results: {save_err}")
                return analysis
            else:
                logger.error(
                    f"Ollama API returned success but response field is empty. Response: {result}"
                )
                return self._fallback_content_analysis(combined_content, query, sources)

        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to Ollama API ({ollama_url})")
            return f"Error: Analysis service timed out. {str(req_err)}"
        except requests.exceptions.RequestException as req_err:
            logger.error(
                f"Network error connecting to Ollama API ({ollama_url}): {req_err}",
                exc_info=True,
            )
            return f"Error: Could not connect to the analysis service. {str(req_err)}"
        except Exception as e:
            logger.error(
                f"Error analyzing combined content with Ollama API: {e}", exc_info=True
            )
            return self._fallback_content_analysis(combined_content, query, sources)

    def _fallback_content_analysis(
        self, combined_content: str, query: str, sources: List[Dict]
    ) -> str:
        """Fallback method if Ollama analysis fails. Provides a structured summary."""
        logger.warning("Executing fallback content analysis.")
        if not combined_content:
            return "Analysis failed: No content was provided."

        output = f"Fallback Analysis for query: '{query}'\n\n"
        output += "Could not perform detailed analysis via LLM. Here's a summary of collected content:\n\n"

        # Split content by source markers more robustly
        source_blocks = re.split(r"--- SOURCE \d+:.*?---", combined_content)
        source_blocks = [
            block.strip() for block in source_blocks if block and block.strip()
        ]  # Clean up empty splits

        if not source_blocks or len(source_blocks) != len(sources):
            logger.warning(
                f"Mismatch/issue splitting content blocks ({len(source_blocks)}) vs sources ({len(sources)}). Summarizing raw content."
            )
            output += f"Raw combined content (first {self.max_observe} chars):\n{combined_content[:self.max_observe]}...\n"
            return output

        for i, source_info in enumerate(sources):
            url = source_info.get("url", f"Unknown Source {i+1}")
            output += f"--- Source {i+1}: {url} ---\n"
            content_snippet = source_blocks[i]
            # Simple summary: first few lines or sentences
            lines = content_snippet.splitlines()
            summary = "\n".join(
                line for line in lines[:5] if line.strip()
            )  # First 5 non-empty lines
            if not summary:  # If very short lines, try sentences
                sentences = re.split(r"(?<=[.!?])\s+", content_snippet)
                summary = " ".join(sentences[:2])  # First 2 sentences
            output += f"Summary Snippet:\n{summary[:500]}...\n\n"

        output += f"\nNote: This is a basic summary due to an analysis error. Full content length: {len(combined_content)}"
        return output

    async def _handle_website_overlays(self):
        """Handle cookie popups, paywalls, and other common website overlays."""
        logger.info("🛡️ Attempting to handle website overlays...")
        # No need to get the browser_tool directly, we'll use execute_browser_use

        # Use a list of scripts/actions to try sequentially
        actions_to_try = [
            # 1. Try known JS library handlers
            {
                "type": "evaluate",
                "script": """
            function handleCommonConsentLibs() { /* ... (full script from previous answer) ... */
                let handled = false;
                // OneTrust
                if (typeof OneTrust !== 'undefined') {
                    if (OneTrust.RejectAll) { try { OneTrust.RejectAll(); handled = true; console.log('OneTrust RejectAll executed.'); } catch(e){} }
                    else if (OneTrust.AllowAll) { try { OneTrust.AllowAll(); handled = true; console.log('OneTrust AllowAll executed.'); } catch(e){} }
                }
                // Cookiebot, CookieConsent, Osano, Quantcast, Didomi, Civic etc.
                 if (typeof Cookiebot !== 'undefined' && Cookiebot.submitCustomConsent) { try { Cookiebot.submitCustomConsent(true, true, true); handled = true; console.log('Cookiebot executed.'); } catch(e){} }
                 if (typeof CookieConsent !== 'undefined' && CookieConsent.acceptAll) { try { CookieConsent.acceptAll(); handled = true; console.log('CookieConsent executed.'); } catch(e) {} }
                 if (typeof Osano !== 'undefined' && Osano.cm && Osano.cm.acceptAll) { try { Osano.cm.acceptAll(); handled = true; console.log('Osano executed.'); } catch(e){} }
                 if (typeof __tcfapi !== 'undefined') { try { __tcfapi('acceptAll', 2, () => {}); handled = true; console.log('Quantcast executed.'); } catch(e){} }
                 if (typeof Didomi !== 'undefined' && Didomi.setUserAgreeToAll) { try { Didomi.setUserAgreeToAll(); handled = true; console.log('Didomi executed.'); } catch(e) {} }
                 if (typeof CookieControl !== 'undefined' && CookieControl.acceptAll) { try { CookieControl.acceptAll(); handled = true; console.log('CookieControl executed.'); } catch(e) {} }
                return handled;
             } handleCommonConsentLibs();
            """,
            },
            # 2. Click common buttons
            {
                "type": "evaluate",
                "script": """
            function clickCommonButtons() { /* ... (full script from previous answer) ... */
                const buttons = [];
                const selectors = [ '#onetrust-accept-btn-handler', '.cc-dismiss', '.cc-btn.cc-allow', '.cc-allow', /* ... more selectors ... */ ];
                selectors.forEach(s => { try { buttons.push(...document.querySelectorAll(s)); } catch(e) {} });
                const commonTexts = [ 'Accept', 'Accept All', 'Allow All', 'Agree', /* ... more texts ... */ 'Reject', 'Close', 'Dismiss' ];
                document.querySelectorAll('button, a, [role="button"]').forEach(el => { /* ... add if text matches ... */ });
                 let clicked = false; /* ... (logic to prioritize reject/close, then accept/agree) ... */
                 // Prioritize reject/close/dismiss if found and visible
                 for (const btn of new Set(buttons)) { /* ... (visibility check) ... */ if (['reject', 'deny', 'decline', 'close', 'dismiss'].some(t => btn.textContent.trim().toLowerCase().includes(t))) { btn.click(); clicked = true; break; } }
                 // If no reject/close was clicked, try accept/agree
                 if (!clicked) { for (const btn of new Set(buttons)) { /* ... (visibility check) ... */ if (['accept', 'allow', 'agree', 'ok'].some(t => btn.textContent.trim().toLowerCase().includes(t))) { btn.click(); clicked = true; break; } } }
                return clicked;
             } clickCommonButtons();
            """,
            },
            # 3. Press Escape Key
            {"type": "press", "args": {"selector": "body", "key": "Escape"}},
            # 4. Remove common overlay elements and fix scrolling
            {
                "type": "evaluate",
                "script": """
            function removeOverlaysAndFixScroll() { /* ... (full script from previous answer) ... */
                 let removed = false; const overlaySelectors = [ '.modal', '#modal', /* ... more selectors ... */ ];
                 overlaySelectors.forEach(s => { try { document.querySelectorAll(s).forEach(el => { el.remove(); removed = true; }); } catch(e){} });
                 // Fix body scrolling
                 if (document.body.style.overflow === 'hidden' || document.body.classList.contains('modal-open')) { document.body.style.overflow = 'auto'; document.body.style.position = 'static'; document.body.classList.remove('modal-open', 'no-scroll'); removed = true; }
                 // Fix html scrolling
                 if (document.documentElement.style.overflow === 'hidden') { document.documentElement.style.overflow = 'auto'; removed = true; }
                 return removed;
             } removeOverlaysAndFixScroll();
            """,
            },
        ]

        for i, action_def in enumerate(actions_to_try):
            try:
                action_type = action_def["type"]
                logger.debug(f"Overlay Action {i+1}: Type='{action_type}'")
                if action_type == "evaluate":
                    # For JavaScript evaluation, we need to use a different approach
                    # The browser_use tool doesn't support direct script evaluation
                    # Let's use a press action on common overlay elements instead
                    try:
                        # Try pressing Escape key first
                        await self.execute_browser_use(
                            "press", selector="body", key="Escape"
                        )
                        # Then try clicking on common overlay dismiss buttons
                        await self.execute_browser_use(
                            "extract_content",
                            goal="Find and dismiss any cookie consent popups or overlays",
                        )
                        result = True
                    except Exception as e:
                        logger.debug(f"Error handling overlay: {e}")
                        result = False
                    logger.debug(f" -> Result: {result}")
                    if result and str(result).lower() == "true":
                        time.sleep(0.7)  # Pause if script reported change
                elif action_type == "press":
                    await self.execute_browser_use("press", **action_def["args"])
                    logger.debug(f" -> Executed key press.")
                    time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error during overlay action {i+1} ({action_type}): {e}")

        logger.info("🔑 Finished overlay handling attempts.")

        # Return True to indicate overlay handling is complete
        return True

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from the given text using robust regex patterns."""
        if not text:
            return []

        # Standard http/https URLs pattern
        url_pattern = r"https?://[^\s()<>]+"

        # Also look for www. URLs without http/https
        www_pattern = r"\b(www\.[^\s()<>]+\.[a-zA-Z]{2,})"

        # Combine results from both patterns
        urls = re.findall(url_pattern, text)
        www_urls = re.findall(www_pattern, text)

        # Process www URLs to add https:// prefix
        for www_url in www_urls:
            if www_url.startswith("www."):
                urls.append("https://" + www_url)

        # Basic validation and deduplication while preserving order
        seen = set()
        valid_urls = []
        for url in urls:
            try:
                # Decode URL-encoded characters
                url = unquote(url)
                parsed = urlparse(url)
                # Ensure it has a domain part with at least one dot
                if "." in parsed.netloc and url not in seen:
                    valid_urls.append(url)
                    seen.add(url)
            except Exception as e:
                logger.debug(f"Error processing URL {url}: {e}")

        return valid_urls

    def _should_use_browser(self, user_message: str) -> bool:
        """Determine if browser should be used based on user message content."""
        urls = self._extract_urls(user_message)
        if urls:
            self.mentioned_urls.update(urls)
            logger.info(f"Detected URLs in message: {urls}. Suggesting browser use.")
            return True

        browse_phrases = [
            "check website",
            "visit site",
            "go to page",
            "look at page",
            "on their website",
            "online at",
            "url for",
        ]
        research_keywords = [
            "find info on",
            "research",
            "what is",
            "tell me about",
            "latest news on",
            "details about",
        ]

        msg_lower = user_message.lower()
        if any(phrase in msg_lower for phrase in browse_phrases):
            logger.info("Detected browse phrase. Suggesting browser use.")
            return True

        # If it looks like a research question requiring external info
        if (
            msg_lower.endswith("?")
            and len(user_message) > 15
            and any(kw in msg_lower for kw in research_keywords)
        ):
            logger.info(
                "Detected potential research question. May involve web research."
            )
            # Let LLM decide initially, suggestion logic will catch it later if needed
            return False

        return False

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings.

        This is a simple implementation using word overlap.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize strings
        str1 = str1.lower().strip()
        str2 = str2.lower().strip()

        # Simple word overlap similarity
        words1 = set(str1.split())
        words2 = set(str2.split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return overlap / union if union > 0 else 0.0

    def _is_redundant_question(self, question: str) -> bool:
        """Check if a question is redundant based on conversation history."""
        if not self._prevent_repeated_questions:
            return False

        question_norm = question.strip().lower()

        # First check for exact matches (highest priority)
        for prev_question in self.tool_tracker.asked_questions:
            prev_norm = prev_question.strip().lower()
            if question_norm == prev_norm:
                logger.warning(f"🔄 Avoiding exactly repeated question: {question}")
                return True

        # Check for semantic similarity with recent questions
        # Only check the last 10 questions to avoid false positives with older questions
        recent_questions = (
            self.tool_tracker.asked_questions[-10:]
            if len(self.tool_tracker.asked_questions) > 10
            else self.tool_tracker.asked_questions
        )

        for prev_question in recent_questions:
            # Calculate similarity
            similarity = self._calculate_similarity(question, prev_question)

            # If very similar (80%+ overlap), consider it redundant
            if similarity > 0.8:
                logger.warning(
                    f"🔄 Avoiding similar question (similarity: {similarity:.2f}): {question}"
                )
                return True

            # For moderately similar questions (60-80% overlap), check if they have the same intent
            if similarity > 0.6:
                # Extract question words (what, who, when, where, why, how)
                q_words_pattern = r"\b(what|who|when|where|why|how)\b"
                import re

                q_words_question = set(re.findall(q_words_pattern, question_norm))
                q_words_prev = set(re.findall(q_words_pattern, prev_norm))

                # If they have the same question words, they likely have the same intent
                if (
                    q_words_question
                    and q_words_prev
                    and q_words_question == q_words_prev
                ):
                    logger.warning(f"🔄 Avoiding question with same intent: {question}")
                    return True

        # If we get here, the question is not redundant
        return False

    def _get_most_relevant_url(self, query: str) -> Optional[str]:
        """Get the most relevant URL from mentioned URLs based on query keywords."""
        if not self.mentioned_urls:
            return None
        if len(self.mentioned_urls) == 1:
            return next(iter(self.mentioned_urls))

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return list(self.mentioned_urls)[-1]  # Return last if no query terms

        best_url = None
        max_score = -1
        for url in self.mentioned_urls:
            score = 0
            parsed_url = urlparse(url)
            url_text = (
                parsed_url.netloc
                + parsed_url.path
                + parsed_url.query
                + parsed_url.fragment
            ).lower()
            url_terms = set(re.findall(r"\w+", url_text))
            score += len(query_terms.intersection(url_terms))
            # Boost score if term is in domain or last path segment
            if any(term in parsed_url.netloc.lower() for term in query_terms):
                score += 2
            if parsed_url.path:
                last_segment = parsed_url.path.split("/")[-1]
                if any(term in last_segment for term in query_terms):
                    score += 1

            if score > max_score:
                max_score = score
                best_url = url

        return (
            best_url if best_url and max_score >= 0 else list(self.mentioned_urls)[-1]
        )

    async def _handle_str_replace_editor(self, command: ToolCall) -> str:
        """Special handler for str_replace_editor to create files and handle content writing."""
        args_str = command.function.arguments or "{}"
        try:
            args = json.loads(args_str)
            path_str = args.get("path", "")
            editor_command = args.get("command", "").lower()
            file_text = args.get("file_text", "")

            if not path_str:
                # Generate a default path if none provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path_str = f"workspace/document_{timestamp}.txt"
                args["path"] = path_str
                command.function.arguments = json.dumps(args)

            target_path = Path(path_str)
            if not target_path.is_absolute():
                target_path = Path(config.workspace_root) / path_str

            # Create parent directories if they don't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Check for creative content generation before file operations
            is_creative_content = (
                editor_command == "create"
                and file_text
                and self._is_creative_content_description(file_text)
            )
            if is_creative_content:
                logger.info(f"Detected creative content request: {file_text[:100]}...")
                generated_content = await self._generate_creative_content(file_text)
                if generated_content:
                    # Update the file_text with generated content
                    file_text = generated_content
                    args["file_text"] = file_text
                    command.function.arguments = json.dumps(args)
                    logger.info(f"Successfully generated creative content")

                    # Update task completer if available
                    if self.task_completer and not self.task_completed:
                        self.task_completer.add_information("content_generated", True)
                        self.task_completer.add_information(
                            "generated_content", file_text
                        )
                        logger.info("Updated task completer with generated content")

            # Handle different editor commands
            if editor_command == "view":
                if not target_path.exists():
                    try:
                        target_path.touch()
                        logger.info(
                            f"📄 Created missing file for viewing: {target_path}"
                        )
                        args_with_message = args.copy()
                        args_with_message["message"] = "Created missing file"
                        self.tool_tracker.record_tool_usage(
                            "str_replace_editor", args_with_message, "success"
                        )
                        return f"Observed output of cmd `str_replace_editor`: File '{path_str}' did not exist and was created empty."
                    except Exception as e:
                        logger.error(
                            f"Failed to create missing file {target_path}: {e}"
                        )
                        return f"Error executing str_replace_editor: Failed to create file '{path_str}'. {str(e)}"
                else:
                    try:
                        with open(target_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        return f"File content:\n{content}"
                    except Exception as e:
                        return f"Error reading file: {str(e)}"

            # Handle create/write commands
            elif editor_command in ["create", "write"]:
                try:
                    mode = (
                        "w"
                        if editor_command == "create" or not target_path.exists()
                        else "a"
                    )
                    with open(target_path, mode, encoding="utf-8") as f:
                        f.write(file_text)

                    action = "Created" if mode == "w" else "Updated"
                    logger.info(f"✅ {action} file: {target_path}")

                    # If this was a creative content task, mark it as complete
                    if (
                        is_creative_content
                        and self.task_completer
                        and not self.task_completed
                    ):
                        self.task_completed = True
                        logger.info(
                            f"Marked task as complete after saving creative content to {target_path}"
                        )

                    # Record successful tool usage
                    args_with_message = args.copy()
                    args_with_message["message"] = f"{action} file successfully"
                    self.tool_tracker.record_tool_usage(
                        "str_replace_editor", args_with_message, "success"
                    )

                    # Return success message with file path and preview
                    preview = file_text[:200] + ("..." if len(file_text) > 200 else "")
                    result_msg = (
                        f"Observed output of cmd `str_replace_editor` executed:\n"
                        f"{action} file successfully at: {path_str}\n\n"
                        f"Preview:\n{preview}"
                    )

                    # Add completion message for creative content
                    if is_creative_content:
                        result_msg += (
                            "\n\n✅ Creative content generated and saved successfully!"
                        )

                    return result_msg

                except Exception as e:
                    error_msg = f"Failed to write to file {target_path}: {str(e)}"
                    logger.error(error_msg, exc_info=True)

                    # Update task completer with error if this was a creative content task
                    if is_creative_content and self.task_completer:
                        self.task_completer.add_information(
                            "content_generation_error", error_msg
                        )
                        logger.error(
                            "Recorded content generation error in task completer"
                        )

                    # Record tool usage with error
                    args_with_message = args.copy()
                    args_with_message["message"] = f"Failed to write to file: {str(e)}"
                    self.tool_tracker.record_tool_usage(
                        "str_replace_editor", args_with_message, "failure"
                    )

                    # Return detailed error message
                    return (
                        f"Error executing str_replace_editor: Failed to write to file '{path_str}'.\n"
                        f"Error: {str(e)}"
                    )

            # Check if this is a creative content task that needs generation
            # (This is a fallback in case the first check was missed)
            if (
                not is_creative_content
                and editor_command == "create"
                and file_text
                and len(file_text) < 100
                and self._is_creative_content_description(file_text)
            ):
                logger.info(
                    f"Fallback detected creative content request: {file_text[:100]}..."
                )
                generated_content = await self._generate_creative_content(file_text)
                if generated_content:
                    # Update the file_text with generated content
                    file_text = generated_content
                    args["file_text"] = file_text
                    command.function.arguments = json.dumps(args)
                    logger.info("Successfully generated creative content in fallback")

                    # Update task completer if available
                    if self.task_completer and not self.task_completed:
                        self.task_completer.add_information("content_generated", True)
                        self.task_completer.add_information(
                            "generated_content", file_text
                        )
                        logger.info(
                            "Updated task completer with generated content in fallback"
                        )

            # Execute original command via base class
            result_str = await super().execute_tool(command)
            # Tool tracker recording handled in the main execute_tool method based on result_str

            return result_str

        except json.JSONDecodeError:
            error_msg = (
                f"Error executing str_replace_editor: Invalid JSON args: {args_str}"
            )
            logger.error(error_msg)
            self.tool_tracker.record_tool_usage(
                "str_replace_editor", {"arguments": args_str}, "failure", "Invalid JSON"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            logger.error(
                f"Unexpected error in _handle_str_replace_editor: {e}", exc_info=True
            )
            arg_info = {"arguments": args_str}
            self.tool_tracker.record_tool_usage(
                "str_replace_editor",
                arg_info,
                "failure",
                f"Outer handler error: {str(e)}",
            )
            return f"Error handling str_replace_editor: Unexpected error. {str(e)}"

    def _is_creative_content_description(self, text: str) -> bool:
        """Determine if the text is a description of creative content that needs to be generated."""
        if not text or len(text.strip()) < 10:  # Skip very short texts
            return False

        text_lower = text.lower().strip()

        # Check for common patterns that indicate a creative content description
        creative_indicators = [
            # Essay patterns
            "write a",
            "create a",
            "compose a",
            "draft a",
            "generate a",
            "write an",
            "create an",
            "compose an",
            "draft an",
            "generate an",
            "essay about",
            "essay on",
            "long-form essay",
            "academic paper",
            "research paper",
            "article about",
            "blog post",
            "write about",
            # Other creative content types
            "poem about",
            "in the style of",
            "story about",
            "song about",
            "script for",
            "letter to",
            "speech about",
            "report on",
            "review of",
            "analysis of",
            "summary of",
        ]

        # Check for content type indicators
        content_types = [
            "essay",
            "article",
            "story",
            "poem",
            "song",
            "script",
            "letter",
            "speech",
            "report",
            "review",
            "analysis",
            "summary",
            "blog post",
            "paper",
            "thesis",
            "dissertation",
        ]

        # Check for task verbs
        task_verbs = [
            "write",
            "create",
            "compose",
            "draft",
            "generate",
            "develop",
            "produce",
            "prepare",
            "construct",
            "formulate",
        ]

        # Check for length indicators
        length_indicators = [
            "long-form",
            "detailed",
            "comprehensive",
            "in-depth",
            "thorough",
            "extensive",
            "lengthy",
            "brief",
        ]

        # Check for any creative indicators
        has_creative_indicator = any(
            indicator in text_lower for indicator in creative_indicators
        )

        # Check for content type + task verb pattern
        has_content_task_pattern = any(
            (f"{verb} {content}" in text_lower or f"{verb} a {content}" in text_lower)
            for verb in task_verbs
            for content in content_types
        )

        # Check for length indicators
        has_length_indicator = any(
            indicator in text_lower for indicator in length_indicators
        )

        # Check for question-like patterns
        is_question = text_lower.endswith("?")

        # Consider it creative content if:
        # 1. It has explicit creative indicators, OR
        # 2. It has a content type + task verb pattern, AND
        # 3. It's not a question (questions are better handled by research)
        # Keyword override
        if (
            hasattr(self, "_explicit_task_type")
            and self._explicit_task_type == "research"
        ):
            return False

    async def _research_subject(self, subject):
        """Research a subject using search tools and return relevant background information."""
        if not subject or len(subject.strip()) < 3:
            return ""

        logger.info(f"Researching background information for: {subject}")
        try:
            # Try to use search_web tool if available
            search_tool = self.available_tools.get_tool("search_web")
            if search_tool:
                logger.info(f"Using search_web tool to research: {subject}")
                search_args = {"query": subject}
                search_result = await search_tool.execute(search_args)
                if search_result and not hasattr(search_result, "error"):
                    search_data = (
                        search_result.output
                        if hasattr(search_result, "output")
                        else str(search_result)
                    )
                    if search_data and len(search_data) > 10:
                        logger.info(f"Found background information for {subject}")
                        return search_data

            # Use browser search as fallback
            if self.browser_context_helper:
                logger.info(f"Using browser to research: {subject}")
                try:
                    await self.browser_context_helper.search(subject)
                    page_content = await self.browser_context_helper.get_page_content()
                    if page_content and len(page_content) > 20:
                        logger.info(f"Found browser-based information for {subject}")
                        return page_content[:2000]  # Limit length to avoid token issues
                except Exception as browser_error:
                    logger.error(f"Browser research error: {browser_error}")

        except Exception as e:
            logger.error(f"Error researching subject {subject}: {e}")

        return ""

    def _create_fallback_content(self, content_type, description):
        """Create fallback content when generation fails."""
        logger.warning(f"Creating fallback content for {content_type}")
        return f"# {content_type.title()} about {description}\n\n[Unable to generate content automatically]"

    async def _generate_creative_content(
        self, description: str, target_filename: str = None
    ) -> str:
        """Generate creative content based on the description. If 'save to' or 'save as' is present, save the result to the specified file."""
        try:
            logger.info(f"Generating creative content for: {description}")
            content_type = "essay"  # Default type

            # Extract filename if present (save to/save as)
            filename = None
            filename_match = re.search(
                r'save (?:to|as)\s*["\']([^"\']+)\s*["\']', description, re.IGNORECASE
            )
            if not filename_match:
                filename_match = re.search(
                    r"save (?:to|as)\s+([\w\-.\/\\]+)", description, re.IGNORECASE
                )
            if filename_match:
                filename = filename_match.group(1).strip("\"'")

            # Extract clean description (remove save to file parts)
            clean_description = description
            if filename:
                clean_description = re.sub(
                    r'save (?:to|as)\s*["\']?[^"\'\.]+["\']?',
                    "",
                    clean_description,
                    flags=re.IGNORECASE,
                )

            # Determine content type
            desc_lower = clean_description.lower()
            content_types = [
                "poem",
                "essay",
                "story",
                "novel",
                "short story",
                "blog post",
                "article",
                "song",
                "lyrics",
                "script",
                "screenplay",
                "play",
                "letter",
                "speech",
                "report",
                "review",
                "analysis",
                "summary",
                "paper",
                "thesis",
                "dissertation",
            ]

            for ctype in content_types:
                if ctype in desc_lower:
                    content_type = ctype
                    break

            # Check for style
            style = None
            style_match = re.search(
                r"in the style of (.+?)(?:\.|,|;|$)", clean_description, re.IGNORECASE
            )
            if style_match:
                style = style_match.group(1).strip()

            # Extract the main subjects to research
            subjects = []
            about_match = re.search(
                r"(?:about|on)\s+([^.,;!?]+)", clean_description, re.IGNORECASE
            )
            if about_match:
                subjects.append(about_match.group(1).strip())
            else:
                # Try to extract a subject from the description
                words = clean_description.split()
                if len(words) > 3 and content_type in clean_description:
                    # Extract words after the content type
                    content_idx = clean_description.lower().find(content_type)
                    if content_idx != -1:
                        remaining = clean_description[
                            content_idx + len(content_type) :
                        ].strip()
                        # Try to get a reasonable subject
                        if remaining:
                            subjects.append(remaining.split(".")[0].strip())

            # Default creative temperature
            temperature = 0.7

            # Research background information for subjects
            background_info = ""
            for subject in subjects:
                if not subject:
                    continue

                logger.info(f"Researching background information for: {subject}")
                try:
                    # Use web search to get contextual information
                    search_tool = self.available_tools.get_tool("search_web")
                    if search_tool:
                        logger.info(f"Using search_web tool to research: {subject}")
                        search_args = {"query": subject}
                        search_result = await search_tool.execute(search_args)
                        if search_result and not hasattr(search_result, "error"):
                            search_data = (
                                search_result.output
                                if hasattr(search_result, "output")
                                else str(search_result)
                            )
                            if search_data and len(search_data) > 10:
                                background_info += f"\n\nBackground information about {subject}:\n{search_data}"
                                logger.info(
                                    f"Found background information for {subject}"
                                )
                except Exception as e:
                    logger.error(f"Error searching for information: {e}")

            # Create prompt based on content type
            prompt = f"I need you to create a {content_type} based on the following description:\n\n{clean_description}\n\nGuidelines:\n1. Focus on the main topic: {clean_description}\n2. Create a {content_type} that is engaging and well-structured"

            # Add style guidelines if provided
            if style:
                prompt += f"\n3. Write in the style of {style}"

            # Add content-specific guidelines
            if content_type in ["poem", "song", "lyrics"]:
                prompt += "\n4. Use poetic devices like metaphor, simile, and imagery\n5. Pay attention to rhythm and flow"
                temperature = 0.8  # Higher temperature for poetry
            elif content_type in [
                "essay",
                "article",
                "report",
                "analysis",
                "paper",
                "thesis",
                "dissertation",
            ]:
                prompt += "\n4. Include a clear introduction, body, and conclusion\n5. Use evidence and logical reasoning to support arguments"
                temperature = 0.4  # Lower temperature for formal writing
            elif content_type in ["story", "novel", "short story"]:
                prompt += "\n4. Include a clear beginning, middle, and end\n5. Develop characters and setting"
                temperature = 0.7  # Balanced temperature for stories
            elif content_type in ["script", "screenplay", "play"]:
                prompt += "\n4. Use dialogue and stage directions\n5. Clearly indicate characters and scenes"
                temperature = 0.6  # Slightly lower for structured scripts

            # Add background information if found
            if background_info:
                prompt += f"\n\n{background_info}"

            prompt += f"\n\nPlease provide the {content_type} without any additional commentary or notes."

            logger.info(f"[LLM PROMPT] Full creative content prompt:\n{prompt}")

            # Generate content using Ollama - ensure we're using the right configuration
            messages = [{"role": "user", "content": prompt}]
            logger.info(f"[LLM MESSAGES] Messages object being sent: {messages}")

            try:
                # Use the LLM to generate content, configured for Ollama
                response = await self.llm.ask(
                    messages, stream=False, temperature=temperature
                )
                content = (
                    response["content"]
                    if isinstance(response, dict) and "content" in response
                    else response
                )

                # Verify we got actual content
                if (
                    content and len(content.strip()) > 50
                ):  # Reasonable minimum length for creative content
                    logger.info(
                        f"Successfully generated {content_type} with {len(content.split())} words"
                    )
                    
                    # If target filename is provided, save content directly to that file
                    if target_filename and content:
                        try:
                            # Create full path relative to workspace
                            full_path = os.path.join(
                                config.workspace_root, target_filename
                            )
                            # Make directory if it doesn't exist
                            os.makedirs(
                                os.path.dirname(os.path.abspath(full_path)),
                                exist_ok=True,
                            )
                            # Save the content
                            with open(full_path, "w") as f:
                                f.write(content)
                            logger.info(
                                f"[DIRECT SAVE] Content saved directly to requested file: {full_path}"
                            )

                            # Set these properties for proper task result display
                            self._deliverable_path = full_path
                            self._deliverable_content = content
                            self._task_result = f"Creative content generated and saved to {target_filename}\n\n```\n{content[:500]}\n```\n{'...[content truncated]...' if len(content) > 500 else ''}"
                        except Exception as e:
                            logger.error(
                                f"Error saving content to {target_filename}: {e}"
                            )

                    return content.strip()
                else:
                    logger.warning(
                        f"Generated content too short ({len(content.split()) if content else 0} words), retrying with different approach"
                    )

                    # Try again with a simpler prompt
                    simplified_prompt = f"Write a {content_type} about {clean_description}. Make it detailed and use {style if style else 'formal'} style."
                    simplified_messages = [
                        {"role": "user", "content": simplified_prompt}
                    ]
                    logger.info(f"Attempting simplified prompt: {simplified_prompt}")

                    response = await self.llm.ask(
                        simplified_messages, stream=False, temperature=temperature
                    )
                    content = (
                        response["content"]
                        if isinstance(response, dict) and "content" in response
                        else response
                    )

                    # If target filename is provided, save content directly to that file
                    if target_filename and content:
                        try:
                            # Create full path relative to workspace
                            full_path = os.path.join(
                                config.workspace_root, target_filename
                            )
                            # Make directory if it doesn't exist
                            os.makedirs(
                                os.path.dirname(os.path.abspath(full_path)),
                                exist_ok=True,
                            )
                            # Save the content
                            with open(full_path, "w") as f:
                                f.write(content)
                            logger.info(
                                f"[DIRECT SAVE] Content saved directly to requested file: {full_path}"
                            )

                            # Set these properties for proper task result display
                            self._deliverable_path = full_path
                            self._deliverable_content = content
                            self._task_result = f"Creative content generated and saved to {target_filename}\n\n```\n{content[:500]}\n```\n{'...[content truncated]...' if len(content) > 500 else ''}"
                        except Exception as e:
                            logger.error(
                                f"Error saving content to {target_filename}: {e}"
                            )

                    return content.strip()
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}", exc_info=True)
                return self._create_fallback_content(content_type, clean_description)

        except Exception as e:
            logger.error(f"Error in _generate_creative_content: {e}", exc_info=True)
            return f"Error: Unable to generate content due to an error: {str(e)}"

    async def _handle_python_execute(self, command: ToolCall) -> str:
        """Special handler for python_execute: adds error handling wrapper."""
        args = {}
        args_str = command.function.arguments or "{}"
        try:
            args = json.loads(args_str)
            code = args.get("code", "")
            if not code:
                return "Error executing python_execute: Missing 'code' parameter"

            logger.info(f"Preparing python_execute code:\n```python\n{code}\n```")

            # --- Add robust error handling wrapper ---
            if not code.strip().startswith(
                ("try:", "import traceback", "async def")
            ):  # Avoid double-wrapping
                indented_code = "\n".join(
                    ["    " + line for line in code.strip().splitlines()]
                )
                # Use import traceback inside the executed code for better context
                wrapped_code = f"""
import traceback
import sys
try:
{indented_code}
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_string = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"--- Python Code Execution Error ---")
    print(f"Error Type: {{type(e).__name__}}")
    print(f"Error Message: {{str(e)}}")
    print(f"Traceback:\\n{{tb_string}}")
    print(f"--- End Error ---", file=sys.stderr) # Print error to stderr
"""
                args["code"] = wrapped_code
                logger.info("Applied error handling wrapper to Python code.")
                command.function.arguments = json.dumps(
                    args
                )  # Update the command object

            # --- Execute via Base Class ---
            logger.info(f"Executing python_execute with final code...")
            result_str = await super().execute_tool(command)
            # Tool tracker recording handled in the main execute_tool method

            return result_str

        except json.JSONDecodeError:
            error_msg = f"Error executing python_execute: Invalid JSON args: {args_str}"
            logger.error(error_msg)
            self.tool_tracker.record_tool_usage(
                "python_execute", {"arguments": args_str}, "failure", "Invalid JSON"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            logger.error(
                f"Unexpected error in _handle_python_execute: {e}", exc_info=True
            )
            arg_info = {"arguments": args_str}
            self.tool_tracker.record_tool_usage(
                "python_execute", arg_info, "failure", f"Outer handler error: {str(e)}"
            )
            return f"Error handling python_execute: Unexpected error. {str(e)}"

    async def _suggest_better_tool(
        self, name: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Suggests a more appropriate tool if ask_human is used for solvable tasks."""
        if name != "ask_human":
            return None
        question = args.get("inquire", "")
        q_lower = question.lower()
        if not question:
            return None
        logger.info(f"Analyzing question for better tool: '{question}'")

        # 1. System/Environment -> python_execute
        system_keywords = [
            "current directory",
            "working directory",
            "pwd",
            "list files",
            "environment variable",
            "system info",
            "python version",
        ]
        if any(kw in q_lower for kw in system_keywords):
            # Specific commands based on keywords
            if (
                "current directory" in q_lower
                or "working directory" in q_lower
                or q_lower == "pwd"
            ):
                code = "import os\nprint(f'{os.getcwd()}')"
            elif "list files" in q_lower:
                code = "import os\nprint(f'{os.listdir()}')"
            elif "environment variable" in q_lower:
                match = re.search(r"variable\s+(['\"]?)([A-Z_]\w*)\1", question)
                var = match.group(2) if match else "PATH"
                code = f"import os\nprint(os.environ.get('{var}', 'Not Set'))"
            else:
                code = "import platform, sys\nprint(f'OS: {platform.system()} {platform.release()}\\nPython: {sys.version}')"
            logger.info("Suggestion: Use python_execute for system question.")
            return {"tool": "python_execute", "args": {"code": code}}

        # 2. URL Interaction -> browser_use
        urls = self._extract_urls(question)
        if urls:
            logger.info(f"Suggestion: Use browser_use for URL: {urls[0]}")
            return {"tool": "browser_use", "args": {"url": urls[0]}}

        # 3. Creative Content Generation -> Generate directly using str_replace_editor
        if self._is_creative_content_description(question):
            logger.info(
                "Creative content generation task detected. Generating content with str_replace_editor."
            )

            # Check if we need to save to a specific file
            filename = None
            if "save to" in q_lower or "save it as" in q_lower:
                # Look for filename in quotes or after 'save to'/'save as'
                save_match = re.search(
                    r'save\s+(?:it\s+)?(?:to|as)[\s"\']+([^\s"\']+)', q_lower
                )
                if save_match:
                    filename = save_match.group(1).strip("\"'")

            # If no filename specified, generate one based on content type and topic
            if not filename:
                content_type = "essay"
                if "poem" in q_lower:
                    content_type = "poem"
                elif "story" in q_lower:
                    content_type = "story"

                # Extract the main topic for the filename
                topic = "creative_content"
                topic_match = re.search(
                    r"(?:about|on|regarding|re:?)\s+(.+?)(?:\?|$)",
                    question,
                    re.IGNORECASE,
                )
                if topic_match:
                    topic = topic_match.group(1).strip()

                # Create a clean filename
                filename = f"{content_type}_{_slugify(topic)}.txt"

            filepath = str(Path(config.workspace_root) / filename)

            # Generate the content
            content = await self._generate_creative_content(question)

            # If we have content, return the tool call to save it
            if content:
                return {
                    "tool": "str_replace_editor",
                    "args": {
                        "command": "create",
                        "path": filepath,
                        "file_text": content,
                    },
                }

            return None

        # 4. Research/Information -> browser_use for web research
        research_keywords = [
            # Question words and phrases
            "what is",
            "who is",
            "when was",
            "where is",
            "why is",
            "how to",
            "tell me about",
            "explain",
            "define",
            "history of",
            "how does",
            "compare",
            "pros cons",
            "find info on",
            "news on",
            "statistics for",
            "example of",
            "template for",
            "design for",
            "layout for",
            "modern",
            "best practice",
            "biography of",
            "about",
            "information on",
            "details about",
            "facts about",
            "who was",
            "what are",
            "how many",
            "how much",
            "when did",
            "where did",
            "why did",
            "how did",
            "what caused",
            "what happened",
            "what makes",
            "what are the",
            # How-to and instructional
            "how to use",
            "how to make",
            "how to create",
            "how to build",
            "how to install",
            "how to set up",
            "how to configure",
            "tutorial on",
            "guide to",
            "tips for",
            "best way to",
            "how can i",
            # Information requests
            "what do you know about",
            "can you tell me about",
            "i need information on",
            "looking for information about",
            "search for",
            "find me",
            "show me",
        ]

        # Check if this is a factual or research question
        is_research_question = (
            (
                q_lower.endswith("?")
                or any(
                    q_word in q_lower
                    for q_word in ["what", "who", "when", "where", "why", "how"]
                )
            )
            and len(question) > 5  # Reduced minimum length to catch more queries
            and any(kw in q_lower for kw in research_keywords)
        )

        # Check for statements that are clearly researchable
        is_research_statement = (
            not q_lower.endswith("?")
            and any(kw in q_lower for kw in ["find", "search", "look up", "research"])
            and any(kw in q_lower for kw in ["about", "on", "for"])
        )

        # Enhanced historical figure mentions with variations and titles
        historical_figures = {
            # Leaders and rulers
            "genghis khan": ["genghis", "chinggis khan", "great khan"],
            "napoleon": ["napoleon bonaparte", "emperor napoleon"],
            "cleopatra": ["cleopatra vii", "queen cleopatra"],
            "julius caesar": ["caesar", "julius caesar", "gaius julius caesar"],
            "alexander the great": ["alexander of macedon", "alexander iii of macedon"],
            "queen victoria": ["victoria of england", "queen victoria"],
            "winston churchill": ["sir winston churchill", "churchill"],
            "abraham lincoln": ["president lincoln", "honest abe"],
            "george washington": ["president washington", "general washington"],
            "queen elizabeth i": ["elizabeth i", "queen elizabeth the first"],
            "queen elizabeth ii": ["elizabeth ii", "queen elizabeth the second"],
            # Scientists and inventors
            "albert einstein": ["einstein", "albert einstein"],
            "marie curie": ["madame curie", "marie sklodowska curie"],
            "isaac newton": ["sir isaac newton", "newton"],
            "charles darwin": ["darwin", "charles darwin"],
            "thomas edison": ["edison", "thomas alva edison"],
            "nikola tesla": ["tesla", "nikola tesla"],
            "galileo galilei": ["galileo", "galileo galilei"],
            "stephen hawking": ["hawking", "stephen hawking"],
            # Artists and writers
            "leonardo da vinci": ["da vinci", "leonardo"],
            "william shakespeare": ["shakespeare", "the bard", "shakespeare"],
            "pablo picasso": ["picasso", "pablo picasso"],
            "vincent van gogh": ["van gogh", "vincent gogh"],
            "wolfgang amadeus mozart": ["mozart", "wolfgang mozart"],
            "ludwig van beethoven": ["beethoven", "ludwig beethoven"],
            # Philosophers and thinkers
            "socrates": ["socrates"],
            "plato": ["plato"],
            "aristotle": ["aristotle"],
            "confucius": ["kong fuzi", "confucius"],
            "buddha": ["siddhartha gautama", "gautama buddha", "the buddha"],
            # Religious figures
            "jesus christ": ["jesus", "jesus of nazareth", "jesus christ"],
            "prophet muhammad": ["muhammad", "prophet muhammad", "mohammed"],
            "mahatma gandhi": ["gandhi", "mahatma gandhi", "mohandas gandhi"],
            "martin luther": ["martin luther", "martin luther king jr"],
            "martin luther king jr": [
                "mlk",
                "dr martin luther king",
                "martin luther king",
            ],
            # Modern figures
            "nelson mandela": ["mandela", "nelson mandela"],
            "steve jobs": ["steve jobs", "steven jobs"],
            "bill gates": ["gates", "william henry gates iii", "bill gates"],
            "mark zuckerberg": ["zuckerberg", "mark zuckerberg"],
            "elon musk": ["musk", "elon musk"],
            "jeff bezos": ["bezos", "jeff bezos"],
            "warren buffett": ["buffett", "warren buffett"],
        }

        # Check for creative tasks about known figures/events
        is_creative_about_known = any(
            phrase in q_lower
            for phrase in [
                "create a",
                "write a",
                "compose a",
                "make a",
                "generate a",
                "create an",
                "write an",
                "compose an",
                "make an",
                "generate an",
            ]
        )

        # Check for direct historical figure questions
        is_historical_figure_question = False
        is_creative_writing = False

        # Check for figure mentions in the question
        mentioned_figure = None
        for figure_name, variations in historical_figures.items():
            # Check canonical name
            if figure_name in q_lower:
                mentioned_figure = figure_name
                break
            # Check variations
            for variation in variations:
                if variation in q_lower:
                    mentioned_figure = figure_name
                    break
            if mentioned_figure:
                break

        # If we found a figure, check the type of question
        if mentioned_figure:
            if any(
                q_lower.startswith(prefix)
                for prefix in ["who is", "who was", "what is"]
            ) or any(
                phrase in q_lower for phrase in ["tell me about", "information about"]
            ):
                is_historical_figure_question = True

            if any(
                phrase in q_lower
                for phrase in [
                    "write a poem about",
                    "compose a story about",
                    "create a tale about",
                    "write an epic about",
                    "create a narrative about",
                    "compose a ballad about",
                ]
            ):
                is_creative_writing = True
                is_creative_about_known = True

        if any(
            [
                is_research_question,
                is_research_statement,
                is_creative_about_known,
                is_historical_figure_question,
                is_creative_writing,
            ]
        ):
            # Extract key search terms from the question
            search_query = question.replace("?", "").strip()

            # For certain types of questions, create a more specific search query
            if mentioned_figure:
                # For known historical figures, create more targeted queries
                if "poem" in q_lower:
                    search_query = (
                        f"{mentioned_figure} key facts and achievements for a poem"
                    )
                elif "story" in q_lower or "tale" in q_lower or "narrative" in q_lower:
                    search_query = f"{mentioned_figure} life story and important events"
                elif any(
                    kw in q_lower for kw in ["biography", "about", "who is", "who was"]
                ):
                    search_query = f"{mentioned_figure} biography key facts"
                elif any(kw in q_lower for kw in ["history", "historical"]):
                    search_query = (
                        f"{mentioned_figure} historical significance and timeline"
                    )
                elif any(kw in q_lower for kw in ["achievements", "accomplishments"]):
                    search_query = (
                        f"{mentioned_figure} major achievements and contributions"
                    )
                else:
                    search_query = f"{mentioned_figure} key information and facts"
            else:
                # For general queries
                if any(
                    kw in q_lower for kw in ["example", "template", "design", "layout"]
                ):
                    search_query = f"examples of {search_query}"
                elif any(
                    kw in q_lower for kw in ["biography", "about", "who is", "who was"]
                ):
                    search_query = f"{search_query} biography"
                elif any(kw in q_lower for kw in ["history", "historical"]):
                    search_query = f"{search_query} history"
                elif is_creative_about_known:
                    if "poem" in q_lower:
                        search_query = f"{search_query} key facts and achievements"
                    elif "story" in q_lower:
                        search_query = f"{search_query} life story"
                    else:
                        search_query = f"{search_query} key information"

            # Clean up the query
            search_query = re.sub(
                r"^(can you|please|could you|would you|i need|i want|find|search|look up|research|information on|about|on|for)",
                "",
                search_query,
                flags=re.IGNORECASE,
            )
            search_query = re.sub(r"\s+", " ", search_query).strip()

            # For creative tasks, set up the task completer if needed
            if (
                is_creative_about_known
                or is_creative_writing
                or is_historical_figure_question
            ):
                if not self.task_completer:
                    self.task_completer = TaskCompleter()
                    self.task_completer.task_type = "creative_writing"

                # Use the mentioned figure if we found one, otherwise try to extract from query
                subject = mentioned_figure if mentioned_figure else search_query

                # If we still don't have a subject, try to extract using patterns
                if not subject or subject == search_query:
                    name_patterns = [
                        r"(?:write|create|compose) (?:a|an) (?:poem|story|tale|epic|ballad|narrative) (?:about|on|for) ([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
                        r"(?:write|create|compose) (?:a|an) (?:poem|story|tale|epic|ballad|narrative) (?:about|on|for) (?:the )?([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})'s",
                        r"(?:who|what) (?:is|was) ([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
                        r"tell me about ([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})",
                        r"(.*?)(?:'s|')? (?:biography|life story|history)",
                    ]

                    for pattern in name_patterns:
                        match = re.search(pattern, question, re.IGNORECASE)
                        if match and match.group(1).strip():
                            potential_subject = match.group(1).strip()
                            # Check if the extracted name matches any known figure or variation
                            for figure_name, variations in historical_figures.items():
                                if (
                                    potential_subject.lower() == figure_name.lower()
                                    or any(
                                        v.lower() == potential_subject.lower()
                                        for v in variations
                                    )
                                ):
                                    subject = figure_name
                                    break
                            if subject != search_query:
                                break

                # Determine the type of creative task
                task_type = "story"  # default
                if "poem" in q_lower:
                    task_type = "poem"
                elif any(term in q_lower for term in ["tale", "narrative"]):
                    task_type = "narrative"
                elif any(term in q_lower for term in ["song", "ballad"]):
                    task_type = "song"

                # Set up task completer with extracted information
                self.task_completer.add_information("subject", subject)
                self.task_completer.add_information("task_type", task_type)

                # Set tone based on the subject and task type
                tone = "informative"
                if any(term in q_lower for term in ["inspiring", "motivational"]):
                    tone = "inspirational"
                elif any(term in q_lower for term in ["dramatic", "epic"]):
                    tone = "dramatic"
                elif any(term in q_lower for term in ["funny", "humorous", "comic"]):
                    tone = "humorous"

                # Set style based on the subject
                style = "historical"
                if any(term in q_lower for term in ["modern", "contemporary"]):
                    style = "modern"
                elif any(term in q_lower for term in ["fantasy", "mythical"]):
                    style = "fantasy"
                elif any(term in q_lower for term in ["sci-fi", "futuristic"]):
                    style = "sci-fi"

                # Add all information to task completer
                self.task_completer.add_information("tone", tone)
                self.task_completer.add_information("style", style)
                self.task_completer.add_information("original_request", question)

                # If we have a historical figure, add context about their time period
                if mentioned_figure:
                    # Add time period context if available
                    time_period = self._get_time_period_for_figure(mentioned_figure)
                    if time_period:
                        self.task_completer.add_information("time_period", time_period)

                    # Add category (scientist, leader, artist, etc.)
                    category = self._get_category_for_figure(mentioned_figure)
                    if category:
                        self.task_completer.add_information("category", category)

                logger.info(
                    f"Initialized task completer for {task_type} about: {subject}"
                )

                # For creative tasks, generate content directly
                if is_creative_about_known or is_creative_writing:
                    logger.info(f"Generating {task_type} about: {subject}")
                    # First try to get background information about the subject
                    background_info = await self._research_subject(subject)
                    creative_prompt = f"A {task_type} about {subject} in the style of {tone} with a {style} style"
                    if background_info:
                        creative_prompt += (
                            f"\n\nUse this background information: {background_info}"
                        )
                    content = await self._generate_creative_content(creative_prompt)

                    # Save the content to the specified file if filename was provided
                    filename = "output.txt"  # Default filename
                    if "save it as" in q_lower:
                        match = re.search(
                            r'save it as ["\']?([^"\'\s]+)["\']?', q_lower
                        )
                        if match:
                            filename = match.group(1)

                    filepath = os.path.join(config.workspace_root, filename)
                    return {
                        "tool": "str_replace_editor",
                        "args": {
                            "command": "create",
                            "path": filepath,
                            "file_text": content,
                        },
                    }

                # For non-creative tasks, perform research
                if (
                    subject and len(subject.split()) <= 3
                ):  # Basic check for a reasonable name
                    logger.info(f"Initiating research for: {subject}")
                    research_query = f"{subject} biography key facts"
                    return {
                        "tool": "browser_use",
                        "args": {
                            "url": f"https://duckduckgo.com/?q={research_query.replace(' ', '+')}"
                        },
                    }

        # 4. File Operations -> str_replace_editor or python_execute
        file_keywords = [
            "read file",
            "file content",
            "edit file",
            "save file",
            "create file",
            "write file",
        ]
        if any(kw in q_lower for kw in file_keywords):
            match = re.search(r"file\s+(['\"]?)([\w\./\-\_]+\.\w+)\1", question)
            filename = match.group(2) if match else None
            if filename:
                if "read file" in q_lower or "content of file" in q_lower:
                    logger.info("Suggestion: Use str_replace_editor (view).")
                    return {
                        "tool": "str_replace_editor",
                        "args": {"command": "view", "path": filename},
                    }
                else:  # For write/edit, let LLM decide exact action
                    logger.info(
                        "Hint: File operation detected. Consider str_replace_editor or python_execute."
                    )
            else:
                logger.info(
                    "Hint: File operation detected but no filename extracted. Consider str_replace_editor or python_execute."
                )
            # Don't force tool for write/edit, just hint

        logger.info("No specific better tool identified. Allowing ask_human.")
        return None


def add_to_context(
    self,
    text: str,
    priority: str = "medium",
    source: str = "agent",
    tags: List[str] = None,
) -> None:
    """Add text to persistent memory for agent context."""
    if tags is None:
        tags = ["context"]
    elif not isinstance(tags, list):
        tags = [str(tags)]

    if priority == "high" or (text and len(text) > 10 and not text.isspace()):
        try:
            self.conversation_memory.store_memory(
                text=text,
                source=source,
                priority=priority,
                tags=list(set(tags)),  # Deduplicate tags
                metadata={"timestamp": time.time()},
            )
            logger.debug(
                f"Stored context (src: {source}, tags: {tags}): {text[:100]}..."
            )
        except Exception as e:
            logger.error(
                f"Failed to store context in persistent memory: {e}", exc_info=True
            )


def _get_time_period_for_figure(self, figure_name: str) -> Optional[str]:
    """Get the time period for a historical figure."""
    time_periods = {
        # Ancient
        "socrates": "Ancient Greece (5th-4th century BCE)",
        "plato": "Ancient Greece (4th century BCE)",
        "aristotle": "Ancient Greece (4th century BCE)",
        "alexander the great": "Ancient Greece (4th century BCE)",
        "julius caesar": "Roman Republic (1st century BCE)",
        "cleopatra": "Hellenistic Period (1st century BCE)",
        "buddha": "Ancient India (6th-5th century BCE)",
        "confucius": "Ancient China (6th-5th century BCE)",
        # Middle Ages
        "genghis khan": "Middle Ages (12th-13th century)",
        "queen elizabeth i": "Elizabethan Era (16th century)",
        # Modern
        "isaac newton": "Scientific Revolution (17th-18th century)",
        "wolfgang amadeus mozart": "Classical Period (18th century)",
        "ludwig van beethoven": "Classical/Romantic Period (late 18th-early 19th century)",
        "napoleon": "Napoleonic Era (early 19th century)",
        "queen victoria": "Victorian Era (19th century)",
        "charles darwin": "19th century",
        "nikola tesla": "Late 19th - Early 20th century",
        "thomas edison": "Late 19th - Early 20th century",
        "marie curie": "Early 20th century",
        "albert einstein": "Early to mid 20th century",
        "winston churchill": "World War II era (mid 20th century)",
        "martin luther king jr": "Civil Rights Movement (mid 20th century)",
        "stephen hawking": "Late 20th - Early 21st century",
        # Contemporary
        "steve jobs": "Late 20th - Early 21st century",
        "bill gates": "Late 20th - Early 21st century",
        "elon musk": "21st century",
        "mark zuckerberg": "21st century",
        "jeff bezos": "21st century",
        "warren buffett": "Late 20th - Early 21st century",
    }

    return time_periods.get(figure_name.lower())


def _get_category_for_figure(self, figure_name: str) -> Optional[str]:
    """Get the category (scientist, leader, etc.) for a historical figure."""
    categories = {
        # Leaders and rulers
        "genghis khan": "military leader",
        "napoleon": "military leader",
        "cleopatra": "ruler",
        "julius caesar": "military leader",
        "alexander the great": "military leader",
        "queen victoria": "monarch",
        "winston churchill": "political leader",
        "abraham lincoln": "political leader",
        "george washington": "political leader",
        "queen elizabeth i": "monarch",
        "queen elizabeth ii": "monarch",
        "nelson mandela": "political leader",
        "mahatma gandhi": "political leader",
        "martin luther king jr": "civil rights leader",
        # Scientists and inventors
        "albert einstein": "scientist",
        "marie curie": "scientist",
        "isaac newton": "scientist",
        "charles darwin": "scientist",
        "thomas edison": "inventor",
        "nikola tesla": "inventor",
        "galileo galilei": "scientist",
        "stephen hawking": "scientist",
        # Artists and writers
        "leonardo da vinci": "artist",
        "william shakespeare": "writer",
        "pablo picasso": "artist",
        "vincent van gogh": "artist",
        "wolfgang amadeus mozart": "composer",
        "ludwig van beethoven": "composer",
        # Philosophers and thinkers
        "socrates": "philosopher",
        "plato": "philosopher",
        "aristotle": "philosopher",
        "confucius": "philosopher",
        "buddha": "spiritual leader",
        # Religious figures
        "jesus christ": "religious leader",
        "prophet muhammad": "religious leader",
        "martin luther": "religious leader",
        # Modern figures
        "steve jobs": "business leader",
        "bill gates": "business leader",
        "mark zuckerberg": "business leader",
        "elon musk": "business leader",
        "jeff bezos": "business leader",
        "warren buffett": "business leader",
    }

    return categories.get(figure_name.lower())


@property
def context_manager(self):
    """Provides access to PersistentMemory instance for context operations."""
    if not self.conversation_memory:
        logger.error("Context manager accessed before PersistentMemory initialized!")
        # Attempt recovery - this might be too late depending on usage context
        self.initialize_helper()
        if not self.conversation_memory:
            raise RuntimeError("PersistentMemory not initialized.")
        return self.conversation_memory


async def think(self) -> bool:
    """Enhanced thinking process with dynamic prompting and context awareness."""
    logger.info(f"--- Starting think cycle (Step {self.step_count + 1}) ---")
    self.step_count += 1  # Increment step counter

    # Ensure agent is initialized (connects to MCP etc. on first think)
    if not self._initialized:
        await self.initialize_mcp_servers()

    # --- Pre-computation & State Checks ---
    # 1. Task Analysis on first user message
    is_new_task_interaction = len(self.memory.messages) <= 2
    if is_new_task_interaction and not self.current_task:
        user_message = next(
            (msg for msg in self.memory.messages if msg.role == "user"), None
        )
        if user_message and user_message.content:
            await self.analyze_new_task(user_message.content)

    # 2. Check for Task Completion Readiness
    if (
        self.task_completer
        and self.task_completer.is_ready_to_complete()
        and not self.task_completed
    ):
        # Skip if we've already handled this through keyword routing
        if hasattr(self, "_task_result") and self._task_result:
            logger.info("✅ Task already completed via keyword routing.")
            return False  # Continue processing

        logger.info("✅ Task completion criteria met. Generating deliverable...")
        try:
            self._add_default_information()  # Ensure defaults filled
            deliverable = self.task_completer.create_deliverable()
            file_path = self._save_deliverable_to_file(deliverable)
            save_msg = (
                f"\nDeliverable saved to workspace: {file_path}" if file_path else ""
            )
            final_output = f"Task '{self.current_task}' completed.\n\nDeliverable:\n{deliverable}{save_msg}"

            # Set task result for display
            self._task_result = f"Task completed\n\n{deliverable}"
            self._deliverable_content = deliverable
            self._deliverable_path = file_path

            # Update memory and context, then mark completed
            self.update_memory("assistant", final_output)
            self.add_to_context(
                final_output,
                priority="high",
                source="task_completion",
                tags=["task_completed", "deliverable"],
            )
            self.task_completed = True

            # Add a prompt for the next task
            next_task_prompt = "\n\nTask completed successfully! What would you like me to help you with next?"
            self.update_memory("assistant", next_task_prompt)
            return False  # Continue processing
        except Exception as e:
            logger.error(f"Error during task completion/saving: {e}", exc_info=True)
            self.add_to_context(
                f"Error finalizing task: {e}",
                priority="high",
                source="error",
                tags=["error", "task_completion"],
            )
            # Continue thinking cycle to report error

        # --- Dynamic Prompt Construction ---
        base_prompt = self.next_step_prompt
        prompt_prefix = ""

        # 1. Browser Context / Hints
        active_url = (
            self.browser_context_helper.get_current_url()
            if self.browser_context_helper
            else None
        )
        if active_url and self.browser_used:
            browser_ctx = (
                await self.browser_context_helper.format_next_step_prompt()
            )  # Viewport info
            prompt_prefix += browser_ctx
            # Navigation suggestion
            next_nav = self.browser_navigator.suggest_browser_action(
                active_url, self.current_task
            )
            if next_nav and next_nav.get("tool") == "browser_use":
                s_url = next_nav["args"].get("url")
                if s_url and s_url != active_url and s_url not in self.visited_urls:
                    prompt_prefix += (
                        f"\nNavigation Suggestion: Consider visiting {s_url} next.\n"
                    )
        elif not self.browser_used and self.mentioned_urls:
            url_to_check = self._get_most_relevant_url(self.current_task)
            if url_to_check and url_to_check not in self.visited_urls:
                prompt_prefix += f"\nHint: Task mentions URLs like {url_to_check}. Use 'browser_use' to visit it.\n"

        # 2. Relevant Memories
        try:
            # Use semantic search (this is an async method)
            relevant_memories = await self.conversation_memory.search_memories_semantic(
                self.current_task, limit=4
            )
            if relevant_memories:
                mem_items = [
                    f"- {mem.text[:150]}... (src: {mem.source}, score: {score:.2f})"
                    for mem, score in relevant_memories
                ]
                prompt_prefix += (
                    "\n--- Relevant Info from Memory ---\n"
                    + "\n".join(mem_items)
                    + "\n---\n"
                )
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")

        # 3. SmartAskHuman Context (Answered Questions)
        smart_ask_tool = self.available_tools.get_tool("ask_human")
        if isinstance(smart_ask_tool, SmartAskHuman):
            answered = (
                smart_ask_tool.get_answered_questions()
            )  # Get dict {question: answer}
            if answered:
                prompt_prefix += "\n--- Previously Answered (Do NOT ask again) ---\n"
                for q, a in answered.items():
                    prompt_prefix += f"Q: {q} -> A: {a[:80]}...\n"
                prompt_prefix += "---\n"

        # 4. Task Completion Reminder (if applicable)
        # Example: Remind after N steps if task not completed
        if (
            not self.task_completed
            and self.step_count > 5
            and self.task_completer
            and not self.task_completer.is_ready_to_complete()
        ):
            missing = self.task_completer.get_missing_info()
            if missing:
                prompt_prefix += f"\nReminder: Still need info for: {', '.join(missing)}. Focus on gathering this or completing the task.\n"
            else:
                prompt_prefix += "\nReminder: You seem to have enough info. Aim to synthesize and complete the task soon.\n"

        # Combine prompt parts
        final_prompt = f"{prompt_prefix}\n{base_prompt}"

        # --- Execute Superclass Think ---
        original_prompt = self.next_step_prompt
        self.next_step_prompt = final_prompt
        logger.debug(f"Using combined next_step_prompt:\n{self.next_step_prompt}")
        try:
            result = (
                await super().think()
            )  # Let ToolCallAgent handle LLM call and tool parsing
        finally:
            self.next_step_prompt = original_prompt  # Restore base prompt

        logger.info(f"--- Finished think cycle (Step {self.step_count}) ---")
        return result

    def _add_default_information(self) -> None:
        """Add default or inferred information if task completer requires it."""
        if not self.task_completer:
            return
        required = self.task_completer.get_missing_info()
        if not required:
            return

        logger.info(f"Attempting to add default/inferred info for: {required}")
        # Example: Infer product name for marketing plan
        if (
            "marketing_plan" in self.task_completer.task_type.lower()
            and "product_name" in required
        ):
            match = re.search(
                r"(?:marketing plan|plan) for ([\w\s\-]+)", self.current_task.lower()
            )
            name = match.group(1).strip() if match else "the specified product/service"
            self.task_completer.add_information("product_name", name)
            logger.info(f"Inferred product_name: {name}")

        # Add more logic here based on task types and common missing info

    def _save_deliverable_to_file(self, deliverable_content: str) -> Optional[str]:
        """Saves deliverable content to a structured file in the workspace."""
        if not deliverable_content:
            return None
        try:
            workspace_root = Path(config.workspace_root)
            deliverables_dir = workspace_root / "deliverables"
            task_slug = _slugify(
                self.current_task or self.task_completer.task_type or "task"
            )[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_folder = deliverables_dir / f"{task_slug}_{timestamp}"
            task_folder.mkdir(parents=True, exist_ok=True)

            ext = ".md"  # Default markdown
            task_type_lower = (self.task_completer.task_type or "").lower()
            if "code" in task_type_lower:
                ext = ".py"  # Simple guess
            elif "report" in task_type_lower:
                ext = ".txt"
            elif "plan" in task_type_lower:
                ext = ".md"

            file_name = (
                f"{_slugify(self.task_completer.task_type or 'deliverable')}{ext}"
            )
            deliverable_path = task_folder / file_name

            with open(deliverable_path, "w", encoding="utf-8") as f:
                f.write(deliverable_content)
            relative_path = str(deliverable_path.relative_to(workspace_root))
            logger.info(f"✅ Deliverable saved to: {relative_path}")
            return relative_path
        except Exception as e:
            logger.error(f"Error saving deliverable: {e}", exc_info=True)
            return None

    def update_memory(self, role: str, content: str) -> None:
        """Adds a message to the main conversation memory (part of ToolCallAgent)."""
        self.memory.add_message(Message(role=role, content=content))
        logger.debug(f"Updated main memory ({role}): {content[:100]}...")

    # Included previously necessary helpers like _process_content_for_task etc. if needed by the above logic
    # Need to ensure all called helpers are defined.
    def _extract_section(
        self, content: str, section_name: str, max_paragraphs: int = 3
    ) -> str:
        """Extract a specific section from content based on section name.

        Args:
            content: The full content to search in
            section_name: The name of the section to extract
            max_paragraphs: Maximum number of paragraphs to include

        Returns:
            str: The extracted section content or empty string if not found
        """
        try:
            # First try to find the section using HTML structure
            soup = BeautifulSoup(content, "html.parser")

            # Look for common section patterns
            patterns = [
                # Headers with the section name
                lambda: soup.find(
                    ["h1", "h2", "h3"],
                    string=lambda t: t and section_name.lower() in t.lower(),
                ),
                # Section/div with class or id containing section name
                lambda: soup.find(
                    ["section", "div"],
                    class_=lambda c: c and section_name.lower() in c.lower(),
                ),
                lambda: soup.find(
                    ["section", "div"],
                    id=lambda i: i and section_name.lower() in i.lower(),
                ),
                # Articles with section name in class/id
                lambda: soup.find(
                    "article", class_=lambda c: c and section_name.lower() in c.lower()
                ),
                # Fallback: find any element with section name in text
                lambda: soup.find(
                    string=lambda t: t and section_name.lower() in t.lower()
                ),
            ]

            section_element = None
            for pattern in patterns:
                try:
                    element = pattern()
                    if element:
                        section_element = element
                        break
                except:
                    continue

            if not section_element:
                return ""

            # Get the parent element if we found a text node
            if not hasattr(section_element, "find_all"):
                section_element = section_element.parent

            # Extract the content after the section header
            content_parts = []
            current = section_element

            # Get the next siblings until we hit another header or max_paragraphs
            while current and len(content_parts) < max_paragraphs:
                current = current.find_next_sibling()
                if not current:
                    break

                # Stop at next section
                if (
                    current.name
                    and current.name.startswith("h")
                    and len(current.name) == 2
                ):
                    break

                # Get text from paragraph or other block element
                if current.name in ["p", "div", "section"]:
                    text = current.get_text(separator=" ", strip=True)
                    if text:
                        content_parts.append(text)

            return " ".join(content_parts) if content_parts else ""

        except Exception as e:
            logger.error(f"Error extracting section '{section_name}': {str(e)}")
            return ""

    def _process_content_for_task(self, url: str, content: str) -> None:
        """Process extracted web content to fill TaskCompleter info.

        This method processes web content and extracts relevant information
        based on the current task type and requirements.
        """
        if not self.task_completer:
            logger.debug("No task completer available, skipping content processing")
            return

        logger.debug(
            f"Processing content from {url} for task type: {self.task_completer.task_type}"
        )
        task_type = (self.task_completer.task_type or "").lower()
        required = self.task_completer.get_missing_info()

        # Always process content to extract and store information
        # even if we don't have specific requirements yet

        # Handle creative content tasks (poems, stories, etc.)
        if any(t in task_type for t in ["poem", "story", "creative"]):
            try:
                # Initialize data structures
                key_facts = []
                metadata = {}

                # Parse HTML and extract basic information
                soup = BeautifulSoup(content, "html.parser")

                # Extract title and metadata
                title = soup.title.string if soup.title else ""
                if title and len(title) < 200:  # Reasonable title length
                    metadata["title"] = title.strip()
                    key_facts.append(f"Title: {title.strip()}")

                # Extract meta description and keywords if available
                for meta in soup.find_all("meta"):
                    name = meta.get("name", "").lower()
                    if name == "description":
                        metadata["description"] = meta.get("content", "")
                    elif name == "keywords":
                        metadata["keywords"] = meta.get("content", "")

                # Extract key sections using semantic HTML and common patterns
                section_headers = [
                    "early life",
                    "biography",
                    "achievements",
                    "legacy",
                    "personal life",
                    "accomplishments",
                    "timeline",
                    "key events",
                    "major works",
                    "notable works",
                ]

                for header in section_headers:
                    section = self._extract_section(content, header)
                    if (
                        section and len(section) > 50
                    ):  # Only include meaningful sections
                        key_facts.append(f"{header.title()}: {section}")

                # Extract dates and key events with better patterns
                date_patterns = [
                    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                    r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
                    r"\b(?:19|20)\d{2}\b",
                ]

                dates = []
                for pattern in date_patterns:
                    dates.extend(re.findall(pattern, content, re.IGNORECASE))

                if dates:
                    unique_dates = sorted(
                        set(dates), key=len, reverse=True
                    )  # Sort by length (most specific first)
                    key_facts.append(
                        f"Key Dates: {', '.join(unique_dates[:10])}"
                    )  # Limit to top 10 dates

                # Add to task completer if we found useful information
                if key_facts:
                    # Store all extracted facts
                    facts_text = "\n".join(key_facts)
                    self.task_completer.add_information("key_facts", facts_text)
                    logger.info(
                        f"Extracted {len(key_facts)} key facts for creative content"
                    )

                    # Store metadata if available
                    if metadata:
                        for key, value in metadata.items():
                            if value:  # Only store non-empty values
                                self.task_completer.add_information(
                                    f"meta_{key}", value
                                )

                    # Set subject if not already set
                    if (
                        "subject" in required
                        and not self.task_completer.get_information("subject")
                    ):
                        if "title" in metadata:
                            self.task_completer.add_information(
                                "subject", metadata["title"]
                            )
                        elif title:
                            self.task_completer.add_information("subject", title)

                    # Infer tone from content if needed
                    if "tone" in required and not self.task_completer.get_information(
                        "tone"
                    ):
                        content_lower = content.lower()
                        tone_keywords = {
                            "serious": [
                                "tragic",
                                "sad",
                                "death",
                                "war",
                                "battle",
                                "struggle",
                            ],
                            "triumphant": [
                                "victory",
                                "success",
                                "achievement",
                                "won",
                                "triumph",
                            ],
                            "inspiring": [
                                "inspire",
                                "courage",
                                "brave",
                                "determination",
                                "overcome",
                            ],
                            "informative": [
                                "report",
                                "study",
                                "research",
                                "findings",
                                "analysis",
                            ],
                        }

                        detected_tones = []
                        for tone, keywords in tone_keywords.items():
                            if any(keyword in content_lower for keyword in keywords):
                                detected_tones.append(tone)

                        # Default to informative if no strong tone detected
                        tone = detected_tones[0] if detected_tones else "informative"
                        self.task_completer.add_information("tone", tone)

                    # Set style if needed
                    if "style" in required and not self.task_completer.get_information(
                        "style"
                    ):
                        # Default to historical for biographies, otherwise narrative
                        style = (
                            "historical"
                            if "biography" in content_lower
                            else "narrative"
                        )
                        self.task_completer.add_information("style", style)

                    # Add source URL for reference
                    self.task_completer.add_information("source_url", url)

            except Exception as e:
                logger.error(
                    f"Error processing content for creative task: {e}", exc_info=True
                )

        # Handle marketing plan tasks
        elif "marketing_plan" in task_type:
            # Simple keyword-based extraction for example
            if "product_name" in required:
                # Try to find product name in title or h1
                soup = BeautifulSoup(content[:5000], "html.parser")  # Parse snippet
                title = soup.title.string if soup.title else ""
                h1 = soup.h1.string if soup.h1 else ""
                # Basic logic, needs improvement
                if title and len(title) < 50:
                    self.task_completer.add_information("product_name", title.strip())
                elif h1 and len(h1) < 50:
                    self.task_completer.add_information("product_name", h1.strip())

            sections_keywords = {
                "target_audience": ["target audience", "who is it for", "customers"],
                "value_proposition": ["value proposition", "benefits", "why choose"],
                "features": ["features", "capabilities", "what it does"],
            }
            for section_key, keywords in sections_keywords.items():
                if section_key in required:
                    # Try to find section using keywords (crude example)
                    for kw in keywords:
                        if kw in content.lower():
                            extracted = self._extract_section(content, kw)  # Use helper
                            self.task_completer.add_information(section_key, extracted)
                            logger.info(
                                f"Extracted '{section_key}' based on keyword '{kw}'"
                            )
                            break  # Stop after first keyword match for this section


# --- Module Level Utility ---
import asyncio  # Ensure asyncio is imported if used standalone


def _slugify(text):
    """Convert text into a filesystem-friendly slug."""
    text = str(text).lower()
    text = re.sub(r"&", "-and-", text)  # Replace &
    text = re.sub(r"[^a-z0-9\-\_]+", "-", text)  # Replace non-alphanumeric by -
    text = re.sub(r"-+", "-", text)  # Collapse multiple dashes
    text = text.strip("-")  # Remove leading/trailing dashes
    return text or "task"  # Return 'task' if empty


# Required for MCP connection logic if run standalone/tested directly
import asyncio
