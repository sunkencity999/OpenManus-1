"""
Improved Manus agent with enhanced reasoning capabilities.
"""
from typing import Dict, List, Optional, Any, Set
import re
import json
import os

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.browser_navigator import BrowserNavigator
from app.agent.url_detector import URLDetector
from app.agent.task_completer import TaskCompleter
from app.agent.memory_manager import ConversationMemory, ToolUsageTracker
from app.agent.persistent_memory import PersistentMemory
from app.agent.task_analyzer import TaskAnalyzer
from app.agent.toolcall import ToolCallAgent
from app.config import config
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
    description: str = "A versatile agent that can solve various tasks using multiple tools including MCP-based tools, with improved reasoning"

    system_prompt: str = IMPROVED_SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = IMPROVED_NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # MCP clients for remote tool access
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # Add general-purpose tools to the tool collection, using SmartAskHuman instead of AskHuman
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            StrReplaceEditor(),
            SmartAskHuman(),
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
    conversation_memory: PersistentMemory = None
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

    # Memory management components
    tool_tracker: ToolUsageTracker = Field(default_factory=ToolUsageTracker)
    
    # Track URLs mentioned in the conversation
    mentioned_urls: Set[str] = Field(default_factory=set)
    
    # Browser navigation helper
    browser_navigator: BrowserNavigator = Field(default_factory=BrowserNavigator)
    
    # Task completion tracking
    current_task: str = ""
    task_completed: bool = False
    
    # Flag to prevent repeated tool calls
    _prevent_repeated_questions: bool = True

    @model_validator(mode="after")
    def initialize_helper(self) -> "ImprovedManus":
        """Initialize basic components synchronously."""
        self.browser_context_helper = BrowserContextHelper(self)
        
        # Use PersistentMemory instead of ConversationMemory
        if self.memory_db_path is not None:
            # If a specific path is provided, use it
            memory_path = os.path.join(config.workspace_root, self.memory_db_path)
            self.conversation_memory = PersistentMemory(db_path=memory_path)
        else:
            # Otherwise use the default platform-specific path
            self.conversation_memory = PersistentMemory()
        
        self.browser_navigator = BrowserNavigator()
        self.url_detector = URLDetector()
        self.task_completer = TaskCompleter()
        self.task_analyzer = TaskAnalyzer(llm=self.llm)
        
        # Get the SmartAskHuman tool and set its context manager
        for tool in self.available_tools.tools:
            if tool.name == "ask_human":
                # Don't set the context manager here - we'll do it in the property
                pass
                
        return self

    @classmethod
    async def create(cls, **kwargs) -> "ImprovedManus":
        """Factory method to create and properly initialize an ImprovedManus instance."""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        instance._initialized = True
        return instance

    async def initialize_mcp_servers(self) -> None:
        """Initialize connections to configured MCP servers."""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """Connect to an MCP server and add its tools."""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # Update available tools with only the new tools from this server
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """Disconnect from an MCP server and remove its tools."""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # Rebuild available tools without the disconnected server's tools
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """Clean up ImprovedManus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # Disconnect from all MCP servers only if we were initialized
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def analyze_new_task(self, task: str) -> None:
        """Analyze a new task to determine required information and steps."""
        # Reset components for new task
        self.conversation_memory = ConversationMemory()
        self.url_detector.extract_urls(task)  # Pre-extract URLs from the task
        
        # Initialize or reset task completer
        from app.agent.task_completer import TaskCompleter
        if not hasattr(self, 'task_completer'):
            self.task_completer = TaskCompleter()
        
        # Analyze task to determine requirements
        self.task_completer.analyze_task(task)
        
        # Store the current task description
        self.current_task = task
        self.task_completed = False
        
        # Initialize step counter if not already set
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        else:
            self.step_count = 0  # Reset for new task
        
        if self.task_analyzer:
            plan = await self.task_analyzer.analyze_task(task)
            
            # Set the required context in the context manager
            required_context = plan.all_required_context()
            self.context_manager.set_required_context(list(required_context))
            
            # Log the plan
            logger.info(f"📋 Task plan created with {len(plan.steps)} steps")
            for i, step in enumerate(plan.steps):
                logger.info(f"  Step {i+1}: {step.description}")
                if step.required_context:
                    logger.info(f"    Required context: {', '.join(step.required_context)}")
                    
            # If this is a task that involves a website, create a navigation plan
            urls = self._extract_urls(task)
            if urls:
                base_url = urls[0]
                navigation_plan = self.browser_navigator.create_navigation_plan(base_url, task)
                logger.info(f"🌐 Created website navigation plan with {len(navigation_plan)} steps")
                for i, url in enumerate(navigation_plan):
                    logger.info(f"  Navigation step {i+1}: {url}")

    async def execute_tool(self, command: ToolCall) -> str:
        """Override execute_tool to add memory tracking and tool suggestion."""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # Parse arguments
            args = json.loads(command.function.arguments or "{}")
            
            # Handle browser_use tool specially to improve navigation
            if name == "browser_use" and "url" in args:
                url = args["url"]
                
                # Check if this URL contains a fragment that should be navigated to
                if "#" in url and not url.endswith("#"):
                    # Record that we're trying to navigate to a specific section
                    logger.info(f"🌐 Navigating to specific section: {url}")
                    
                # Add to mentioned URLs
                self.mentioned_urls.add(url)
                
                # Execute the browser tool
                result = await self.available_tools.execute(name=name, tool_input=args)
                
                # Record the visit in the browser navigator
                if isinstance(result, str):
                    self.browser_navigator.record_visit(url, result)
                    
                    # If we're working on a task that requires information gathering
                    if "business plan" in self.current_task.lower() or "report" in self.current_task.lower():
                        # Extract key information from the page
                        keywords = ["features", "about", "pricing", "benefits", "description"]
                        extracted_info = self.browser_navigator.extract_key_information(result, keywords)
                        
                        if extracted_info:
                            logger.info(f"📓 Extracted key information from {url}: {list(extracted_info.keys())}")
                            for key, value in extracted_info.items():
                                self.context_manager.add_context(
                                    key=f"website_{key}",
                                    value=value,
                                    confidence=0.9,
                                    source=f"browser:{url}"
                                )
                
                # Record tool usage
                self.tool_tracker.record_tool_usage(name, args, "success")
                
                # Format result
                observation = f"Observed output of cmd `{name}` executed:\n{str(result)}" if result else f"Cmd `{name}` completed with no output"
                
                # Check if we should suggest the next navigation step
                next_action = self.browser_navigator.suggest_browser_action(url, self.current_task)
                if next_action and next_action["tool"] == "browser_use":
                    next_url = next_action["args"]["url"]
                    if self.browser_navigator.should_visit_url(next_url):
                        observation += f"\n\nI notice there's a relevant section at {next_url} that might contain useful information for your task."
                
                return observation
            
            # Handle ask_human tool specially
            if name == "ask_human" and "inquire" in args:
                question = args["inquire"]
                
                # Check if this is a redundant question
                if self._is_redundant_question(question):
                    # If we have previous answers that are relevant, use those
                    relevant_context = self.conversation_memory.get_relevant_context(question)
                    if relevant_context:
                        return f"Based on previous conversation, I already know: {'; '.join(relevant_context)}"
                    
                    # If we should be using the browser instead
                    if self.mentioned_urls and not any(tc.function.name == "browser_use" for tc in self.tool_calls):
                        # Find the most relevant URL to visit
                        url = self._get_most_relevant_url(question)
                        return f"I should check the website first. Let me browse to {url} instead of asking redundant questions."
                
                # Check if the question contains a URL
                urls = self._extract_urls(question)
                if urls:
                    url = urls[0]
                    self.mentioned_urls.add(url)
                    return f"I see a URL in your message. Let me visit {url} to gather information instead of asking questions."
                
                # Record this question
                self.conversation_memory.add_question(question)
            
            # Check if there's a better tool to use
            better_tool = await self._suggest_better_tool(name, args)
            if better_tool:
                logger.info(f"🔄 Suggesting better tool: {better_tool.function.name} instead of {name}")
                # Execute the better tool instead
                return await super().execute_tool(better_tool)
            
            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)
            
            # Record tool usage
            self.tool_tracker.record_tool_usage(name, args, "success")
            
            # If this was ask_human, record the response
            if name == "ask_human" and "inquire" in args and isinstance(result, str):
                self.conversation_memory.add_response(args["inquire"], result)
                
                # Check if the response contains a URL
                urls = self._extract_urls(result)
                if urls:
                    self.mentioned_urls.update(urls)
            
            # Handle special tools
            await self._handle_special_tool(name=name, result=result)

            # Check if result is a ToolResult with base64_image
            if hasattr(result, "base64_image") and result.base64_image:
                # Store the base64_image for later use in tool_message
                self._current_base64_image = result.base64_image

            # Format result for display (standard case)
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except Exception as e:
            # Record tool failure
            self.tool_tracker.record_tool_usage(name, args, "failure")
            
            # Handle the error
            error_msg = f"Error executing {name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def execute_browser_use(self, action: str, **kwargs) -> str:
        """Execute browser use tool with enhanced navigation."""
        # Track that we're using the browser
        self.browser_used = True
        
        # Execute the browser action
        try:
            result = await super().execute_browser_use(action, **kwargs)
            
            # If this was a navigation action, record the URL
            if action == "go_to_url" and "url" in kwargs:
                url = kwargs["url"]
                self.visited_urls.add(url)
                
                # Extract content from the page for context
                content = await self.get_browser_content()
                
                # Store in persistent memory with higher priority
                self.conversation_memory.store_memory(
                    text=f"Content from {url}:\n{content[:1000]}...",
                    source="browser",
                    priority="high",
                    tags=["web_content", "visited_url"],
                    metadata={"url": url, "timestamp": time.time()}
                )
                
                # Add summary to context
                self.add_to_context(f"I've visited {url} and found relevant content.")
                
                # Process content for task completion
                self._process_content_for_task(url, content)
                
                # Suggest next navigation based on task
                next_action = self.browser_navigator.suggest_next_action(url, self.current_task)
                if next_action:
                    self.add_to_context(f"I should navigate to {next_action['url']} next.")
            
            return result
        except Exception as e:
            error_msg = f"Error using browser: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def add_to_context(self, text: str, priority: str = "medium", source: str = "agent", 
                      tags: List[str] = None) -> None:
        """Add text to the agent's context for better reasoning and store in persistent memory."""
        # Initialize context if it doesn't exist
        if not hasattr(self, 'context'):
            self.context = ""
        
        # Add to in-memory context
        self.context += f"\n{text}"
        
        # Store in persistent memory
        if tags is None:
            tags = ["context"]
        
        # Avoid storing very short or uninformative text
        if len(text) > 10 and not text.isspace():
            self.conversation_memory.store_memory(
                text=text,
                source=source,
                priority=priority,
                tags=tags
            )
        
    @property
    def context_manager(self):
        """Property to provide backward compatibility with code expecting context_manager."""
        return self.conversation_memory
        
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from content based on section name."""
        # Look for section headers or divs with this name
        pattern = re.compile(f"<h[1-6][^>]*>{section_name}[^<]*</h[1-6]>|<div[^>]*id=['\"]{0,1}{section_name}['\"]{0,1}[^>]*>|<section[^>]*id=['\"]{0,1}{section_name}['\"]{0,1}[^>]*>", re.IGNORECASE)
        match = pattern.search(content)
        
        if match:
            # Extract a reasonable chunk of content after this section header
            start_idx = match.start()
            end_idx = min(start_idx + 1000, len(content))  # Get about 1000 chars of content
            return content[start_idx:end_idx]
        
        # If no section header found, look for paragraphs containing the section name
        paragraphs = re.findall(r'<p[^>]*>.*?</p>', content, re.DOTALL)
        for para in paragraphs:
            if section_name.lower() in para.lower():
                return para
                
        # If still not found, just return a snippet around the first mention
        idx = content.lower().find(section_name.lower())
        if idx >= 0:
            start_idx = max(0, idx - 100)
            end_idx = min(len(content), idx + 500)
            return content[start_idx:end_idx]
            
        return f"No information about {section_name} found."
        
    async def _gather_relevant_memories(self) -> str:
        """Gather relevant memories for the current task using semantic search."""
        if not self.current_task:
            return ""
            
        try:
            # Use semantic search to find relevant memories
            results = await self.conversation_memory.search_memories_semantic(self.current_task, limit=10)
            
            # Filter and format relevant memories
            relevant_content = []
            for memory, score in results:
                if score > 0.7:  # Only include highly relevant memories
                    relevant_content.append(f"- {memory.text} (relevance: {score:.2f})")
            
            if relevant_content:
                return "\n\nRelevant information from memory:\n" + "\n".join(relevant_content)
            return ""
        except Exception as e:
            logger.error(f"Error gathering memories: {str(e)}")
            return ""
    
    async def _force_task_completion(self) -> None:
        """Force the completion of the current task using persistent memory."""
        if not hasattr(self, 'task_completer') or self.task_completed:
            return
            
        # Add default information if we don't have enough
        self._add_default_information()
        
        # Gather relevant memories for this task
        relevant_memories = await self._gather_relevant_memories()
        if relevant_memories:
            self.add_to_context(relevant_memories, priority="high", tags=["task_completion", "memory_retrieval"])
        
        # Create the deliverable
        deliverable = self.task_completer.create_deliverable()
        
        # Store the completed task in memory
        self.conversation_memory.store_task(
            task_description=self.current_task,
            completed=True,
            outcome="Task completed successfully",
            metadata={"deliverable_length": len(deliverable)}
        )
        
        # Check if the task involves saving to a file
        file_path = self._extract_file_path_from_task()
        if file_path:
            # Save the deliverable to the file
            try:
                self._save_content_to_file(file_path, deliverable)
                save_message = f"\n\nThe content has been saved to {file_path}"
                deliverable += save_message
                logger.warning(f"Content saved to file: {file_path}")
            except Exception as e:
                error_message = f"\n\nError saving to {file_path}: {str(e)}"
                deliverable += error_message
                logger.error(f"Error saving to file: {str(e)}")
        
        # Add to memory instead of context
        self.update_memory("assistant", f"Based on the information I've gathered, here's the completed deliverable:\n\n{deliverable}")
        
        # Store the response for later retrieval
        self.response = deliverable
        
        # Mark task as completed
        self.task_completed = True
        logger.warning("🎉 Task completion forced successfully!")
        
    def _extract_file_path_from_task(self) -> Optional[str]:
        """Extract file path from task description if it exists."""
        if not self.current_task:
            return None
            
        # Common patterns for file paths
        patterns = [
            r"save (?:it|this|the poem|the content) (?:to|as|in) ['\"]?([\w\./]+\.\w+)['\"]?",
            r"save (?:as|to|in) ['\"]?([\w\./]+\.\w+)['\"]?",
            r"create (?:a|the) file ['\"]?([\w\./]+\.\w+)['\"]?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.current_task.lower())
            if match:
                file_path = match.group(1)
                # If path doesn't start with /, assume it's relative to workspace
                if not file_path.startswith("/"):
                    file_path = f"{config.workspace_root}/{file_path}"
                return file_path
                
        return None
        
    def _save_content_to_file(self, file_path: str, content: str) -> None:
        """Save content to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write content to file
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _add_default_information(self) -> None:
        """Add default information for the current task if missing."""
        # For marketing plans
        if self.task_completer.task_type == "marketing_plan":
            # Extract product name from task if not already set
            if "product_name" not in self.task_completer.gathered_info:
                product_name = self._extract_product_name_from_task()
                if product_name:
                    self.task_completer.add_information("product_name", product_name)
                else:
                    self.task_completer.add_information("product_name", "the product")
            
            # Add default target audience if not set
            if "target_audience" not in self.task_completer.gathered_info:
                self.task_completer.add_information(
                    "target_audience", 
                    "Potential customers interested in the product's features and benefits."
                )
            
            # Add default value proposition if not set
            if "value_proposition" not in self.task_completer.gathered_info:
                product_name = self.task_completer.gathered_info.get("product_name", "The product")
                self.task_completer.add_information(
                    "value_proposition", 
                    f"{product_name} provides significant value through its unique features and capabilities."
                )
        # For poems
        elif self.task_completer.task_type == "poem":
            # Add default subject if not set
            if "subject" not in self.task_completer.gathered_info:
                subject = self._extract_subject_from_task()
                if subject:
                    self.task_completer.add_information("subject", subject)
                else:
                    self.task_completer.add_information("subject", "nature")
            
            # Add default style if not set
            if "style" not in self.task_completer.gathered_info:
                style = self._extract_style_from_task()
                if style:
                    self.task_completer.add_information("style", style)
                else:
                    self.task_completer.add_information("style", "modern")
    
    def _extract_product_name_from_task(self) -> Optional[str]:
        """Extract product name from the current task."""
        if not self.current_task:
            return None
            
        # Look for capitalized words that might be product names
        words = self.current_task.split()
        for word in words:
            if len(word) > 3 and word[0].isupper() and word.lower() not in ["create", "make", "develop", "build", "this", "that"]:
                return word
                
        return None
        
    def _extract_subject_from_task(self) -> Optional[str]:
        """Extract the subject of a poem from the task description."""
        if not self.current_task:
            return None
            
        # Common patterns for poem subjects
        patterns = [
            r"poem about ([\w\s]+)",
            r"poem on ([\w\s]+)",
            r"poem for ([\w\s]+)",
            r"about ([\w\s]+) in the style"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.current_task.lower())
            if match:
                return match.group(1).strip()
                
        return None
        
    def _extract_style_from_task(self) -> Optional[str]:
        """Extract the style of a poem from the task description."""
        if not self.current_task:
            return None
            
        # Common patterns for poem styles
        patterns = [
            r"in the style of ([\w\s]+)",
            r"like ([\w\s]+)",
            r"similar to ([\w\s]+)",
            r"in ([\w\s]+)'s style"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.current_task.lower())
            if match:
                return match.group(1).strip()
                
        return None
        
    def _process_content_for_task(self, url: str, content: str) -> None:
        """Process content from a webpage for task completion."""
        if not hasattr(self, 'task_completer'):
            return
            
        # Extract product name
        if "choiceengine" in url.lower():
            self.task_completer.add_information("product_name", "ChoiceEngine")
            
        # Extract information based on task type
        if self.task_completer.task_type == "marketing_plan":
            # Add general content
            self.task_completer.add_information("about", content[:2000])
            
            # Extract key sections for marketing plan
            sections = [
                ("target_audience", ["target audience", "users", "customers", "who uses", "demographic"]),
                ("value_proposition", ["value proposition", "benefits", "why use", "advantages"]),
                ("marketing_channels", ["marketing channels", "promotion", "advertising", "reach users"]),
                ("key_messaging", ["messaging", "key points", "selling points", "features"]),
                ("competitive_analysis", ["competition", "competitors", "market position", "alternative"]),
                ("budget", ["budget", "cost", "pricing", "investment"])
            ]
            
            # Extract each section
            for section_key, keywords in sections:
                for keyword in keywords:
                    if keyword in content.lower():
                        section_content = self._extract_section(content, keyword)
                        self.task_completer.add_information(section_key, section_content)
                        break
                        
            # If we're on the features page, extract that specifically
            if "#features" in url:
                features_content = self._extract_section(content, "features")
                self.task_completer.add_information("key_messaging", features_content)
                
            # If we're on the about page, extract company information
            if "about" in url:
                about_content = self._extract_section(content, "about")
                self.task_completer.add_information("executive_summary", about_content)
                
        elif self.task_completer.task_type == "business_plan":
            # Similar extraction for business plans
            self.task_completer.add_information("overview", content[:2000])
            
            # Extract key sections for business plan
            sections = [
                ("executive_summary", ["summary", "overview", "about"]),
                ("market_analysis", ["market", "industry", "trends", "opportunity"]),
                ("product_line", ["product", "features", "services", "offerings"]),
                ("marketing_strategy", ["marketing", "promotion", "advertising", "sales"]),
                ("financial_projections", ["financial", "revenue", "profit", "pricing"])
            ]
            
            # Extract each section
            for section_key, keywords in sections:
                for keyword in keywords:
                    if keyword in content.lower():
                        section_content = self._extract_section(content, keyword)
                        self.task_completer.add_information(section_key, section_content)
                        break

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return re.findall(url_pattern, text)

    def _should_use_browser(self, user_message: str) -> bool:
        """Determine if browser should be used based on user message."""
        # Check for explicit URL mentions
        urls = self._extract_urls(user_message)
        if urls:
            self.mentioned_urls.update(urls)
            return True
        
        # Check for phrases suggesting web browsing
        browse_phrases = [
            "check the website", "visit the site", "go to", "browse to", 
            "look at the page", "check the page", "about page", "website"
        ]
        return any(phrase in user_message.lower() for phrase in browse_phrases)

    def _is_redundant_question(self, question: str) -> bool:
        """Check if a question is redundant based on conversation history."""
        if not self._prevent_repeated_questions:
            return False
        
        # Check if this is similar to a previously asked question
        if self.conversation_memory.is_similar_question(question):
            logger.warning(f"🔄 Avoiding redundant question: {question}")
            return True
        
        # Check if we're stuck in a loop
        if self.conversation_memory.is_stuck_in_loop():
            logger.warning("🔄 Detected question loop, changing strategy")
            return True
        
        return False

    def _get_most_relevant_url(self, query: str) -> str:
        """Get the most relevant URL for a query from mentioned URLs."""
        if not self.mentioned_urls:
            return ""
            
        # If we only have one URL, return it
        if len(self.mentioned_urls) == 1:
            return next(iter(self.mentioned_urls))
            
        # Check for fragment identifiers that match the query
        query_terms = set(query.lower().split())
        
        best_url = None
        best_score = 0
        
        for url in self.mentioned_urls:
            # Extract fragment
            fragment = url.split("#")[-1] if "#" in url else ""
            
            # Calculate relevance score
            score = 0
            
            # Check if fragment matches query terms
            if fragment:
                fragment_terms = set(fragment.lower().split("-"))
                matching_terms = query_terms.intersection(fragment_terms)
                score += len(matching_terms) * 2
                
            # Check if URL contains query terms
            for term in query_terms:
                if term in url.lower():
                    score += 1
                    
            # Update best URL if this one has a higher score
            if score > best_score:
                best_score = score
                best_url = url
                
        # Return the best URL or the first one if no match
        return best_url if best_url else next(iter(self.mentioned_urls))

    async def _suggest_better_tool(self, current_tool: str, args: Dict[str, Any]) -> Optional[ToolCall]:
        """Suggest a better tool to use instead of the current one."""
        # If asking redundant questions, suggest using browser instead
        if current_tool == "ask_human" and self._is_redundant_question(args.get("question", "")):
            # If we have URLs mentioned, suggest browser
            if self.mentioned_urls:
                url = next(iter(self.mentioned_urls))
                return ToolCall(
                    id="suggested_browser_call",
                    function={"name": "browser_use", "arguments": json.dumps({"url": url})}
                )
        
        # Use tool usage patterns to suggest alternatives
        suggested = self.tool_tracker.suggest_next_tool(current_tool)
        if suggested and suggested != current_tool:
            # Would need to implement proper argument mapping between tools
            return None
        
        return None

    async def think(self) -> bool:
        """Enhanced thinking process with improved reasoning."""
        # Check if we should complete the task
        if hasattr(self, 'task_completer') and self.task_completer.should_complete_task():
            deliverable = self.task_completer.create_deliverable()
            self.add_to_context(f"\n\nI've completed the requested task. Here's the deliverable:\n\n{deliverable}")
            return True
            
        # Check if we have URLs in the task but haven't used the browser yet
        if not self.browser_used and self.url_detector.mentioned_urls:
            url = next(iter(self.url_detector.mentioned_urls))
            self.add_to_context(f"The task mentions a URL ({url}). I should use the browser to visit it.")
            return True
            
        # Check if we've asked too many questions
        if hasattr(self, 'conversation_memory') and len(self.conversation_memory.asked_questions) > 5:
            self.add_to_context("I've asked several questions already. Let me try to complete the task with what I know.")
            # Force task completion
            if hasattr(self, 'task_completer'):
                deliverable = self.task_completer.create_deliverable()
                self.add_to_context(f"\n\nBased on the information I've gathered, here's the deliverable:\n\n{deliverable}")
            return True
            
        # Default to standard thinking
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True
        
        # Check if this is a new task and analyze it
        if self.memory.messages and len(self.memory.messages) <= 2:
            # This appears to be the start of a new task
            user_message = next((msg for msg in self.memory.messages if msg.role == "user"), None)
            if user_message and user_message.content:
                # Extract URLs from the user message
                urls = self._extract_urls(user_message.content)
                if urls:
                    self.mentioned_urls.update(urls)
                
                # Check if we should use browser based on message content
                if self._should_use_browser(user_message.content):
                    # Modify the prompt to encourage browser use
                    browser_hint = "\n\nIMPORTANT: The user's request mentions a website or URL. Use the browser_use tool to visit the mentioned website before asking questions. Make sure to check all relevant sections like 'about', 'features', etc.\n"
                    self.next_step_prompt += browser_hint
                
                await self.analyze_new_task(user_message.content)
                # Reset the context manager for the new task
                self.context_manager.reset_for_new_task()
                self.conversation_memory.reset()
                self.tool_tracker.reset()
                
                # Get the SmartAskHuman tool and reset it
                for tool in self.available_tools.tools:
                    if isinstance(tool, SmartAskHuman):
                        tool.reset_for_new_task()
                        break

        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        
        # Check if browser is already in use
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )
            
            # Add navigation suggestions if we have URLs
            if self.mentioned_urls:
                current_url = next(iter(self.mentioned_urls))
                next_action = self.browser_navigator.suggest_browser_action(current_url, self.current_task)
                if next_action and next_action["tool"] == "browser_use":
                    next_url = next_action["args"]["url"]
                    if self.browser_navigator.should_visit_url(next_url):
                        navigation_hint = f"\n\nIMPORTANT: Consider visiting {next_url} to gather more information for the task.\n"
                        self.next_step_prompt += navigation_hint
        
        # Add conversation context to the prompt
        if self.conversation_memory.user_responses:
            context_items = []
            for question, answer in self.conversation_memory.user_responses.items():
                if answer != "I already answered that" and answer != "continue" and answer != "exit":
                    context_items.append(f"Q: {question}\nA: {answer}")
            
            if context_items:
                context_prompt = "IMPORTANT - Previous conversation context:\n" + "\n".join(context_items) + "\n\nDO NOT ask these questions again.\n\n"
                self.next_step_prompt = context_prompt + self.next_step_prompt
        
        # Add context awareness to the prompt if we have context
        if self.context_manager.context:
            context_items = [f"- {k}: {v.value[:200]}..." for k, v in self.context_manager.context.items()]
            context_prompt = "Current context:\n" + "\n".join(context_items) + "\n\n" + self.next_step_prompt
            self.next_step_prompt = context_prompt
            
        # Check if we need to remind about task completion
        if "business plan" in self.current_task.lower() and not self.task_completed:
            # Count how many steps we've taken
            step_count = len(self.memory.messages) // 2
            
            if step_count > 3 and self.context_manager.context:
                # We have some context and have taken several steps, remind to complete the task
                task_reminder = "\n\nIMPORTANT: Remember to complete the original task of creating a business plan. Don't just gather information - synthesize it into a comprehensive plan with sections like Executive Summary, Market Analysis, Product/Service Description, Marketing Strategy, Financial Projections, etc.\n"
                self.next_step_prompt += task_reminder

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result
