import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "extract_analyze_save",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' or 'extract_analyze_save' actions",
            },
            "filename": {
                "type": "string",
                "description": "Filename to save the analysis to for 'extract_analyze_save' action",
            },
            "format": {
                "type": "string",
                "enum": ["markdown", "json", "html", "text"],
                "description": "Format of the output file for 'extract_analyze_save' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
            "extract_analyze_save": ["goal", "filename"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(self, *args, **kwargs) -> ToolResult:
        """
        Wrapper for browser tool execution. Accepts either (action, ...) or (args_dict), and defaults to go_to_url if only a url is provided.
        """
        # Handle dict-style call
        if args and isinstance(args[0], dict):
            params = args[0]
            # If action is missing but url is present, default to go_to_url
            if 'action' not in params and 'url' in params:
                params['action'] = 'go_to_url'
            if 'action' not in params:
                return ToolResult(error="Missing required parameter: 'action' (or 'url' for navigation)")
            return await self._execute_internal(**params)
        # Handle keyword-style call
        if 'action' not in kwargs:
            if 'url' in kwargs:
                kwargs['action'] = 'go_to_url'
            else:
                return ToolResult(error="Missing required parameter: 'action' (or 'url' for navigation)")
        return await self._execute_internal(**kwargs)

    async def _execute_internal(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()

                # Get max content length from config
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 2000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    page = await context.get_current_page()
                    await page.goto(url)
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Navigated to {url}")

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    page = await context.get_current_page()
                    import markdownify

                    content = markdownify.markdownify(await page.content())

                    prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}

Page content:
{content[:max_content_length]}
"""
                    messages = [{"role": "system", "content": prompt}]

                    # Define extraction function schema
                    extraction_function = {
                        "type": "function",
                        "function": {
                            "name": "extract_content",
                            "description": "Extract specific information from a webpage based on a goal",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "extracted_content": {
                                        "type": "object",
                                        "description": "The content extracted from the page according to the goal",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content extracted from the page",
                                            },
                                            "metadata": {
                                                "type": "object",
                                                "description": "Additional metadata about the extracted content",
                                                "properties": {
                                                    "source": {
                                                        "type": "string",
                                                        "description": "Source of the extracted content",
                                                    }
                                                },
                                            },
                                        },
                                    }
                                },
                                "required": ["extracted_content"],
                            },
                        },
                    }

                    # Use LLM to extract content with required function calling
                    response = await self.llm.ask_tool(
                        messages,
                        tools=[extraction_function],
                        tool_choice="required",
                    )

                    if response and response.tool_calls:
                        args = json.loads(response.tool_calls[0].function.arguments)
                        extracted_content = args.get("extracted_content", {})
                        return ToolResult(
                            output=f"Extracted from page:\n{extracted_content}\n"
                        )

                    return ToolResult(output="No content was extracted from the page.")
                    
                elif action == "extract_analyze_save":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_analyze_save' action"
                        )
                    
                    # Get the filename from kwargs or generate a default one
                    filename = kwargs.get("filename")
                    if not filename:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"analysis_{timestamp}.md"
                    
                    # Get the format from kwargs or default to markdown
                    format = kwargs.get("format", "markdown")
                    
                    # Call the extract_analyze_save method
                    result_path = await self.extract_analyze_save(context, goal, filename, format)
                    
                    if result_path:
                        return ToolResult(output=f"Analysis saved to {result_path}")
                    else:
                        return ToolResult(error="Failed to extract, analyze, and save content")

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    async def extract_analyze_save(self, context, goal, filename, format="markdown"):
        """
        Extract content from a webpage, analyze it using Ollama, and save the result to a file.
        
        Args:
            context: The browser context
            goal: The goal of the extraction (what to analyze)
            filename: The filename to save the analysis to
            format: The format of the output file (markdown, json, html, text)
            
        Returns:
            The path to the saved file or None if extraction failed
        """
        import logging
        import os
        import time
        import json
        from datetime import datetime
        
        logger = logging.getLogger("browser_use_tool")
        logger.info(f"Extracting and analyzing content with goal: {goal}")
        
        try:
            # Get the current page
            browser_context = await self._ensure_browser_initialized()
            page = await browser_context.get_current_page()
            
            if not page:
                logger.error("No active page found")
                return None
            
            # Extract raw content
            logger.info("Extracting raw content from page")
            raw_html = await page.content()
            page_url = page.url
            page_title = await page.title()
            
            logger.info(f"Page title: {page_title}")
            logger.info(f"Page URL: {page_url}")
            logger.info(f"HTML content length: {len(raw_html)}")
            
            # Create workspace directory structure
            workspace_dir = os.path.join(os.getcwd(), "workspace")
            os.makedirs(workspace_dir, exist_ok=True)
            
            # Create a raw content directory
            raw_content_dir = os.path.join(workspace_dir, "raw_content")
            os.makedirs(raw_content_dir, exist_ok=True)
            
            # Generate a timestamp for the raw content file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a directory for this specific extraction
            extraction_dir = os.path.join(raw_content_dir, f"extraction_{timestamp}")
            os.makedirs(extraction_dir, exist_ok=True)
            
            # Save the raw HTML
            raw_html_path = os.path.join(extraction_dir, "raw.html")
            with open(raw_html_path, 'w', encoding='utf-8') as f:
                f.write(raw_html)
            logger.info(f"Saved raw HTML content to {raw_html_path}")
            
            # Convert HTML to markdown for better readability
            try:
                import markdownify
                content = markdownify.markdownify(raw_html)
            except ImportError:
                logger.warning("markdownify not installed, using raw HTML")
                content = raw_html
            
            # Save the raw markdown
            raw_md_path = os.path.join(extraction_dir, "raw.md")
            with open(raw_md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved raw markdown content to {raw_md_path}")
            
            # Save metadata about the extraction
            metadata = {
                "url": page_url,
                "title": page_title,
                "timestamp": timestamp,
                "goal": goal,
                "extraction_time": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(extraction_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved extraction metadata to {metadata_path}")
            
            # Analyze the content using Ollama
            logger.info("Analyzing content using Ollama")
            
            # Get the maximum content length to analyze
            max_content_length = 10000  # Reasonable default
            
            # Create a prompt for analysis
            analysis_prompt = f"""
            You are an expert content analyzer. Your task is to extract and summarize information from the following webpage content.
            
            GOAL: {goal}
            
            PAGE TITLE: {page_title}
            PAGE URL: {page_url}
            
            Please provide a detailed analysis focusing specifically on this goal. Include all relevant information, facts, figures, and quotes.
            Structure your response with appropriate headings and sections. Be comprehensive but concise.
            
            CONTENT:
            {content[:max_content_length] if len(content) > max_content_length else content}
            
            {"[Content truncated due to length]" if len(content) > max_content_length else ""}
            """
            
            # Get LLM config
            if not self.llm:
                logger.error("LLM not initialized")
                return None
                
            model = self.llm.model
            api_type = getattr(self.llm, 'api_type', 'ollama')
            base_url = self.llm.base_url
            
            # Remove /v1 suffix for Ollama as per memory
            original_base_url = base_url
            if base_url and base_url.endswith("/v1"):
                base_url = base_url[:-3]
                self.llm.base_url = base_url  # Update the base_url in the LLM object
                logger.info(f"Removed /v1 suffix from base_url: {original_base_url} -> {base_url}")
            
            # Get a response from Ollama
            logger.info(f"Sending request to Ollama with model: {model}")
            
            # Try to get a response from Ollama
            analysis_response = None
            
            try:
                # Import the ollama_integration module if available
                try:
                    import ollama_integration
                    logger.info("Using ollama_integration module")
                    analysis_response = await ollama_integration.get_ollama_response(self.llm, analysis_prompt)
                except ImportError:
                    logger.warning("ollama_integration module not found, using direct API calls")
                    
                    # If ollama_integration is not available, use direct API calls
                    import aiohttp
                    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        # Prepare the request payload for the chat API
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": analysis_prompt}],
                            "stream": False
                        }
                        
                        logger.info(f"Sending request to: {base_url}/api/chat")
                        start_time = datetime.now()
                        logger.info("Waiting for Ollama to generate response (this may take a while)...")
                        
                        async with session.post(f"{base_url}/api/chat", json=payload) as response:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            logger.info(f"Received HTTP status: {response.status} after {elapsed:.2f} seconds")
                            
                            if response.status == 200:
                                result = await response.json()
                                logger.info(f"Received response from Ollama chat API with keys: {result.keys() if result else 'None'}")
                                
                                if "message" in result and "content" in result["message"]:
                                    analysis_response = result["message"]["content"]
                                    logger.info(f"Successfully extracted content from Ollama chat response, length: {len(analysis_response)}")
                                else:
                                    logger.warning(f"Unexpected response structure from Ollama chat API: {result.keys() if result else 'None'}")
                                    
                                    # Try fallback to generate API
                                    logger.info("Falling back to Ollama generate API")
                                    payload = {
                                        "model": model,
                                        "prompt": analysis_prompt,
                                        "stream": False
                                    }
                                    
                                    async with session.post(f"{base_url}/api/generate", json=payload) as gen_response:
                                        if gen_response.status == 200:
                                            gen_result = await gen_response.json()
                                            if "response" in gen_result:
                                                analysis_response = gen_result["response"]
                                                logger.info(f"Successfully extracted content from Ollama generate response, length: {len(analysis_response)}")
                            else:
                                logger.error(f"Error from Ollama API: {response.status}")
            
            except Exception as e:
                logger.error(f"Error during LLM analysis: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Check if we got a response
            if not analysis_response:
                logger.error("Failed to get a response from Ollama")
                return None
            
            logger.info(f"Analysis response length: {len(analysis_response)}")
            
            # Save the raw analysis
            analysis_path = os.path.join(extraction_dir, "analysis.md")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write(analysis_response)
            logger.info(f"Saved raw analysis to {analysis_path}")
            
            # Format the analysis based on the requested format
            logger.info(f"Formatting analysis in {format} format...")
            
            # Ensure filename has the correct extension based on format
            file_extension = {
                "json": ".json",
                "html": ".html",
                "text": ".txt",
                "markdown": ".md"
            }.get(format, ".md")
            
            if not filename.endswith(file_extension):
                filename = filename + file_extension
            
            # Format the content based on the requested format
            if format == "json":
                # Create a JSON structure with the analysis and metadata
                formatted_data = {
                    "goal": goal,
                    "url": page_url,
                    "title": page_title,
                    "analysis": analysis_response,
                    "metadata": {
                        "extraction_time": datetime.now().isoformat(),
                        "source": page_url,
                        "raw_content_path": extraction_dir
                    }
                }
                formatted_content = json.dumps(formatted_data, indent=2)
                
            elif format == "html":
                # Create a nicely formatted HTML document
                formatted_content = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>{goal} - {page_title}</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 800px; margin: 0 auto; }}
                        h1 {{ color: #333; }}
                        .content {{ margin: 20px 0; }}
                        .source {{ color: #666; font-style: italic; }}
                        .extraction-info {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8em; color: #999; }}
                    </style>
                </head>
                <body>
                    <h1>{goal}</h1>
                    <h2>{page_title}</h2>
                    <div class="content">
                        {analysis_response}
                    </div>
                    <p class="source">Source: <a href="{page_url}">{page_url}</a></p>
                    <div class="extraction-info">
                        Extraction time: {datetime.now().isoformat()}<br>
                        Raw content available at: {extraction_dir}
                    </div>
                </body>
                </html>"""
                
            elif format == "text":
                # Create a simple text format
                formatted_content = f"""GOAL: {goal}

TITLE: {page_title}

SOURCE: {page_url}

{analysis_response}

Extraction time: {datetime.now().isoformat()}
Raw content available at: {extraction_dir}
"""
                
            else:  # Default to markdown
                # Create a nicely formatted markdown document
                formatted_content = f"""# {goal}

## {page_title}

{analysis_response}

---

*Source: [{page_url}]({page_url})*

*Extraction time: {datetime.now().isoformat()}*

*Raw content available at: `{extraction_dir}`*
"""
            
            # Write the formatted content to the output file
            output_path = os.path.join(workspace_dir, filename)
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the content to the file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)
            
            logger.info(f"Saved formatted content to {output_path}")
            
            # Return the path to the saved file
            return output_path
        
        except Exception as e:
            logger.error(f"Error during extraction and analysis: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
