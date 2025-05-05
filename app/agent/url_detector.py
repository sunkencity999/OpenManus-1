"""
URL detector middleware for Manus agent.
This module helps detect URLs in user requests and encourages using the browser tool.
"""
import re
from typing import List, Optional, Dict, Any
import json

from app.schema import ToolCall
from app.logger import logger


class URLDetector:
    """Detects URLs in text and suggests using browser tools instead of asking questions."""
    
    def __init__(self):
        self.mentioned_urls = set()
        
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return re.findall(url_pattern, text)
    
    def should_use_browser(self, text: str) -> bool:
        """Determine if browser should be used based on text content."""
        # Check for explicit URL mentions
        urls = self.extract_urls(text)
        if urls:
            self.mentioned_urls.update(urls)
            return True
            
        # Check for phrases suggesting web browsing - be more aggressive
        browse_phrases = [
            "check the website", "visit the site", "go to", "browse to", 
            "look at the page", "check the page", "about page", "website",
            "check", "visit", "browse", "page", "site", "web", "url", "link",
            "http", "https", "www", ".com", ".net", ".org", ".io"
        ]
        
        # Check for domain names in the text
        domain_pattern = r'\b[a-zA-Z0-9-]+\.(com|net|org|io|app|co)\b'
        if re.search(domain_pattern, text):
            return True
            
        return any(phrase in text.lower() for phrase in browse_phrases)
    
    def intercept_tool_call(self, tool_call: ToolCall) -> Optional[ToolCall]:
        """
        Intercept tool calls and suggest browser use when appropriate.
        
        Args:
            tool_call: The original tool call
            
        Returns:
            Modified tool call or None if no change needed
        """
        if not tool_call or not tool_call.function or not tool_call.function.name:
            return None
            
        name = tool_call.function.name
        
        # Only intercept ask_human calls
        if name != "ask_human":
            return None
            
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            question = args.get("inquire", "")
            
            # ALWAYS intercept the first question if we have URLs
            if self.mentioned_urls:
                url = next(iter(self.mentioned_urls))
                logger.warning(f"🔄 Intercepting first question. Using browser instead for: {url}")
                
                # Create a browser_use tool call instead with the required 'action' parameter
                return ToolCall(
                    id="intercepted_browser_call",
                    function={"name": "browser_use", "arguments": json.dumps({"action": "go_to_url", "url": url})}
                )
                
            # Check if the question is about a website or mentions URLs
            if self.should_use_browser(question):
                # Extract URLs from the question
                urls = self.extract_urls(question)
                if urls:
                    url = urls[0]
                    self.mentioned_urls.add(url)
                    logger.warning(f"🔄 Found URL in question. Using browser instead for: {url}")
                    
                    # Create a browser_use tool call instead with the required 'action' parameter
                    return ToolCall(
                        id="intercepted_browser_call",
                        function={"name": "browser_use", "arguments": json.dumps({"action": "go_to_url", "url": url})}
                    )
                    
                # If no URL in question but we have a domain hint, try to construct a URL
                domain_pattern = r'\b([a-zA-Z0-9-]+\.(com|net|org|io|app|co))\b'
                match = re.search(domain_pattern, question)
                if match:
                    domain = match.group(1)
                    url = f"https://{domain}"
                    self.mentioned_urls.add(url)
                    logger.warning(f"🔄 Found domain in question. Using browser instead for: {url}")
                    
                    # Create a browser_use tool call instead with the required 'action' parameter
                    return ToolCall(
                        id="intercepted_browser_call",
                        function={"name": "browser_use", "arguments": json.dumps({"action": "go_to_url", "url": url})}
                    )
        except Exception as e:
            logger.error(f"Error in URL detector: {e}")
            
        return None
