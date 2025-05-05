"""
Browser navigation helper for OpenManus agents.
Helps agents navigate websites more effectively and extract relevant information.
"""
from typing import List, Dict, Any, Optional, Set
import re
import logging
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)


class BrowserNavigator(BaseModel):
    """Helper class to assist agents in navigating websites."""
    
    # Track visited URLs to avoid loops
    visited_urls: Set[str] = Field(default_factory=set)
    
    # Track extracted content to avoid redundancy
    extracted_content: Dict[str, str] = Field(default_factory=dict)
    
    # Track navigation attempts to avoid repeated failures
    navigation_attempts: Dict[str, int] = Field(default_factory=dict)
    
    # Track important elements for revisit decisions
    important_elements: Dict[str, bool] = Field(default_factory=dict)
    
    # Track content by URL for efficient lookup
    content_by_url: Dict[str, str] = Field(default_factory=dict)
        
    def handle_url_navigation(self, url: str) -> Dict[str, Any]:
        """Handle navigation to a URL and extract relevant information."""
        # Add the URL to visited URLs
        self.visited_urls.add(url)
        
        # Check if URL has a fragment (e.g., #features)
        base_url, fragment = url.split('#', 1) if '#' in url else (url, None)
        
        # Log the navigation with fragment information
        if fragment:
            logger.warning(f"🧭 Navigating to URL with fragment: {url} (section: #{fragment})")
        
        # Prepare the browser_use tool call
        return {
            "action": "go_to_url",
            "url": url  # Keep the full URL with fragment
        }
        
    def should_visit_url(self, url: str) -> bool:
        """Determine if a URL should be visited."""
        # Don't revisit URLs unless they're important
        if url in self.visited_urls and url not in self.important_elements:
            return False
            
        # Check if URL contains hash fragment for navigation
        if "#" in url and url.split("#")[0] in self.visited_urls:
            # Only revisit if it's a new fragment
            base_url = url.split("#")[0]
            for visited in self.visited_urls:
                if visited.startswith(base_url + "#"):
                    fragment1 = visited.split("#")[1] if "#" in visited else ""
                    fragment2 = url.split("#")[1] if "#" in url else ""
                    if fragment1 == fragment2:
                        return False
                        
        return True
        
    def extract_navigation_elements(self, content: str) -> List[Dict[str, str]]:
        """
        Extract navigation elements from page content.
        Returns a list of dictionaries with element text and target URLs.
        """
        # Simple regex-based extraction (in a real implementation, use proper HTML parsing)
        nav_elements = []
        
        # Look for navigation links
        nav_pattern = r'<nav[^>]*>(.*?)</nav>'
        nav_matches = re.findall(nav_pattern, content, re.DOTALL)
        
        for nav_match in nav_matches:
            # Extract links within navigation
            link_pattern = r'<a[^>]*href=["\'](.*?)["\'][^>]*>(.*?)</a>'
            links = re.findall(link_pattern, nav_match, re.DOTALL)
            
            for href, text in links:
                # Clean up text
                clean_text = re.sub(r'<[^>]*>', '', text).strip()
                if clean_text and href:
                    nav_elements.append({
                        "text": clean_text,
                        "url": href
                    })
                    
        # Also look for important standalone links
        important_keywords = ["about", "features", "pricing", "contact", "faq", "help"]
        link_pattern = r'<a[^>]*href=["\'](.*?)["\'][^>]*>(.*?)</a>'
        links = re.findall(link_pattern, content, re.DOTALL)
        
        for href, text in links:
            clean_text = re.sub(r'<[^>]*>', '', text).strip()
            if any(keyword in clean_text.lower() for keyword in important_keywords):
                if clean_text and href:
                    nav_elements.append({
                        "text": clean_text,
                        "url": href
                    })
                    
        return nav_elements
        
    def suggest_next_action(self, current_url: str, task_description: str) -> Optional[Dict[str, Any]]:
        """Suggest the next action based on the current URL and task."""
        # Add current URL to visited
        self.visited_urls.add(current_url)
        
        # Check if URL has a fragment (e.g., #features)
        base_url, fragment = current_url.split('#', 1) if '#' in current_url else (current_url, None)
        
        # If we have a fragment in the task but not in the current URL, navigate to it
        if '#' in task_description and not fragment:
            # Extract fragment from task description
            for word in task_description.split():
                if word.startswith('http') and '#' in word:
                    _, fragment = word.split('#', 1)
                    target_url = f"{current_url}#{fragment}"
                    logger.warning(f"🧭 Found fragment in task description, navigating to: {target_url}")
                    return self.handle_url_navigation(target_url)
        
        # Extract navigation elements
        navigation_elements = self.extract_navigation_elements(current_url)
        
        # If we're on the homepage, suggest checking about or features pages
        if self._is_homepage(current_url):
            # Look for specific sections mentioned in the task
            sections = ["about", "features", "pricing", "contact", "faq", "help"]
            for section in sections:
                if section in task_description.lower():
                    # First check for links
                    matching_links = [link for link in navigation_elements if section in link["text"].lower()]
                    if matching_links:
                        return self.handle_url_navigation(matching_links[0]["url"])
                    # Then try fragment navigation
                    target_url = f"{current_url}#{section}"
                    logger.warning(f"🧭 Section '{section}' mentioned in task, trying fragment navigation: {target_url}")
                    return self.handle_url_navigation(target_url)
        
        # Default: no suggestion
        return None
        
    def extract_key_information(self, content: str, keywords: List[str]) -> Dict[str, str]:
        """
        Extract key information from page content based on keywords.
        
        Args:
            content: HTML content of the page
            keywords: List of keywords to look for
            
        Returns:
            Dictionary of extracted information
        """
        # Remove HTML tags for text analysis
        text_content = re.sub(r'<[^>]*>', ' ', content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Extract paragraphs that might contain relevant information
        paragraphs = re.split(r'\n\s*\n', text_content)
        
        extracted_info = {}
        
        for keyword in keywords:
            relevant_paragraphs = []
            
            for paragraph in paragraphs:
                if keyword.lower() in paragraph.lower():
                    relevant_paragraphs.append(paragraph)
                    
            if relevant_paragraphs:
                extracted_info[keyword] = "\n".join(relevant_paragraphs)
                
        return extracted_info
        
    def extract_content_for_task(self, content: str, task_description: str) -> str:
        """Extract relevant content from a page based on the task description."""
        # Identify key sections based on task
        key_sections = []
        
        # Common business plan/marketing plan sections
        if "business plan" in task_description.lower() or "marketing plan" in task_description.lower():
            key_sections = ["features", "about", "pricing", "overview", "benefits", "target audience", "market"]
        
        # Extract content from these sections if they exist
        extracted_content = []
        for section in key_sections:
            # Look for section headers or divs with this ID/class
            section_pattern = f"<h[1-6][^>]*>{section}[^<]*</h[1-6]>|<div[^>]*id=['\"]{0,1}{section}['\"]{0,1}[^>]*>|<section[^>]*id=['\"]{0,1}{section}['\"]{0,1}[^>]*>"
            match = re.search(section_pattern, content, re.IGNORECASE)
            if match:
                # Extract a reasonable chunk of content after this section header
                start_idx = match.start()
                end_idx = min(start_idx + 2000, len(content))  # Get about 2000 chars of content
                section_content = content[start_idx:end_idx]
                extracted_content.append(f"--- {section.upper()} SECTION ---\n{section_content}\n")
        
        # If we found specific sections, return those
        if extracted_content:
            return "\n\n".join(extracted_content)
            
        # Otherwise return a summary of the content
        return "Content summary: " + content[:1000] + "...\n(Content truncated for brevity)"
        
    def create_navigation_plan(self, base_url: str, task_description: str) -> List[str]:
        """
        Create a plan for navigating a website based on the task.
        
        Args:
            base_url: The base URL of the website
            task_description: Description of the task
            
        Returns:
            List of URLs to visit in order
        """
        # Extract domain from base_url
        domain = base_url.split("//")[-1].split("/")[0]
        
        # Create a basic navigation plan
        plan = [base_url]  # Start with the homepage
        
        # Add common important pages
        common_pages = []
        
        # Check if task mentions specific sections
        task_lower = task_description.lower()
        
        if "about" in task_lower:
            common_pages.append(f"{base_url}/about")
            common_pages.append(f"{base_url}/#about")
            
        if "feature" in task_lower:
            common_pages.append(f"{base_url}/features")
            common_pages.append(f"{base_url}/#features")
            
        if "price" in task_lower or "cost" in task_lower:
            common_pages.append(f"{base_url}/pricing")
            common_pages.append(f"{base_url}/#pricing")
            
        # Add the common pages to the plan
        for page in common_pages:
            if page not in plan:
                plan.append(page)
                
        return plan
        
    def suggest_browser_action(self, current_url: str, task_description: str) -> Optional[Dict[str, Any]]:
        """
        Suggest the next browser action based on the current state and task.
        
        Args:
            current_url: The current URL
            task_description: Description of the current task
            
        Returns:
            Dictionary with suggested action parameters, or None if no suggestion
        """
        # If we haven't visited the current URL yet, suggest visiting it
        if current_url not in self.visited_urls:
            return {
                "tool": "browser_use",
                "args": {"url": current_url}
            }
            
        # If we have a navigation suggestion, use it
        next_url = self.suggest_next_navigation(current_url, task_description)
        if next_url:
            # Make the URL absolute if it's relative
            if not next_url.startswith("http"):
                if next_url.startswith("/"):
                    # Absolute path
                    base_url = "/".join(current_url.split("/")[:3])  # http(s)://domain.com
                    next_url = base_url + next_url
                else:
                    # Relative path
                    current_path = "/".join(current_url.split("/")[:-1])
                    next_url = f"{current_path}/{next_url}"
                    
            return {
                "tool": "browser_use",
                "args": {"url": next_url}
            }
            
        # If we've tried navigation and still need information, suggest searching
        return None
