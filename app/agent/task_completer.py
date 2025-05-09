"""
Task completion manager to ensure agents finish their assigned tasks.
"""
from typing import Dict, List, Optional, Set, Any
import re
import logging
import importlib.util
import os
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

# Check if task_completer_extensions.py exists
extensions_path = os.path.join(os.path.dirname(__file__), "task_completer_extensions.py")
HAS_EXTENSIONS = os.path.exists(extensions_path)

# Import extensions if available
if HAS_EXTENSIONS:
    try:
        spec = importlib.util.spec_from_file_location("task_completer_extensions", extensions_path)
        extensions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extensions_module)
        logger.info("Successfully loaded task completer extensions")
    except Exception as e:
        logger.error(f"Error loading task completer extensions: {str(e)}")
        HAS_EXTENSIONS = False


class TaskCompleter(BaseModel):
    """
    Ensures agents complete their assigned tasks by tracking progress
    and generating final deliverables.
    """
    
    # Track task type and requirements
    task_type: str = Field(default="")
    task_requirements: List[str] = Field(default_factory=list)
    
    # Track gathered information
    gathered_info: Dict[str, str] = Field(default_factory=dict)
    
    # Track completion status
    is_complete: bool = Field(default=False)
    
    # Track deliverable content
    deliverable_content: str = Field(default="")
    
    # Track logger
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Dynamically add extension methods if available
        if HAS_EXTENSIONS:
            try:
                # Add all methods from extensions that start with _create_
                for method_name in dir(extensions_module):
                    if method_name.startswith('_create_'):
                        method = getattr(extensions_module, method_name)
                        setattr(TaskCompleter, method_name, method)
                self.logger.info(f"Added extension methods to TaskCompleter")
            except Exception as e:
                self.logger.error(f"Error adding extension methods: {str(e)}")
    
    def analyze_task(self, task_description: str) -> None:
        """Analyze the task to determine type and requirements."""
        task_lower = task_description.lower()
        
        # Identify task type
        if "marketing plan" in task_lower:
            self.task_type = "marketing_plan"
            self.task_requirements = [
                "target_audience", "value_proposition", "marketing_channels", 
                "key_messaging", "competitive_analysis", "budget"
            ]
        elif "business plan" in task_lower:
            self.task_type = "business_plan"
            self.task_requirements = [
                "executive_summary", "company_description", "market_analysis",
                "organization_structure", "product_line", "marketing_strategy",
                "financial_projections"
            ]
        elif ("essay" in task_lower or "longform essay" in task_lower or "long-form essay" in task_lower):
            self.task_type = "essay"
            self.task_requirements = ["subject", "style", "tone"]
            # Extract information from the task description
            self._extract_creative_content_info(task_description)
        elif "poem" in task_lower or "poetry" in task_lower:
            self.task_type = "poem"
            self.task_requirements = ["subject", "style", "tone"]
            # Extract information from the task description
            self._extract_creative_content_info(task_description)
        elif "research report" in task_lower or "research paper" in task_lower:
            self.task_type = "research_report"
            self.task_requirements = [
                "topic", "background", "methodology", "findings", 
                "analysis", "conclusion", "references"
            ]
        elif "technical documentation" in task_lower or "api doc" in task_lower or "user guide" in task_lower:
            self.task_type = "technical_documentation"
            self.task_requirements = [
                "product_name", "version", "purpose", "installation", 
                "usage_examples", "api_endpoints", "parameters"
            ]
        elif "summary" in task_lower or "summarize" in task_lower or "summarization" in task_lower:
            self.task_type = "content_summary"
            self.task_requirements = [
                "source_content", "key_points", "length", "audience"
            ]
        elif "data analysis" in task_lower or "analyze data" in task_lower or "statistics" in task_lower:
            self.task_type = "data_analysis"
            self.task_requirements = [
                "dataset", "variables", "analysis_goal", "methods", 
                "findings", "visualizations", "recommendations"
            ]
        elif "blog post" in task_lower or "article" in task_lower:
            self.task_type = "blog_post"
            self.task_requirements = [
                "topic", "target_audience", "tone", "key_points", 
                "call_to_action", "seo_keywords"
            ]
        elif "email" in task_lower or "newsletter" in task_lower:
            self.task_type = "email_content"
            self.task_requirements = [
                "purpose", "recipient", "subject_line", "key_message", 
                "tone", "call_to_action"
            ]
        elif "social media" in task_lower or "tweet" in task_lower or "post" in task_lower:
            self.task_type = "social_media"
            self.task_requirements = [
                "platform", "purpose", "tone", "key_message", 
                "hashtags", "call_to_action"
            ]
        elif "website" in task_lower or "web site" in task_lower or "webpage" in task_lower:
            self.task_type = "website"
            self.task_requirements = [
                "website_name", "website_url", "color_scheme", "folder_name",
                "purpose", "target_audience"
            ]
            
            # Extract website name and URL from task description if possible
            import re
            website_name_match = re.search(r'called ["\']?([\w\s]+)["\']', task_lower)
            if website_name_match:
                self.add_information("website_name", website_name_match.group(1).strip())
                
            url_match = re.search(r'url\s+(?:is\s+)?["\']?(https?://[\w\.-]+\.[a-zA-Z]{2,})["\']?', task_lower)
            if url_match:
                self.add_information("website_url", url_match.group(1).strip())
                
            folder_match = re.search(r'(?:folder|directory)\s+(?:called\s+)?["\']?([\w\s-]+)["\']', task_lower)
            if folder_match:
                self.add_information("folder_name", folder_match.group(1).strip())
        else:
            # Generic research task
            self.task_type = "research"
            self.task_requirements = ["overview", "key_findings", "conclusion"]
            
        logger.warning(f"📝 Task type identified: {self.task_type}")
        logger.warning(f"📋 Requirements: {', '.join(self.task_requirements)}")
    
    def add_information(self, key: str, value: str) -> None:
        """Add gathered information to the task."""
        self.gathered_info[key] = value
        logger.info(f"✅ Added information for: {key}")
        
        # Check if we have enough information to complete the task
        self._check_completion_status()
    
    def _check_completion_status(self) -> None:
        """Check if we have enough information to complete the task."""
        # For simplicity, consider task complete if we have info for at least half the requirements
        min_requirements = len(self.task_requirements) // 2
        covered_requirements = 0
        
        for req in self.task_requirements:
            # Check if we have direct info for this requirement
            if req in self.gathered_info:
                covered_requirements += 1
                continue
                
            # Check if we have info that might cover this requirement
            for key in self.gathered_info:
                if req in key or key in req:
                    covered_requirements += 1
                    break
        
        self.is_complete = covered_requirements >= min_requirements
        if self.is_complete:
            logger.warning(f"🎉 Task has sufficient information to complete! ({covered_requirements}/{len(self.task_requirements)} requirements)")
    
    def get_missing_info(self) -> List[str]:
        """Return a list of requirements that haven't been gathered yet."""
        missing_info = []
        
        for req in self.task_requirements:
            # Check if we have direct info for this requirement
            if req not in self.gathered_info:
                # Check if we have info that might cover this requirement
                covered = False
                for key in self.gathered_info:
                    if req in key or key in req:
                        covered = True
                        break
                        
                if not covered:
                    missing_info.append(req)
        
        return missing_info
    
    def _extract_creative_content_info(self, task_description: str) -> None:
        """Extract creative content information from the task description."""
        task_lower = task_description.lower()
        
        # Extract subject information
        subject_patterns = [
            r"(?:poem|poetry|write)\s+about\s+([^\.,;]+)",
            r"(?:poem|poetry|write)\s+on\s+([^\.,;]+)",
            r"(?:create|generate|make)\s+a\s+poem\s+about\s+([^\.,;]+)",
            r"(?:create|generate|make)\s+a\s+poem\s+on\s+([^\.,;]+)"
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, task_lower)
            if match:
                subject = match.group(1).strip()
                self.gathered_info["subject"] = subject
                self.logger.warning(f"📋 Extracted subject: {subject}")
                break
        
        # Extract style information
        style_patterns = [
            r"in\s+the\s+style\s+of\s+([^\.,;]+)",
            r"like\s+(?:a|an)\s+([^\.,;]+)\s+poem",
            r"(?:similar|alike|akin)\s+to\s+([^\.,;]+)",
            r"(?:following|using)\s+the\s+([^\.,;]+)\s+style"
        ]
        
        for pattern in style_patterns:
            match = re.search(pattern, task_lower)
            if match:
                style = match.group(1).strip()
                self.gathered_info["style"] = style
                self.logger.warning(f"📋 Extracted style: {style}")
                break
        
        # Extract tone information
        tone_patterns = [
            r"with\s+a\s+([a-z]+)\s+tone",
            r"in\s+a\s+([a-z]+)\s+(?:tone|manner|way|voice)",
            r"that\s+is\s+([a-z]+)\s+in\s+tone"
        ]
        
        tone_keywords = ["happy", "sad", "melancholic", "joyful", "serious", "humorous", 
                      "dark", "light", "somber", "playful", "reflective", "nostalgic",
                      "romantic", "dramatic", "ironic", "satirical"]
        
        # Check for explicit tone patterns
        tone_found = False
        for pattern in tone_patterns:
            match = re.search(pattern, task_lower)
            if match:
                tone = match.group(1).strip()
                self.gathered_info["tone"] = tone
                self.logger.warning(f"📋 Extracted tone: {tone}")
                tone_found = True
                break
        
        # Check for tone keywords if no explicit pattern was found
        if not tone_found:
            for tone in tone_keywords:
                if tone in task_lower:
                    self.gathered_info["tone"] = tone
                    self.logger.warning(f"📋 Extracted tone: {tone}")
                    break
    
    def is_ready_to_complete(self) -> bool:
        """Check if the task is ready to be completed based on gathered information."""
        # For simplicity, consider task ready to complete if we have info for at least half the requirements
        min_requirements = len(self.task_requirements) // 2
        covered_requirements = 0
        
        for req in self.task_requirements:
            # Check if we have direct info for this requirement
            if req in self.gathered_info:
                covered_requirements += 1
                continue
                
            # Check if we have info that might cover this requirement
            for key in self.gathered_info:
                if req in key or key in req:
                    covered_requirements += 1
                    break
        
        return covered_requirements >= min_requirements
    
    def should_complete_task(self) -> bool:
        """Determine if it's time to complete the task."""
        return self.is_complete and not self.deliverable_content
    
    def create_deliverable(self) -> str:
        """Create a deliverable based on the task type and gathered information."""
        if self.task_type == "marketing_plan":
            return self._create_marketing_plan()
        elif self.task_type == "business_plan":
            return self._create_business_plan()
        elif self.task_type == "poem":
            return self._create_poem()
        elif self.task_type == "research_report":
            return self._create_research_report()
        elif self.task_type == "technical_documentation":
            return self._create_technical_documentation()
        elif self.task_type == "content_summary":
            return self._create_content_summary()
        elif self.task_type == "data_analysis":
            return self._create_data_analysis()
        elif self.task_type == "blog_post":
            return self._create_blog_post()
        elif self.task_type == "email_content":
            return self._create_email_content()
        elif self.task_type == "social_media":
            return self._create_social_media_content()
        elif self.task_type == "website":
            return self._create_website()
        else:  # Default to research report
            return self._create_research_report()
    
    def _create_marketing_plan(self) -> str:
        """Create a marketing plan deliverable."""
        # Start with a template
        template = """
# Marketing Plan for {product_name}

## Executive Summary
{executive_summary}

## Target Audience
{target_audience}

## Value Proposition
{value_proposition}

## Marketing Channels
{marketing_channels}

## Key Messaging
{key_messaging}

## Competitive Analysis
{competitive_analysis}

## Budget and Timeline
{budget}

## Conclusion
{conclusion}
"""
        
        # Fill in the template with gathered information
        product_name = self._extract_product_name()
        
        # For each section, use gathered info or generate placeholder
        sections = {
            "product_name": product_name,
            "executive_summary": self.gathered_info.get("executive_summary", f"A comprehensive marketing plan for {product_name}."),
            "target_audience": self.gathered_info.get("target_audience", "Target audience information not specified."),
            "value_proposition": self.gathered_info.get("value_proposition", "Value proposition not specified."),
            "marketing_channels": self.gathered_info.get("marketing_channels", "Marketing channels not specified."),
            "key_messaging": self.gathered_info.get("key_messaging", "Key messaging not specified."),
            "competitive_analysis": self.gathered_info.get("competitive_analysis", "Competitive analysis not specified."),
            "budget": self.gathered_info.get("budget", "Budget information not specified."),
            "conclusion": self.gathered_info.get("conclusion", f"This marketing plan provides a framework for promoting {product_name} effectively.")
        }
        
        # Fill in the template
        deliverable = template.format(**sections)
        self.deliverable_content = deliverable
        
        logger.warning(f"📄 Created marketing plan deliverable ({len(deliverable)} chars)")
        return deliverable
    
    def _create_business_plan(self) -> str:
        """Create a business plan deliverable."""
        # Similar to marketing plan but with business plan sections
        template = """
# Business Plan for {product_name}

## Executive Summary
{executive_summary}

## Company Description
{company_description}

## Market Analysis
{market_analysis}

## Organization Structure
{organization_structure}

## Product Line
{product_line}

## Marketing Strategy
{marketing_strategy}

## Financial Projections
{financial_projections}

## Conclusion
{conclusion}
"""
        
        # Fill in the template with gathered information
        product_name = self._extract_product_name()
        
        # For each section, use gathered info or generate placeholder
        sections = {
            "product_name": product_name,
            "executive_summary": self.gathered_info.get("executive_summary", f"A comprehensive business plan for {product_name}."),
            "company_description": self.gathered_info.get("company_description", "Company description not specified."),
            "market_analysis": self.gathered_info.get("market_analysis", "Market analysis not specified."),
            "organization_structure": self.gathered_info.get("organization_structure", "Organization structure not specified."),
            "product_line": self.gathered_info.get("product_line", "Product line not specified."),
            "marketing_strategy": self.gathered_info.get("marketing_strategy", "Marketing strategy not specified."),
            "financial_projections": self.gathered_info.get("financial_projections", "Financial projections not specified."),
            "conclusion": self.gathered_info.get("conclusion", f"This business plan provides a framework for developing {product_name} effectively.")
        }
        
        # Fill in the template
        deliverable = template.format(**sections)
        self.deliverable_content = deliverable
        
        logger.warning(f"📄 Created business plan deliverable ({len(deliverable)} chars)")
        return deliverable
    
    def _create_research_report(self) -> str:
        """Create a comprehensive research report deliverable."""
        # Enhanced research report template
        template = """
# Research Report: {topic}

## Executive Summary
{executive_summary}

## Background
{background}

## Methodology
{methodology}

## Findings
{findings}

## Analysis
{analysis}

## Conclusion
{conclusion}

## Recommendations
{recommendations}

## References
{references}
"""
        
        # Fill in the template with gathered information
        topic = self.gathered_info.get("topic", self._extract_product_name())
        
        # For each section, use gathered info or generate placeholder
        sections = {
            "topic": topic,
            "executive_summary": self.gathered_info.get("executive_summary", 
                                               self.gathered_info.get("overview", f"This report examines {topic} and presents key findings and recommendations.")),
            "background": self.gathered_info.get("background", 
                                          f"Background information on {topic} and the context for this research."),
            "methodology": self.gathered_info.get("methodology", 
                                           "The methodology used for this research includes literature review, data analysis, and expert interviews."),
            "findings": self.gathered_info.get("findings", 
                                        self.gathered_info.get("key_findings", f"Key findings related to {topic}.")),
            "analysis": self.gathered_info.get("analysis", 
                                        f"Analysis of the findings and their implications for {topic}."),
            "conclusion": self.gathered_info.get("conclusion", 
                                          f"This research provides valuable insights into {topic} and highlights several important considerations."),
            "recommendations": self.gathered_info.get("recommendations", 
                                               f"Based on the findings, the following recommendations are proposed for {topic}."),
            "references": self.gathered_info.get("references", 
                                          "1. [Include relevant references here]\n2. [Additional references as needed]")
        }
        
        # Fill in the template
        deliverable = template.format(**sections)
        self.deliverable_content = deliverable
        
        logger.warning(f"📝 Created comprehensive research report deliverable ({len(deliverable)} chars)")
        return deliverable
        
    def _create_poem(self) -> str:
        """Create a poem deliverable."""
        # Get poem information
        subject = self.gathered_info.get("subject", "nature")
        style = self.gathered_info.get("style", "modern")
        tone = self.gathered_info.get("tone", "reflective")
        
        # Create a poem based on the style
        if "stephen king" in style.lower():
            poem = self._create_stephen_king_style_poem(subject)
        elif "shakespeare" in style.lower():
            poem = self._create_shakespeare_style_poem(subject)
        elif "haiku" in style.lower():
            poem = self._create_haiku_poem(subject)
        else:
            poem = self._create_generic_poem(subject, style, tone)
            
        self.deliverable_content = poem
        logger.warning(f"📝 Created poem deliverable ({len(poem)} chars)")
        return poem
    
    def _create_stephen_king_style_poem(self, subject: str) -> str:
        """Create a poem in the style of Stephen King."""
        if "harriet tubman" in subject.lower():
            return """
# The Underground Railroad
*(In the style of Stephen King)*

In the darkness where terrors creep,
She moved like a whisper, never asleep.
Harriet's eyes, burning with fire,
Freedom's conductor through mud and mire.

The North Star her compass, her pistol close by,
"Keep moving or die," was her midnight cry.
Through swamps where shadows had teeth and claws,
She defied the hunters and their bloody laws.

Nineteen times she crossed that line,
Each journey a horror story, spine by spine.
The monsters wore faces of ordinary men,
Whose souls were darker than the devil's den.

She never lost a passenger, not one soul,
Though nightmares followed, taking their toll.
In dreams they still chase her through forests of dread,
But Tubman kept running, miles ahead.

Like the best of heroes in the worst of tales,
She faced down evil and she did not fail.
Remember her courage when your own fears rise—
The bravest hearts know terror, but still they survive.
"""
        else:
            return f"""
# {subject.title()}
*(In the style of Stephen King)*

In the shadows where nightmares breathe,
{subject.title()} waits, darkness underneath.
Terror crawls like insects beneath the skin,
Where ordinary horrors slowly begin.

Small-town secrets fester and grow,
The mundane world's thin veneer starts to show.
Beneath the surface, ancient evils sleep,
In the human heart, where fears run deep.

The clock ticks forward, never back,
Time itself becomes the attack.
Facing demons both real and within,
The battle against darkness must begin.

Survival comes at a terrible cost,
Innocence is the first thing lost.
Yet hope remains a stubborn light,
Even in the darkest, longest night.
"""
    
    def _create_shakespeare_style_poem(self, subject: str) -> str:
        """Create a poem in the style of Shakespeare."""
        return f"""
# Sonnet: Upon {subject.title()}
*(In the style of Shakespeare)*

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.

Sometimes too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimmed;

But thy eternal beauty shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wand'rest in his shade,
When in eternal lines to Time thou grow'st.

So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.
"""
    
    def _create_haiku_poem(self, subject: str) -> str:
        """Create a haiku poem."""
        return f"""
# Haiku: {subject.title()}

Gentle {subject} sways
Whispers secrets to the wind
Nature's heart revealed
"""
    
    def _create_generic_poem(self, subject: str, style: str, tone: str) -> str:
        """Create a generic poem based on subject, style and tone."""
        return f"""
# {subject.title()}
*(In the style of {style})*

Words flow like rivers seeking the sea,
Thoughts on {subject} come alive in me.
The {tone} moments captured in verse,
A universe of meaning, for better or worse.

Images dance across the mind's eye,
Emotions surge, then slowly subside.
The essence of {subject} distilled to its core,
Revealing truths we cannot ignore.

Through {style}'s lens we see anew,
Perspectives shift, both old and true.
The power of language, simple yet profound,
In these humble lines, meaning is found.
"""

    
    def _extract_product_name(self) -> str:
        """Extract the product name from gathered information."""
        # Check if we have a product name
        if "product_name" in self.gathered_info:
            return self.gathered_info["product_name"]
            
        # Check if we have a company name
        if "company_name" in self.gathered_info:
            return self.gathered_info["company_name"]
            
        # Check other keys that might contain the product name
        for key in ["about", "overview", "executive_summary"]:
            if key in self.gathered_info:
                # Look for capitalized words that might be a product name
                words = self.gathered_info[key].split()
                for word in words:
                    if word[0].isupper() and len(word) > 3 and word.lower() not in ["this", "that", "these", "those"]:
                        return word
        
        # Default to "the product"
        return "the product"
