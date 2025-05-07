"""
Task Completer Extensions for LocalManus

This module contains the implementation of additional task types for the TaskCompleter class.
These methods are imported and added to the TaskCompleter class to support a wider range of tasks.
"""

def _create_technical_documentation(self) -> str:
    """Create technical documentation deliverable."""
    # Technical documentation template
    template = """
# {product_name} Documentation
Version: {version}

## Overview
{purpose}

## Installation
{installation}

## Usage
{usage_examples}

## API Reference
{api_endpoints}

### Parameters
{parameters}

## Troubleshooting
{troubleshooting}

## Additional Resources
{additional_resources}
"""
    
    # Fill in the template with gathered information
    product_name = self.gathered_info.get("product_name", self._extract_product_name())
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "product_name": product_name,
        "version": self.gathered_info.get("version", "1.0.0"),
        "purpose": self.gathered_info.get("purpose", 
                                     f"{product_name} is a tool designed to help users accomplish their tasks efficiently."),
        "installation": self.gathered_info.get("installation", 
                                          f"```\npip install {product_name.lower()}\n```"),
        "usage_examples": self.gathered_info.get("usage_examples", 
                                            f"```python\nimport {product_name.lower()}\n\n# Example usage\nresult = {product_name.lower()}.process('example input')\nprint(result)\n```"),
        "api_endpoints": self.gathered_info.get("api_endpoints", 
                                           f"### `{product_name.lower()}.process(input_data)`\nProcesses the input data and returns a result."),
        "parameters": self.gathered_info.get("parameters", 
                                        "- `input_data` (str): The input data to process.\n- `options` (dict, optional): Additional options for processing."),
        "troubleshooting": self.gathered_info.get("troubleshooting", 
                                             "If you encounter any issues, please check the following:\n- Ensure you have the correct version installed\n- Verify your input format\n- Check the logs for error messages"),
        "additional_resources": self.gathered_info.get("additional_resources", 
                                                  f"- [GitHub Repository](https://github.com/example/{product_name.lower()})\n- [Official Website](https://example.com/{product_name.lower()})")
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created technical documentation deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_content_summary(self) -> str:
    """Create a content summary deliverable."""
    # Content summary template
    template = """
# Summary: {title}

## Key Points
{key_points}

## Summary
{summary}

## Audience Takeaways
{audience_takeaways}
"""
    
    # Fill in the template with gathered information
    source_content = self.gathered_info.get("source_content", "")
    title = self.gathered_info.get("title", self._extract_product_name())
    
    # Extract length preference if available
    length_preference = self.gathered_info.get("length", "medium")
    summary_length = {
        "short": "A brief summary of the key points.",
        "medium": "A comprehensive overview of the main ideas and supporting details.",
        "long": "A detailed summary covering all major points and significant supporting information."
    }.get(length_preference.lower(), "A comprehensive overview of the main ideas and supporting details.")
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "title": title,
        "key_points": self.gathered_info.get("key_points", 
                                        "- Key point 1\n- Key point 2\n- Key point 3"),
        "summary": self.gathered_info.get("summary", 
                                     summary_length),
        "audience_takeaways": self.gathered_info.get("audience_takeaways", 
                                                self.gathered_info.get("takeaways", 
                                                                  f"What the {self.gathered_info.get('audience', 'reader')} should remember from this content."))
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created content summary deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_data_analysis(self) -> str:
    """Create a data analysis report deliverable."""
    # Data analysis template
    template = """
# Data Analysis Report: {title}

## Executive Summary
{executive_summary}

## Dataset Description
{dataset_description}

## Methodology
{methodology}

## Findings
{findings}

## Visualizations
{visualizations}

## Statistical Analysis
{statistical_analysis}

## Conclusions
{conclusions}

## Recommendations
{recommendations}
"""
    
    # Fill in the template with gathered information
    title = self.gathered_info.get("title", self.gathered_info.get("analysis_goal", "Data Analysis"))
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "title": title,
        "executive_summary": self.gathered_info.get("executive_summary", 
                                               f"This report presents an analysis of {self.gathered_info.get('dataset', 'the dataset')} with the goal of {self.gathered_info.get('analysis_goal', 'understanding key patterns and insights')}."),
        "dataset_description": self.gathered_info.get("dataset_description", 
                                                 self.gathered_info.get("dataset", "The dataset used for this analysis contains information about [describe dataset variables and structure].") +
                                                 f"\n\nVariables analyzed: {self.gathered_info.get('variables', '[list key variables]')}"),
        "methodology": self.gathered_info.get("methodology", 
                                         self.gathered_info.get("methods", "The analysis employed statistical methods including descriptive statistics, correlation analysis, and visualization techniques.")),
        "findings": self.gathered_info.get("findings", 
                                      "The analysis revealed the following key findings:\n\n1. Finding 1\n2. Finding 2\n3. Finding 3"),
        "visualizations": self.gathered_info.get("visualizations", 
                                            "[Insert visualizations here - charts, graphs, etc.]"),
        "statistical_analysis": self.gathered_info.get("statistical_analysis", 
                                                  "Statistical tests performed:\n\n- Descriptive statistics (mean, median, standard deviation)\n- Correlation analysis\n- Hypothesis testing"),
        "conclusions": self.gathered_info.get("conclusions", 
                                         "Based on the analysis, we can conclude that [key conclusions about the data and what it reveals]."),
        "recommendations": self.gathered_info.get("recommendations", 
                                             "Based on the findings, the following recommendations are proposed:\n\n1. Recommendation 1\n2. Recommendation 2\n3. Recommendation 3")
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created data analysis report deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_blog_post(self) -> str:
    """Create a blog post deliverable."""
    # Blog post template
    template = """
# {title}

*{date} · {reading_time} min read*

![{featured_image_alt}]({featured_image_url})

{introduction}

## {section1_title}
{section1_content}

## {section2_title}
{section2_content}

## {section3_title}
{section3_content}

## Conclusion
{conclusion}

---

*{author_bio}*

**Tags:** {tags}
"""
    
    # Fill in the template with gathered information
    topic = self.gathered_info.get("topic", self._extract_product_name())
    title = self.gathered_info.get("title", f"Everything You Need to Know About {topic}")
    
    # Generate current date
    import datetime
    current_date = datetime.datetime.now().strftime("%B %d, %Y")
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "title": title,
        "date": self.gathered_info.get("date", current_date),
        "reading_time": self.gathered_info.get("reading_time", "5"),
        "featured_image_alt": self.gathered_info.get("featured_image_alt", f"{topic} featured image"),
        "featured_image_url": self.gathered_info.get("featured_image_url", "https://example.com/image.jpg"),
        "introduction": self.gathered_info.get("introduction", 
                                          f"In this article, we'll explore {topic} and why it matters. {self.gathered_info.get('key_points', 'We will cover the essential aspects and provide actionable insights.')}"),
        "section1_title": self.gathered_info.get("section1_title", "Background"),
        "section1_content": self.gathered_info.get("section1_content", f"Let's start by understanding what {topic} is and why it's important."),
        "section2_title": self.gathered_info.get("section2_title", "Key Considerations"),
        "section2_content": self.gathered_info.get("section2_content", f"When dealing with {topic}, there are several important factors to consider."),
        "section3_title": self.gathered_info.get("section3_title", "Best Practices"),
        "section3_content": self.gathered_info.get("section3_content", f"Here are some best practices for {topic} that you can implement right away."),
        "conclusion": self.gathered_info.get("conclusion", 
                                        f"To summarize, {topic} is a crucial aspect that deserves attention. By following the guidelines outlined in this article, you'll be well-equipped to handle it effectively. {self.gathered_info.get('call_to_action', 'Start implementing these strategies today and see the difference they make!')}"),
        "author_bio": self.gathered_info.get("author_bio", "Written by an expert in the field with over 10 years of experience."),
        "tags": self.gathered_info.get("tags", self.gathered_info.get("seo_keywords", f"{topic}, guide, best practices"))
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created blog post deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_email_content(self) -> str:
    """Create email content deliverable."""
    # Email content template
    template = """
# Email: {subject_line}

**To:** {recipient}
**From:** {sender}
**Subject:** {subject_line}

Dear {recipient_name},

{opening}

{body}

{closing}

{signature}

---

**Purpose:** {purpose}
**Call to Action:** {call_to_action}
"""
    
    # Fill in the template with gathered information
    purpose = self.gathered_info.get("purpose", "To inform the recipient about important information")
    recipient = self.gathered_info.get("recipient", "Valued Customer")
    recipient_name = self.gathered_info.get("recipient_name", recipient.split('@')[0] if '@' in recipient else recipient)
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "subject_line": self.gathered_info.get("subject_line", f"Important Information About {self._extract_product_name()}"),
        "recipient": recipient,
        "sender": self.gathered_info.get("sender", "Your Name <your.email@example.com>"),
        "recipient_name": recipient_name,
        "opening": self.gathered_info.get("opening", f"I hope this email finds you well. I'm reaching out regarding {self._extract_product_name()}."),
        "body": self.gathered_info.get("body", 
                                  self.gathered_info.get("key_message", f"We have some important updates about {self._extract_product_name()} that we wanted to share with you.")),
        "closing": self.gathered_info.get("closing", "Thank you for your attention to this matter. Please don't hesitate to reach out if you have any questions."),
        "signature": self.gathered_info.get("signature", "Best regards,\n[Your Name]\n[Your Position]\n[Your Contact Information]"),
        "purpose": purpose,
        "call_to_action": self.gathered_info.get("call_to_action", "Please review the information and respond by [date].")
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created email content deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_social_media_content(self) -> str:
    """Create social media content deliverable."""
    # Social media content template
    template = """
# Social Media Content: {platform}

## Post
{post_content}

## Hashtags
{hashtags}

## Image Description
{image_description}

## Call to Action
{call_to_action}

## Posting Schedule
{posting_schedule}

## Engagement Strategy
{engagement_strategy}
"""
    
    # Fill in the template with gathered information
    platform = self.gathered_info.get("platform", "Twitter")
    topic = self._extract_product_name()
    
    # Adjust content length based on platform
    platform_limits = {
        "twitter": 280,
        "instagram": 2200,
        "facebook": 5000,
        "linkedin": 3000
    }
    
    platform_key = platform.lower()
    if platform_key in ["twitter", "x"]:
        platform_key = "twitter"
        platform_display = "Twitter/X"
    else:
        platform_display = platform
    
    char_limit = platform_limits.get(platform_key, 1000)
    
    # Default post content based on platform
    default_post = f"Excited to share our latest updates about {topic}! Check out the link below to learn more. #NewAnnouncement"
    if len(default_post) > char_limit:
        default_post = default_post[:char_limit-3] + "..."
    
    # For each section, use gathered info or generate placeholder
    sections = {
        "platform": platform_display,
        "post_content": self.gathered_info.get("post_content", 
                                          self.gathered_info.get("key_message", default_post)),
        "hashtags": self.gathered_info.get("hashtags", f"#{topic.replace(' ', '')} #NewAnnouncement #Update"),
        "image_description": self.gathered_info.get("image_description", f"[Image showing {topic} with engaging visuals]"),
        "call_to_action": self.gathered_info.get("call_to_action", "Click the link to learn more"),
        "posting_schedule": self.gathered_info.get("posting_schedule", "Best times to post:\n- Weekdays: 12 PM - 1 PM\n- Weekends: 10 AM - 11 AM"),
        "engagement_strategy": self.gathered_info.get("engagement_strategy", "- Respond to comments within 2 hours\n- Ask questions to encourage discussion\n- Share user-generated content when available")
    }
    
    # Fill in the template
    deliverable = template.format(**sections)
    self.deliverable_content = deliverable
    
    self.logger.warning(f"📝 Created social media content deliverable ({len(deliverable)} chars)")
    return deliverable

def _create_website(self) -> str:
    """Create a website deliverable with HTML, CSS, and JavaScript files."""
    # Get website information from gathered info
    website_name = self.gathered_info.get("website_name", self._extract_product_name())
    website_url = self.gathered_info.get("website_url", "")
    color_scheme = self.gathered_info.get("color_scheme", "modern and minimalist")
    
    # Handle folder name based on user preference or generate automatically
    if "folder_name" in self.gathered_info and self.gathered_info["folder_name"]:
        # Use the user-specified folder name
        folder_name = self.gathered_info["folder_name"]
    else:
        # Generate a folder name based on the website name
        # Convert to lowercase, replace spaces with underscores, and remove special characters
        import re
        folder_name = re.sub(r'[^\w\s-]', '', website_name.lower())
        folder_name = re.sub(r'[\s]+', '_', folder_name)
        
        # If folder name is still empty, use a generic name
        if not folder_name:
            folder_name = "website_project"
    
    # Determine colors based on color scheme
    colors = {
        "background": "#ffffff",
        "text": "#333333",
        "primary": "#4a6fa5",
        "secondary": "#e9ecef",
        "accent": "#28a745"
    }
    
    if "dark" in color_scheme.lower():
        colors = {
            "background": "#121212",
            "text": "#e0e0e0",
            "primary": "#4a6fa5",
            "secondary": "#2d2d2d",
            "accent": "#28a745"
        }
    
    # Create folder structure
    import os
    
    # Use workspace folder as the base directory
    workspace_dir = os.path.join(os.getcwd(), "workspace")
    os.makedirs(workspace_dir, exist_ok=True)  # Create workspace folder if it doesn't exist
    
    # Create the project folder inside the workspace
    base_dir = os.path.join(workspace_dir, folder_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["css", "js", "images"]:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    pages = ["index", "projects", "services", "contact"]
    
    # Create CSS file
    css_content = f"""/* Styles for {website_name} */
* {{\n  margin: 0;\n  padding: 0;\n  box-sizing: border-box;\n}}\n\nbody {{\n  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n  line-height: 1.6;\n  color: {colors['text']};\n  background-color: {colors['background']};\n}}\n\nheader {{\n  background-color: {colors['primary']};\n  color: white;\n  padding: 1rem;\n  text-align: center;\n}}\n\nnav {{\n  display: flex;\n  justify-content: center;\n  background-color: {colors['secondary']};\n  padding: 1rem;\n}}\n\nnav a {{\n  color: {colors['text']};\n  text-decoration: none;\n  padding: 0.5rem 1rem;\n  margin: 0 0.5rem;\n  border-radius: 4px;\n  transition: background-color 0.3s;\n}}\n\nnav a:hover {{\n  background-color: {colors['primary']};\n  color: white;\n}}\n\n.container {{\n  max-width: 1200px;\n  margin: 0 auto;\n  padding: 2rem;\n}}\n\n.hero {{\n  text-align: center;\n  padding: 3rem 0;\n}}\n\n.hero h1 {{\n  font-size: 2.5rem;\n  margin-bottom: 1rem;\n}}\n\n.hero p {{\n  font-size: 1.2rem;\n  max-width: 800px;\n  margin: 0 auto;\n}}\n\n.btn {{\n  display: inline-block;\n  background-color: {colors['accent']};\n  color: white;\n  padding: 0.5rem 1rem;\n  border-radius: 4px;\n  text-decoration: none;\n  margin-top: 1rem;\n}}\n\n.btn:hover {{\n  opacity: 0.9;\n}}\n\n.features {{\n  display: grid;\n  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));\n  gap: 2rem;\n  margin: 2rem 0;\n}}\n\n.feature {{\n  padding: 1.5rem;\n  border-radius: 8px;\n  background-color: {colors['secondary']};\n  text-align: center;\n}}\n\nfooter {{\n  background-color: {colors['primary']};\n  color: white;\n  text-align: center;\n  padding: 1rem;\n  margin-top: 2rem;\n}}\n\n/* Responsive design */\n@media (max-width: 768px) {{\n  nav {{\n    flex-direction: column;\n  }}\n  \n  nav a {{\n    margin: 0.2rem 0;\n  }}\n  \n  .features {{\n    grid-template-columns: 1fr;\n  }}\n}}\n"""
    
    with open(base_dir / "css" / "style.css", "w") as f:
        f.write(css_content)
    
    # Create JavaScript file
    js_content = """// JavaScript for website functionality
document.addEventListener('DOMContentLoaded', function() {
  // Mobile menu toggle
  const navToggle = document.querySelector('.nav-toggle');
  const navMenu = document.querySelector('nav');
  
  if (navToggle) {
    navToggle.addEventListener('click', function() {
      navMenu.classList.toggle('active');
    });
  }
  
  // Form validation for contact page
  const contactForm = document.getElementById('contact-form');
  
  if (contactForm) {
    contactForm.addEventListener('submit', function(event) {
      event.preventDefault();
      
      const name = document.getElementById('name').value;
      const email = document.getElementById('email').value;
      const message = document.getElementById('message').value;
      
      if (!name || !email || !message) {
        alert('Please fill in all fields');
        return;
      }
      
      // Simulate form submission
      alert('Thank you for your message! We will get back to you soon.');
      contactForm.reset();
    });
  }
});
"""
    
    with open(base_dir / "js" / "main.js", "w") as f:
        f.write(js_content)
    
    # Create HTML files
    for page in pages:
        page_title = page.capitalize()
        if page == "index":
            page_title = "Home"
        
        content = ""
        
        if page == "index":
            content = f"""<div class="hero">\n  <h1>Welcome to {website_name}</h1>\n  <p>A modern and clean website with beautiful typography for a software development and media firm based in the Santa Cruz mountains.</p>\n  <a href="services.html" class="btn">Our Services</a>\n</div>\n\n<div class="features">\n  <div class="feature">\n    <h2>Professional Photography</h2>\n    <p>High-quality photography services for all your needs.</p>\n  </div>\n  <div class="feature">\n    <h2>Creative Writing</h2>\n    <p>Engaging content that captures your audience's attention.</p>\n  </div>\n  <div class="feature">\n    <h2>Videography</h2>\n    <p>Professional video production for various purposes.</p>\n  </div>\n</div>"""
        elif page == "projects":
            content = f"""<h1>Our Projects</h1>\n<p>Here are some of our recent projects:</p>\n\n<div class="features">\n  <div class="feature">\n    <h2>Project 1</h2>\n    <p>A software solution for local businesses.</p>\n  </div>\n  <div class="feature">\n    <h2>Project 2</h2>\n    <p>Custom website development for a non-profit organization.</p>\n  </div>\n  <div class="feature">\n    <h2>Project 3</h2>\n    <p>Photography and videography for a marketing campaign.</p>\n  </div>\n</div>"""
        elif page == "services":
            content = f"""<h1>Our Services</h1>\n<p>We offer a range of professional services:</p>\n\n<div class="features">\n  <div class="feature">\n    <h2>Software Development</h2>\n    <p>Custom software solutions that solve real problems for real people.</p>\n  </div>\n  <div class="feature">\n    <h2>Professional Photography</h2>\n    <p>High-quality photography for various purposes.</p>\n  </div>\n  <div class="feature">\n    <h2>Content Writing</h2>\n    <p>Engaging and informative content for your website or marketing materials.</p>\n  </div>\n  <div class="feature">\n    <h2>Videography</h2>\n    <p>Professional video production for various purposes.</p>\n  </div>\n</div>"""
        elif page == "contact":
            content = f"""<h1>Contact Us</h1>\n<p>Get in touch with us:</p>\n\n<form id="contact-form">\n  <div class="form-group">\n    <label for="name">Name</label>\n    <input type="text" id="name" name="name" required>\n  </div>\n  <div class="form-group">\n    <label for="email">Email</label>\n    <input type="email" id="email" name="email" required>\n  </div>\n  <div class="form-group">\n    <label for="message">Message</label>\n    <textarea id="message" name="message" rows="5" required></textarea>\n  </div>\n  <button type="submit" class="btn">Send Message</button>\n</form>\n\n<div class="contact-info">\n  <h2>Our Location</h2>\n  <p>Santa Cruz Mountains, California</p>\n  <h2>Email</h2>\n  <p>info@{website_url.replace('https://', '') if website_url else 'example.com'}</p>\n</div>"""
        
        html_content = f"""<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n  <title>{website_name} - {page_title}</title>\n  <link rel="stylesheet" href="css/style.css">\n</head>\n<body>\n  <header>\n    <h1>{website_name}</h1>\n    <button class="nav-toggle">Menu</button>\n  </header>\n  \n  <nav>\n    <a href="index.html">Home</a>\n    <a href="projects.html">Projects</a>\n    <a href="services.html">Services</a>\n    <a href="contact.html">Contact</a>\n  </nav>\n  \n  <div class="container">\n    {content}\n  </div>\n  \n  <footer>\n    <p>&copy; {website_name} {self.gathered_info.get('year', '2025')}. All rights reserved.</p>\n  </footer>\n  \n  <script src="js/main.js"></script>\n</body>\n</html>"""
        
        file_name = f"{page}.html"
        with open(base_dir / file_name, "w") as f:
            f.write(html_content)
    
    # Create a README file
    readme_content = f"""# {website_name}\n\nA modern and clean website with beautiful typography.\n\n## Pages\n\n- Home: Introduction to the company\n- Projects: Showcase of recent projects\n- Services: Details of services offered\n- Contact: Contact form and information\n\n## Technologies Used\n\n- HTML5\n- CSS3\n- JavaScript\n\n## How to Use\n\n1. Clone this repository\n2. Open index.html in your browser\n\n## License\n\nAll rights reserved.\n"""
    
    with open(base_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create a summary of what was created
    deliverable = f"""# Website: {website_name}\n\n## Files Created\n\n- HTML: index.html, projects.html, services.html, contact.html\n- CSS: css/style.css\n- JavaScript: js/main.js\n- README.md\n\n## Structure\n\n```\n{folder_name}/\n├── css/\n│   └── style.css\n├── js/\n│   └── main.js\n├── images/\n├── index.html\n├── projects.html\n├── services.html\n├── contact.html\n└── README.md\n```\n\n## Design\n\n- Color Scheme: {color_scheme}\n- Typography: Modern sans-serif fonts\n- Responsive: Yes, works on mobile and desktop\n\n## Features\n\n- Clean, modern design\n- Responsive navigation\n- Contact form with validation\n- Showcase of services and projects\n\nThe website has been created in the '{folder_name}' directory.\n"""
    
    self.deliverable_content = deliverable
    self.logger.warning(f"📝 Created website deliverable in folder '{folder_name}'")
    return deliverable
