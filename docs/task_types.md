# LocalManus Task Types

This document outlines the various task types supported by LocalManus and how they can be used to generate different kinds of content.

## Overview

LocalManus has been enhanced to support a wide range of task types beyond the original research capabilities. The agent can now detect the type of task from the user's request and gather the appropriate information to create specialized deliverables.

## Supported Task Types

LocalManus currently supports the following task types:

### 1. Research Report

A comprehensive research report with structured sections including background, methodology, findings, analysis, and recommendations.

**Example prompt:** "Create a research report on renewable energy sources"

**Output sections:**
- Executive Summary
- Background
- Methodology
- Findings
- Analysis
- Conclusion
- Recommendations
- References

### 2. Technical Documentation

Technical documentation for software, APIs, or other technical products, including installation instructions, usage examples, and API references.

**Example prompt:** "Write technical documentation for my Python library called DataProcessor"

**Output sections:**
- Product Overview
- Installation
- Usage Examples
- API Reference
- Parameters
- Troubleshooting
- Additional Resources

### 3. Content Summary

A concise summary of longer content, highlighting key points and takeaways for a specified audience.

**Example prompt:** "Summarize the key points of the latest climate change report"

**Output sections:**
- Key Points
- Summary
- Audience Takeaways

### 4. Data Analysis

A detailed analysis of data, including findings, visualizations, and recommendations based on the analysis.

**Example prompt:** "Analyze the sales data for Q1 2023 and provide insights"

**Output sections:**
- Executive Summary
- Dataset Description
- Methodology
- Findings
- Visualizations
- Statistical Analysis
- Conclusions
- Recommendations

### 5. Blog Post

A structured blog post with an engaging introduction, well-organized sections, and a clear conclusion.

**Example prompt:** "Write a blog post about machine learning applications in healthcare"

**Output sections:**
- Title
- Introduction
- Multiple Content Sections
- Conclusion
- Author Bio
- Tags

### 6. Email Content

Professional email content with a clear subject line, message body, and call to action.

**Example prompt:** "Draft an email to customers about our new product launch"

**Output sections:**
- Subject Line
- Recipient
- Opening
- Body
- Closing
- Signature
- Call to Action

### 7. Social Media Content

Content optimized for social media platforms, including post text, hashtags, and engagement strategies.

**Example prompt:** "Create social media content for Twitter announcing our company anniversary"

**Output sections:**
- Post Content
- Hashtags
- Image Description
- Call to Action
- Posting Schedule
- Engagement Strategy

### 8. Business Plan

A comprehensive business plan with all essential sections for a new business or product.

**Example prompt:** "Create a business plan for a mobile app startup"

**Output sections:**
- Executive Summary
- Company Description
- Market Analysis
- Organization Structure
- Product Line
- Marketing Strategy
- Financial Projections

### 9. Marketing Plan

A detailed marketing plan with target audience analysis, value proposition, and marketing channels.

**Example prompt:** "Develop a marketing plan for a new fitness product"

**Output sections:**
- Target Audience
- Value Proposition
- Marketing Channels
- Key Messaging
- Competitive Analysis
- Budget

### 10. Poem

Creative poetry in various styles and forms.

**Example prompt:** "Write a poem about autumn in sonnet style"

**Output sections:**
- Title
- Poem Content
- (Style and structure will vary based on the requested poetic form)

## Using Task Types

The agent automatically detects the appropriate task type based on your request. To get the best results:

1. Be specific about the type of content you need (e.g., "Create a research report on..." rather than just "Tell me about...")
2. Include any specific requirements or preferences in your request
3. Provide additional information when prompted by the agent

## Extending Task Types

LocalManus is designed to be extensible. New task types can be added to the system by:

1. Adding detection logic in the `analyze_task` method
2. Creating a template for the new task type
3. Implementing a method to generate the deliverable

For developers, see the `task_completer.py` and `task_completer_extensions.py` files for implementation details.

## Demo

You can try out the different task types using the included demo script:

```bash
python task_type_demo.py
```

This will allow you to select a task type and see how the agent handles it, from task detection to deliverable generation.
