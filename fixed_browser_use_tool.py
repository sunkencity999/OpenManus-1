"""
This is a fixed version of the extract_and_save method and _try_ollama_generate method
for the BrowserUseTool class, focusing on proper Ollama integration.

Replace the corresponding sections in the original browser_use_tool.py file.
"""

# EXTRACT_AND_SAVE METHOD
# Replace the extract_and_save action case in the _execute_internal method

elif action == "extract_and_save":
    if not goal:
        return ToolResult(
            error="Goal is required for 'extract_and_save' action"
        )
    if not filename:
        return ToolResult(
            error="Filename is required for 'extract_and_save' action"
        )
        
    # Get the content format, default to markdown
    content_format = kwargs.get("format", "markdown")
    
    # STEP 1: Save the raw content from the webpage
    try:
        page = await context.get_current_page()
        import markdownify
        
        # Get the page content and metadata
        raw_html = await page.content()
        page_url = page.url
        page_title = await page.title()
        logger.info(f"Got raw HTML content from {page_url}, length: {len(raw_html)}")
        
        # Convert HTML to markdown for better readability
        content = markdownify.markdownify(raw_html)
        logger.info(f"Converted HTML to markdown, length: {len(content)}")
        
        # Create the workspace directory structure
        workspace_dir = os.path.join(os.getcwd(), "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Create a raw content directory
        raw_content_dir = os.path.join(workspace_dir, "raw_content")
        os.makedirs(raw_content_dir, exist_ok=True)
        
        # Generate a timestamp for the raw content file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a directory for this specific extraction
        extraction_dir = os.path.join(raw_content_dir, f"extraction_{timestamp}")
        os.makedirs(extraction_dir, exist_ok=True)
        
        # Save the raw HTML
        raw_html_path = os.path.join(extraction_dir, "raw.html")
        with open(raw_html_path, 'w', encoding='utf-8') as f:
            f.write(raw_html)
        logger.info(f"Saved raw HTML content to {raw_html_path}")
        
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
        
        # STEP 2: Analyze the raw content directly without using function calling
        # This is more reliable than using function calling which may not be supported by all LLMs
        logger.info("Starting content analysis...")
        
        # Create a simpler prompt that doesn't rely on function calling
        analysis_prompt = f"""
        You are an expert content analyzer. Your task is to extract and summarize information from the following webpage content.
        
        GOAL: {goal}
        
        PAGE TITLE: {page_title}
        PAGE URL: {page_url}
        
        Please provide a detailed analysis focusing specifically on this goal. Include all relevant information, facts, figures, and quotes.
        Structure your response with appropriate headings and sections. Be comprehensive but concise.
        
        CONTENT:
        {content[:5000] if len(content) > 5000 else content}
        
        {"[Content truncated due to length]" if len(content) > 5000 else ""}
        """
        
        # Initialize analysis_response to None
        analysis_response = None
        
        # Try to get a response from the LLM
        try:
            # Handle Ollama integration based on the memory about LocalManus
            # LocalManus uses Ollama for LLM functionality
            logger.info(f"Current LLM configuration: model={self.llm.model}, api_type={self.llm.api_type}, base_url={self.llm.base_url}")
            
            # Determine if we're using Ollama
            is_ollama = False
            if hasattr(self.llm, 'api_type') and self.llm.api_type.lower() == 'ollama':
                is_ollama = True
            elif 'ollama' in self.llm.base_url.lower():
                is_ollama = True
            
            # For Ollama, we need to use the correct API format
            if is_ollama:
                logger.info("Using direct Ollama API for content analysis")
                
                # Get the base URL without /v1 if present
                # LocalManus uses Ollama and the base_url includes '/v1' which needs to be removed
                base_url = self.llm.base_url
                if base_url.endswith('/v1'):
                    base_url = base_url[:-3]
                logger.info(f"Ollama base URL: {base_url}")
                
                # Create a session with a longer timeout (5 minutes)
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes in seconds
                logger.info(f"Using timeout of {timeout.total} seconds for Ollama API request")
                
                # Try the chat API first
                try:
                    # Format messages for Ollama chat API - use 'user' role instead of 'system'
                    ollama_messages = [
                        {"role": "user", "content": analysis_prompt}
                    ]
                    
                    # Prepare the request payload
                    chat_payload = {
                        "model": self.llm.model,
                        "messages": ollama_messages,
                        "stream": False
                    }
                    
                    logger.info(f"Sending request to Ollama chat API with model: {self.llm.model}")
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        # Use the /api/chat endpoint for chat completions
                        chat_url = f"{base_url}/api/chat"
                        logger.info(f"Sending request to: {chat_url}")
                        logger.info("Waiting for Ollama to generate response (this may take a while)...")
                        
                        start_time = datetime.now()
                        async with session.post(chat_url, json=chat_payload) as response:
                            end_time = datetime.now()
                            duration = (end_time - start_time).total_seconds()
                            logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                            
                            if response.status == 200:
                                result = await response.json()
                                logger.info(f"Received response from Ollama chat API with keys: {result.keys() if result else 'None'}")
                                
                                # Extract the content from the response
                                if result and 'message' in result and 'content' in result['message']:
                                    analysis_response = result['message']['content']
                                    logger.info(f"Successfully extracted content from Ollama chat response, length: {len(analysis_response)}")
                                else:
                                    logger.warning(f"Unexpected response format from Ollama chat API: {result}")
                                    logger.info("Falling back to /api/generate endpoint...")
                                    analysis_response = await self._try_ollama_generate(base_url, self.llm.model, analysis_prompt, timeout)
                            else:
                                error_text = await response.text()
                                logger.warning(f"Error from Ollama chat API: {response.status} - {error_text}")
                                logger.info("Falling back to /api/generate endpoint...")
                                analysis_response = await self._try_ollama_generate(base_url, self.llm.model, analysis_prompt, timeout)
                except Exception as e:
                    logger.warning(f"Exception during Ollama chat API request: {str(e)}")
                    logger.info("Falling back to /api/generate endpoint...")
                    analysis_response = await self._try_ollama_generate(base_url, self.llm.model, analysis_prompt, timeout)
            else:
                # Use the standard LLM interface for non-Ollama models
                logger.info("Using standard LLM interface for content analysis")
                analysis_messages = [{"role": "system", "content": analysis_prompt}]
                analysis_response = await self.llm.ask(analysis_messages, stream=False)
                
            # Log the response details
            logger.info(f"Analysis response type: {type(analysis_response)}")
            if analysis_response:
                logger.info(f"Analysis response length: {len(analysis_response)}")
            else:
                logger.info("Analysis response is None or empty")
            
            logger.info(f"Received analysis response, length: {len(analysis_response) if analysis_response else 0}")
            
        except Exception as e:
            logger.error(f"Error during LLM analysis: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            analysis_response = None
        
        # Use analysis_response as the analysis content
        analysis_content = analysis_response
        
        # If analysis_response is None or empty, create a fallback analysis
        if not analysis_response:
            logger.info("Creating fallback analysis from raw content")
            analysis_response = f"# {page_title}\n\n## {goal}\n\n*This is an automated extraction of content from the webpage.*\n\n"
            
            # Add a preview of the content
            content_preview = content[:1000] + "..." if len(content) > 1000 else content
            analysis_response += f"### Content Preview\n\n{content_preview}\n\n"
            
            # Add a note about the raw content
            analysis_response += f"*Complete content is available in the raw files at: `{extraction_dir}`*\n"
            
            # Update analysis_content with the fallback
            analysis_content = analysis_response
        
        # Create the formatted content based on the requested format
        if content_format == "json":
            # Create a JSON structure with the analysis and metadata
            formatted_data = {
                "goal": goal,
                "url": page_url,
                "title": page_title,
                "analysis": analysis_content,
                "metadata": {
                    "extraction_time": datetime.now().isoformat(),
                    "source": page_url,
                    "raw_content_path": extraction_dir
                }
            }
            formatted_content = json.dumps(formatted_data, indent=2)
            
        elif content_format == "html":
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
                    {analysis_content}
                </div>
                <p class="source">Source: <a href="{page_url}">{page_url}</a></p>
                <div class="extraction-info">
                    Extraction time: {datetime.now().isoformat()}<br>
                    Raw content available at: {extraction_dir}
                </div>
            </body>
            </html>"""
            
        elif content_format == "text":
            # Create a simple text format
            formatted_content = f"""GOAL: {goal}

TITLE: {page_title}

SOURCE: {page_url}

{analysis_content}

Extraction time: {datetime.now().isoformat()}
Raw content available at: {extraction_dir}
"""
            
        else:  # Default to markdown
            # Create a nicely formatted markdown document
            formatted_content = f"""# {goal}

## {page_title}

{analysis_content}

---

*Source: [{page_url}]({page_url})*

*Extraction time: {datetime.now().isoformat()}*

*Raw content available at: `{extraction_dir}`*
"""
        
        # Write the formatted content to the output file
        output_path = os.path.join(os.getcwd(), "workspace", filename)
        try:
            # Create the workspace directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write the content to the file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_content)
            
            logger.info(f"Saved formatted content to {output_path}")
            
            # Verify the file was created
            if os.path.exists(output_path):
                logger.info(f"Verified file exists at {output_path}")
                file_size = os.path.getsize(output_path)
                logger.info(f"File size: {file_size} bytes")
            else:
                logger.error(f"File was not created at {output_path}")
            
            return ToolResult(
                output=f"Successfully extracted and analyzed content from {page_url}.\n\nAnalysis saved to: {output_path}\nRaw content saved to: {extraction_dir}"
            )
            
        except Exception as e:
            logger.error(f"Error saving formatted output: {str(e)}")
            return ToolResult(
                error=f"Failed to save formatted output: {str(e)}"
            )
    
    except Exception as e:
        logger.error(f"Error in extraction process: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ToolResult(
            error=f"Failed to extract content: {str(e)}"
        )


# _TRY_OLLAMA_GENERATE METHOD
# Add this method to the BrowserUseTool class

async def _try_ollama_generate(self, base_url, model, prompt, timeout):
    """Try the Ollama /api/generate endpoint as a fallback."""
    logger.info("Using Ollama /api/generate endpoint as fallback...")
    
    # Prepare the request payload for the generate API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    logger.info(f"Sending request to Ollama generate API with model: {model}")
    
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Use the /api/generate endpoint
            generate_url = f"{base_url}/api/generate"
            logger.info(f"Sending request to: {generate_url}")
            
            # Log that we're waiting for the response
            logger.info("Waiting for Ollama to generate response (this may take a while)...")
            from datetime import datetime
            start_time = datetime.now()
            
            # Make the request with a longer timeout
            async with session.post(generate_url, json=payload) as response:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Received HTTP status: {response.status} after {duration:.2f} seconds")
                
                if response.status == 200:
                    # Log that we're parsing the response
                    logger.info("Parsing response from Ollama generate API...")
                    
                    # Parse the JSON response
                    result = await response.json()
                    logger.info(f"Received response from Ollama generate API with keys: {result.keys() if result else 'None'}")
                    
                    # Extract the content from the response
                    if result and 'response' in result:
                        content = result['response']
                        logger.info(f"Successfully extracted content from Ollama generate response, length: {len(content)}")
                        return content
                    else:
                        logger.error(f"Unexpected response format from Ollama generate API: {result}")
                        return f"Error: Unexpected response format from Ollama generate API: {result}"
                else:
                    error_text = await response.text()
                    logger.error(f"Error from Ollama generate API: {response.status} - {error_text}")
                    return f"Error from Ollama generate API: {response.status} - {error_text}"
    
    except Exception as e:
        logger.error(f"Exception during Ollama generate API request: {str(e)}")
        return f"Error communicating with Ollama generate API: {str(e)}"
