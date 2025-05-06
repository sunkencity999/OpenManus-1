    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from the given text using multiple robust regex patterns."""
        if not text:
            return []
            
        # Standard http/https URLs pattern
        url_pattern = r'https?://[^\s()<>"\'`]+?(?:[\]\)]*[^\s`!()\[\]{};:\'".,<>?«»""'']|$)'
        
        # Also look for www. URLs without http/https
        www_pattern = r'\b(www\.[^\s()<>"\'`]+\.[a-zA-Z]{2,})'
        
        # Combine results from both patterns
        urls = re.findall(url_pattern, text)
        www_urls = re.findall(www_pattern, text)
        
        # Process www URLs to add https:// prefix
        for www_url in www_urls:
            if www_url.startswith('www.'):
                urls.append('https://' + www_url)
        
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
