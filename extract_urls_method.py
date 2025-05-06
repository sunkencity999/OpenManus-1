    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from the given text using multiple robust regex patterns."""
        if not text:
            return []
            
        # Multiple patterns to catch different URL formats
        patterns = [
            # Standard http/https URLs
            r'https?://[^\s()<>"\'`]+?(?:[\]\)]*[^\s`!()\[\]{};:\'".,<>?«»""'']|$)',
            # URLs in href attributes
            r'href=["\'](https?://[^"\']+)["\']',
            # URLs without scheme but with www
            r'\b(www\.[^\s()<>"\'`]+\.[a-zA-Z]{2,})(?:[\]\)]*[^\s`!()\[\]{};:\'".,<>?«»""'']|$)',
            # URLs in JSON data
            r'"url":\s*"(https?://[^"]+)"',
            # URLs with IP addresses
            r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}[^\s()<>"\'`]*'
        ]
        
        all_urls = []
        for pattern in patterns:
            found_urls = re.findall(pattern, text)
            # For tuple results (from groups), take the first item
            processed_urls = [u[0] if isinstance(u, tuple) else u for u in found_urls]
            all_urls.extend(processed_urls)
        
        # Process URLs: add scheme if missing, decode URL-encoded characters
        processed_urls = []
        for url in all_urls:
            # Add scheme if missing
            if url.startswith('www.'):
                url = 'https://' + url
            # Decode URL-encoded characters
            try:
                url = unquote(url)
            except Exception:
                pass
            processed_urls.append(url)
        
        # Basic validation and deduplication while preserving order
        seen = set()
        valid_urls = []
        for url in processed_urls:
            parsed = urlparse(url)
            # Ensure it has a domain part with at least one dot
            if "." in parsed.netloc and url not in seen:
                valid_urls.append(url)
                seen.add(url)
                
        return valid_urls
