import re

def extract_url(text):
    """
    Extracts the first URL from the given text.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: The extracted URL, or None if no URL is found.
    """
    # Regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    
    # Search for the pattern in the text
    match = re.search(url_pattern, text)
    
    if match:
        # If a match is found, return the matched URL
        url = match.group()
        if not url.startswith('http'):
            url = 'http://' + url
        return url
    else:
        # If no match is found, return None
        return None