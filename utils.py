import re

def extract_youtube_url(text: str) -> str | None:
    match = re.search(r"(https?://[^\s]+)", text)
    return match.group(1) if match else None
