
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def extract_documentation(url: str) -> dict:
    """Fetch one public documentation page and extract readable text."""

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https"):
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed_url.hostname:
        raise ValueError("Please provide a valid URL.")

    headers = {
        "User-Agent": "DevDocsAI/0.1 (documentation reader)"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15,
        allow_redirects=True,
    )
    response.raise_for_status()

    # Confirm the final destination after redirects.
    final_url = urlparse(response.url)

    if final_url.scheme not in ("http", "https"):
        raise ValueError("Unsupported redirect destination.")

    content_type = response.headers.get("Content-Type", "")

    if "text/html" not in content_type.lower():
        raise ValueError("The URL did not return an HTML page.")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that are generally not documentation content.
    for element in soup.select(
        "script, style, nav, footer, header, noscript"
    ):
        element.decompose()

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled documentation"
    )

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    text = main_content.get_text(" ", strip=True)

    if not text:
        raise ValueError("No readable text found on this page.")

    return {
        "title": title,
        "url": response.url,
        "text": text,
    }


if __name__ == "__main__":
    page = extract_documentation(
        "https://docs.python.org/3/tutorial/"
    )

    print("TITLE:", page["title"])
    print("SOURCE:", page["url"])
    print("EXTRACTED TEXT:")
    print(page["text"][:2000])
