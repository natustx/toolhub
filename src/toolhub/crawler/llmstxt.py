"""llms.txt crawler for AI-friendly documentation.

Implements the llms.txt standard (https://llmstxt.org/) for fetching
structured documentation designed for LLM consumption.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from toolhub.crawler.base import Crawler, CrawlResult

logger = logging.getLogger(__name__)

# Pattern to extract markdown links: [text](url)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def normalize_llmstxt_url(url: str) -> str:
    """Normalize URL to point to llms.txt.

    Handles various input formats:
    - https://example.com -> https://example.com/llms.txt
    - https://example.com/ -> https://example.com/llms.txt
    - https://example.com/llms.txt -> https://example.com/llms.txt
    - https://example.com/docs/llms.txt -> https://example.com/docs/llms.txt
    """
    parsed = urlparse(url)

    # If URL already ends with llms.txt, use as-is
    if parsed.path.endswith("llms.txt"):
        return url

    # Otherwise append /llms.txt to the path
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")

    # If path is empty or just /, add /llms.txt
    if not path:
        return f"{base}/llms.txt"

    # Otherwise append to existing path
    return f"{base}{path}/llms.txt"


def extract_links(content: str, base_url: str) -> list[tuple[str, str, str]]:
    """Extract markdown links from llms.txt content.

    Args:
        content: The llms.txt markdown content
        base_url: Base URL for resolving relative links

    Returns:
        List of (name, url, description) tuples
    """
    links = []
    for line in content.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue

        # Find the markdown link
        match = LINK_PATTERN.search(line)
        if not match:
            continue

        name = match.group(1).strip()
        url = match.group(2).strip()

        # Resolve relative URLs
        if not url.startswith(("http://", "https://")):
            url = urljoin(base_url, url)

        # Extract description (text after the link, usually after a colon)
        description = ""
        after_link = line[match.end() :].strip()
        if after_link.startswith(":"):
            description = after_link[1:].strip()

        links.append((name, url, description))

    return links


def fetch_url(url: str, timeout: float = 30.0) -> tuple[str | None, str | None]:
    """Fetch content from URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Tuple of (content, error) - one will be None
    """
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text, None
    except httpx.HTTPStatusError as e:
        return None, f"HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return None, str(e)


class LlmsTxtCrawler(Crawler):
    """Crawler for llms.txt files.

    Fetches the llms.txt file and optionally follows links to
    fetch additional markdown documentation.
    """

    def __init__(self, follow_links: bool = True, max_files: int = 100):
        """Initialize the llms.txt crawler.

        Args:
            follow_links: Whether to follow links in llms.txt to fetch more docs
            max_files: Maximum number of linked files to fetch
        """
        self.follow_links = follow_links
        self.max_files = max_files

    @property
    def source_type(self) -> str:
        return "llmstxt"

    def can_handle(self, url: str) -> bool:
        """Check if URL is or could be an llms.txt file."""
        url_lower = url.lower()

        # Direct llms.txt URLs
        if "llms.txt" in url_lower:
            return True

        # URLs that might have llms.txt at root
        if url_lower.startswith(("http://", "https://")):
            return True

        return False

    def crawl(self, url: str, cache_dir: Path) -> CrawlResult:
        """Fetch llms.txt and linked documentation.

        Args:
            url: URL to llms.txt or website root
            cache_dir: Directory to save fetched files

        Returns:
            CrawlResult with status and list of files
        """
        # Normalize URL to point to llms.txt
        llms_url = normalize_llmstxt_url(url)
        logger.info(f"Fetching llms.txt from {llms_url}")

        # Fetch main llms.txt
        content, error = fetch_url(llms_url)
        if error:
            return CrawlResult(
                source_url=url,
                source_type=self.source_type,
                cache_dir=cache_dir,
                files_crawled=[],
                error=f"Failed to fetch llms.txt: {error}",
            )

        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save main llms.txt
        files_crawled = []
        main_file = cache_dir / "llms.txt"
        main_file.write_text(content, encoding="utf-8")
        files_crawled.append(main_file)
        logger.info("Saved llms.txt")

        # Also check for llms-full.txt (comprehensive version)
        full_url = llms_url.replace("llms.txt", "llms-full.txt")
        full_content, _ = fetch_url(full_url)
        if full_content:
            full_file = cache_dir / "llms-full.txt"
            full_file.write_text(full_content, encoding="utf-8")
            files_crawled.append(full_file)
            logger.info("Saved llms-full.txt")

        # Follow links if enabled
        if self.follow_links:
            links = extract_links(content, llms_url)
            logger.info(f"Found {len(links)} links in llms.txt")

            fetched = 0
            for name, link_url, _ in links:
                if fetched >= self.max_files:
                    logger.info(f"Reached max files limit ({self.max_files})")
                    break

                # Only fetch markdown files
                if not link_url.endswith((".md", ".txt", ".markdown")):
                    # Try appending .md for HTML pages (per spec suggestion)
                    if link_url.endswith(".html"):
                        link_url = link_url[:-5] + ".md"
                    elif not link_url.endswith("/"):
                        link_url = link_url + ".md"
                    else:
                        continue

                link_content, link_error = fetch_url(link_url)
                if link_error:
                    logger.debug(f"Failed to fetch {link_url}: {link_error}")
                    continue

                # Generate safe filename from URL path
                parsed = urlparse(link_url)
                filename = parsed.path.strip("/").replace("/", "_")
                if not filename:
                    filename = f"doc_{fetched}.md"

                # Ensure .md extension
                if not filename.endswith((".md", ".txt")):
                    filename = filename + ".md"

                # Save file
                file_path = cache_dir / filename
                file_path.write_text(link_content, encoding="utf-8")
                files_crawled.append(file_path)
                fetched += 1
                logger.debug(f"Saved {filename}")

            logger.info(f"Fetched {fetched} linked documents")

        return CrawlResult(
            source_url=url,
            source_type=self.source_type,
            cache_dir=cache_dir,
            files_crawled=files_crawled,
        )
