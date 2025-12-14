"""Website crawler for documentation sites.

Crawls HTML pages and converts them to markdown for indexing.
Respects robots.txt and implements rate limiting.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from toolhub.crawler.base import Crawler, CrawlResult

logger = logging.getLogger(__name__)

# Default user agent
USER_AGENT = "toolhub/0.1 (documentation indexer)"

# Elements to remove before conversion
REMOVE_SELECTORS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    ".sidebar",
    ".navigation",
    ".nav",
    ".menu",
    ".toc",
    ".breadcrumb",
    ".pagination",
    ".comments",
    ".advertisement",
    ".ad",
    "#sidebar",
    "#navigation",
    "#nav",
    "#menu",
    "#toc",
    "#comments",
]

# Content selectors to try (in order of preference)
CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".content",
    ".documentation",
    ".docs",
    ".markdown-body",
    ".post-content",
    "#content",
    "#main",
    ".main-content",
]


@dataclass
class CrawlConfig:
    """Configuration for website crawling."""

    max_pages: int = 250
    max_depth: int = 3
    rate_limit: float = 1.0  # seconds between requests
    timeout: float = 30.0
    respect_robots: bool = True
    follow_external: bool = False


@dataclass
class PageResult:
    """Result of crawling a single page."""

    url: str
    title: str
    content: str
    links: list[str] = field(default_factory=list)
    error: str | None = None


def get_robots_parser(base_url: str, timeout: float = 10.0) -> RobotFileParser | None:
    """Fetch and parse robots.txt for a site.

    Args:
        base_url: Base URL of the site
        timeout: Request timeout

    Returns:
        RobotFileParser or None if robots.txt not found/parseable
    """
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    try:
        response = httpx.get(robots_url, timeout=timeout, follow_redirects=True)
        if response.status_code == 200:
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser
    except Exception as e:
        logger.debug(f"Could not fetch robots.txt: {e}")

    return None


def can_fetch(robots: RobotFileParser | None, url: str) -> bool:
    """Check if URL can be fetched according to robots.txt."""
    if robots is None:
        return True
    return robots.can_fetch(USER_AGENT, url)


def extract_content(html: str, url: str) -> PageResult:
    """Extract content from HTML and convert to markdown.

    Args:
        html: Raw HTML content
        url: URL of the page (for resolving relative links)

    Returns:
        PageResult with extracted content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Remove unwanted elements
    for selector in REMOVE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()

    # Find main content area
    content_element = None
    for selector in CONTENT_SELECTORS:
        content_element = soup.select_one(selector)
        if content_element:
            break

    # Fall back to body if no content area found
    if content_element is None:
        content_element = soup.find("body")

    if content_element is None:
        return PageResult(url=url, title=title, content="", error="No content found")

    # Extract links before converting to markdown
    links = []
    for a_tag in content_element.find_all("a", href=True):
        href = a_tag["href"]
        # Resolve relative URLs
        full_url = urljoin(url, href)
        # Only include http(s) links
        if full_url.startswith(("http://", "https://")):
            links.append(full_url)

    # Convert to markdown
    content = md(str(content_element), heading_style="ATX", strip=["img"])

    # Clean up the markdown
    content = re.sub(r"\n{3,}", "\n\n", content)  # Remove excessive newlines
    content = content.strip()

    return PageResult(url=url, title=title, content=content, links=links)


def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are on the same domain."""
    parsed1 = urlparse(url1)
    parsed2 = urlparse(url2)
    return parsed1.netloc == parsed2.netloc


def is_doc_page(url: str) -> bool:
    """Check if URL looks like a documentation page."""
    url_lower = url.lower()

    # Skip non-doc URLs
    skip_patterns = [
        "/blog/",
        "/news/",
        "/press/",
        "/about/",
        "/contact/",
        "/login",
        "/signup",
        "/register",
        "/account/",
        "/cart",
        "/checkout",
        "/search",
        ".pdf",
        ".zip",
        ".png",
        ".jpg",
        ".gif",
        ".svg",
        ".css",
        ".js",
    ]
    for pattern in skip_patterns:
        if pattern in url_lower:
            return False

    return True


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and trailing slashes."""
    parsed = urlparse(url)
    # Remove fragment
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # Remove trailing slash (but keep root /)
    if normalized.endswith("/") and len(parsed.path) > 1:
        normalized = normalized[:-1]
    return normalized


class WebsiteCrawler(Crawler):
    """Crawler for documentation websites.

    Converts HTML pages to markdown and respects robots.txt.
    """

    def __init__(self, config: CrawlConfig | None = None):
        """Initialize the website crawler.

        Args:
            config: Crawl configuration (uses defaults if not provided)
        """
        self.config = config or CrawlConfig()

    @property
    def source_type(self) -> str:
        return "website"

    def can_handle(self, url: str) -> bool:
        """Check if URL is a website (http/https)."""
        return url.lower().startswith(("http://", "https://"))

    def crawl(
        self,
        url: str,
        cache_dir: Path,
        on_progress: callable | None = None,
    ) -> CrawlResult:
        """Crawl a documentation website.

        Args:
            url: Starting URL for crawl
            cache_dir: Directory to save converted markdown files
            on_progress: Optional callback(pages_crawled, max_pages) for progress updates

        Returns:
            CrawlResult with status and list of files
        """
        start_url = normalize_url(url)
        parsed = urlparse(start_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        logger.info(f"Crawling website: {start_url}")

        # Fetch robots.txt if configured
        robots = None
        if self.config.respect_robots:
            robots = get_robots_parser(base_url, timeout=self.config.timeout)
            if robots:
                logger.info("Loaded robots.txt")

        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Track visited URLs and queue
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_url, 0)]  # (url, depth)
        files_crawled: list[Path] = []

        # HTTP client
        client = httpx.Client(
            timeout=self.config.timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

        try:
            while queue and len(files_crawled) < self.config.max_pages:
                current_url, depth = queue.pop(0)
                normalized = normalize_url(current_url)

                # Skip if already visited
                if normalized in visited:
                    continue
                visited.add(normalized)

                # Check robots.txt
                if not can_fetch(robots, current_url):
                    logger.debug(f"Blocked by robots.txt: {current_url}")
                    continue

                # Check if it's a doc page
                if not is_doc_page(current_url):
                    logger.debug(f"Skipping non-doc URL: {current_url}")
                    continue

                # Rate limiting
                if files_crawled:
                    time.sleep(self.config.rate_limit)

                # Fetch page
                try:
                    logger.debug(f"Fetching: {current_url}")
                    response = client.get(current_url)
                    response.raise_for_status()

                    # Only process HTML
                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        continue

                    # Extract content
                    result = extract_content(response.text, current_url)
                    if result.error or not result.content:
                        continue

                    # Generate filename from URL path
                    path = parsed.path.strip("/") or "index"
                    if urlparse(current_url).path.strip("/"):
                        path = urlparse(current_url).path.strip("/")
                    filename = path.replace("/", "_") + ".md"

                    # Add title as header if not already present
                    content = result.content
                    if result.title and not content.startswith("#"):
                        content = f"# {result.title}\n\n{content}"

                    # Save markdown file
                    file_path = cache_dir / filename
                    file_path.write_text(content, encoding="utf-8")
                    files_crawled.append(file_path)
                    logger.debug(f"Saved: {filename}")

                    # Report progress
                    if on_progress:
                        on_progress(len(files_crawled), self.config.max_pages)

                    # Add links to queue if within depth limit
                    if depth < self.config.max_depth:
                        for link in result.links:
                            link_normalized = normalize_url(link)
                            if link_normalized in visited:
                                continue
                            # Only follow same-domain links unless configured otherwise
                            if not self.config.follow_external:
                                if not is_same_domain(start_url, link):
                                    continue
                            queue.append((link, depth + 1))

                except httpx.HTTPStatusError as e:
                    logger.debug(f"HTTP error for {current_url}: {e.response.status_code}")
                except httpx.RequestError as e:
                    logger.debug(f"Request error for {current_url}: {e}")

            logger.info(f"Crawled {len(files_crawled)} pages")

        finally:
            client.close()

        return CrawlResult(
            source_url=url,
            source_type=self.source_type,
            cache_dir=cache_dir,
            files_crawled=files_crawled,
        )
