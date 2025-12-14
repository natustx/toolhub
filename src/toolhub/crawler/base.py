"""Abstract base class for documentation crawlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrawlResult:
    """Result of crawling a documentation source."""

    source_url: str
    source_type: str
    cache_dir: Path
    files_crawled: list[Path]
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class Crawler(ABC):
    """Abstract base class for documentation crawlers.

    Subclasses implement crawling for specific source types:
    - GitHub repos
    - llms.txt files
    - Documentation websites
    - OpenAPI specs
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier (e.g., 'github', 'llmstxt')."""
        ...

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this crawler can handle the given URL."""
        ...

    @abstractmethod
    def crawl(self, url: str, cache_dir: Path) -> CrawlResult:
        """Crawl the source and save files to cache_dir.

        Args:
            url: The source URL to crawl
            cache_dir: Directory to save crawled files to

        Returns:
            CrawlResult with crawl status and list of files
        """
        ...


def detect_source_type(url: str) -> str | None:
    """Detect the source type from a URL.

    Returns source type string or None if unknown.
    """
    url_lower = url.lower()

    if "github.com" in url_lower or "gitlab.com" in url_lower:
        return "github"

    if url_lower.endswith("llms.txt") or "/llms.txt" in url_lower:
        return "llmstxt"

    if url_lower.endswith((".yaml", ".yml", ".json")):
        # Could be OpenAPI spec
        if any(x in url_lower for x in ["openapi", "swagger"]):
            return "openapi"

    # Default to website for http(s) URLs
    if url_lower.startswith(("http://", "https://")):
        return "website"

    return None
